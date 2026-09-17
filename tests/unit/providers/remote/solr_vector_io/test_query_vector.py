"""Unit tests for Solr vector query construction."""

from typing import Any

import numpy as np
import pytest
from ogx_api.vector_stores import VectorStore as VectorDB

# pylint: disable=line-too-long
from lightspeed_stack_providers.providers.remote.solr_vector_io.solr_vector_io.src.solr_vector_io.solr import (
    SolrIndex,
)


class FakeResponse:
    """Minimal successful Solr response used by the query test."""

    def raise_for_status(self) -> None:
        """Represent a successful HTTP response."""

    def json(self) -> dict[str, Any]:
        """Return an empty Solr result set."""
        return {"response": {"docs": []}}


class FakeHttpClient:
    """Capture the request issued by SolrIndex."""

    def __init__(self) -> None:
        """Initialize the captured request fields."""
        self.url = ""
        self.data: dict[str, Any] = {}

    async def __aenter__(self) -> "FakeHttpClient":
        """Enter the async client context."""
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Exit the async client context."""

    async def post(self, url: str, *, data: dict[str, Any]) -> FakeResponse:
        """Capture a form-encoded POST and return a successful response."""
        self.url = url
        self.data = data
        return FakeResponse()


@pytest.fixture(name="solr_index")
def solr_index_fixture() -> SolrIndex:
    """Create a Solr index without connecting to a server."""
    embedding_model = "ibm-granite/granite-embedding-30m-english"
    vector_store = VectorDB(
        identifier="test-store",
        embedding_dimension=3,
        embedding_model=embedding_model,
        provider_id="solr",
    )
    return SolrIndex(
        vector_store=vector_store,
        solr_url="http://localhost:8983/solr",
        collection_name="portal-rag",
        vector_field="chunk_vector",
        content_field="chunk",
        id_field="id",
        dimension=3,
        embedding_model=embedding_model,
    )


@pytest.mark.asyncio
async def test_query_vector_sends_knn_expression_in_q(
    solr_index: SolrIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the SearchHandler receives an executable KNN query."""
    client = FakeHttpClient()
    monkeypatch.setattr(solr_index, "_create_http_client", lambda: client)

    result = await solr_index.query_vector(
        embedding=np.array([0.1, 0.2, 0.3]),
        k=5,
        score_threshold=0.0,
    )

    assert result.chunks == []
    assert client.url == "http://localhost:8983/solr/portal-rag/semantic-search"
    assert client.data == {
        "q": "{!knn f=chunk_vector topK=5}[0.1,0.2,0.3]",
        "rows": 5,
        "fl": "*,score",
        "wt": "json",
    }
