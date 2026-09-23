"""
Unit tests for SolrVectorIOAdapter.query_chunks graceful degradation.

When Solr/OKP is unreachable or times out, query_chunks returns an empty
QueryChunksResponse instead of propagating httpx errors to OGX.
"""

import httpx
import pytest
from ogx_api.vector_io import QueryChunksRequest, QueryChunksResponse
from pytest_mock import MockerFixture

# pylint: disable=line-too-long
from lightspeed_stack_providers.providers.remote.solr_vector_io.solr_vector_io.src.solr_vector_io.config import (
    SolrVectorIOConfig,
)
from lightspeed_stack_providers.providers.remote.solr_vector_io.solr_vector_io.src.solr_vector_io.solr import (
    SolrVectorIOAdapter,
)

EMBEDDING_DIM = 384
EMBEDDING_MODEL = "ibm-granite/granite-embedding-30m-english"
VECTOR_STORE_ID = "test-store"
EMPTY_RESPONSE = QueryChunksResponse(chunks=[], scores=[])


@pytest.fixture(name="adapter")
def adapter_fixture(mocker: MockerFixture) -> SolrVectorIOAdapter:
    """
    Create a SolrVectorIOAdapter without initializing Solr connections.

    Parameters:
        mocker: pytest-mock fixture for creating async mocks.

    Returns:
        SolrVectorIOAdapter: Adapter configured for unit tests with no
        persistence and a minimal Solr connection config.
    """
    config = SolrVectorIOConfig(
        solr_url="http://localhost:8983/solr",
        collection_name="test",
        vector_field="chunk_vector",
        content_field="chunk",
        embedding_dimension=EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
        persistence=None,
    )
    return SolrVectorIOAdapter(config=config, inference_api=mocker.AsyncMock())


@pytest.fixture(name="query_request")
def query_request_fixture() -> QueryChunksRequest:
    """
    Build a minimal QueryChunksRequest for degradation tests.

    Returns:
        QueryChunksRequest: Request targeting the test vector store with a
        simple keyword query.
    """
    return QueryChunksRequest(
        vector_store_id=VECTOR_STORE_ID,
        query="test query",
        params={"max_chunks": 5},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transport_error",
    [
        pytest.param(httpx.ReadTimeout("timed out"), id="read_timeout"),
        pytest.param(httpx.ConnectError("connection refused"), id="connect_error"),
    ],
)
async def test_query_chunks_returns_empty_on_transport_error(
    adapter: SolrVectorIOAdapter,
    query_request: QueryChunksRequest,
    mocker: MockerFixture,
    transport_error: httpx.TransportError,
) -> None:
    """
    Transport errors from Solr should yield an empty QueryChunksResponse.

    Parameters:
        adapter: SolrVectorIOAdapter under test.
        query_request: Minimal query request fixture.
        mocker: pytest-mock fixture for patching index lookup.
        transport_error: httpx error raised by the Solr index query.
    """
    mock_index = mocker.MagicMock()
    mock_index.query_chunks = mocker.AsyncMock(side_effect=transport_error)
    mocker.patch.object(
        adapter,
        "_get_and_cache_vector_store_index",
        mocker.AsyncMock(return_value=mock_index),
    )

    result = await adapter.query_chunks(query_request)

    assert result == EMPTY_RESPONSE


@pytest.mark.asyncio
async def test_query_chunks_delegates_to_vector_store_with_index(
    adapter: SolrVectorIOAdapter,
    query_request: QueryChunksRequest,
    mocker: MockerFixture,
) -> None:
    """query_chunks should delegate to VectorStoreWithIndex.query_chunks."""
    mock_index = mocker.MagicMock()
    mock_index.query_chunks = mocker.AsyncMock(return_value=EMPTY_RESPONSE)
    mock_get = mocker.patch.object(
        adapter,
        "_get_and_cache_vector_store_index",
        mocker.AsyncMock(return_value=mock_index),
    )

    result = await adapter.query_chunks(query_request)

    mock_get.assert_awaited_once_with(VECTOR_STORE_ID)
    mock_index.query_chunks.assert_awaited_once_with(query_request)
    assert result == EMPTY_RESPONSE
