"""
Unit tests for SolrIndex._doc_to_chunk.

Tests the chunk-building logic directly without requiring a running Solr
instance.
"""

import pytest
from llama_stack_api.vector_stores import VectorStore as VectorDB

from lightspeed_stack_providers.providers.remote.solr_vector_io.solr_vector_io.src.solr_vector_io.config import (
    ChunkWindowConfig,
)
from lightspeed_stack_providers.providers.remote.solr_vector_io.solr_vector_io.src.solr_vector_io.solr import (
    OKP_SOURCE,
    SolrIndex,
)

EMBEDDING_DIM = 384
EMBEDDING_MODEL = "ibm-granite/granite-embedding-30m-english"


@pytest.fixture
def chunk_window_config():
    return ChunkWindowConfig(
        chunk_parent_id_field="parent_id",
        chunk_index_field="chunk_index",
        chunk_content_field="chunk",
        chunk_token_count_field="num_tokens",
        parent_total_chunks_field="total_chunks",
        parent_total_tokens_field="total_tokens",
        parent_content_id_field="doc_id",
        parent_content_title_field="title",
        parent_content_url_field="reference_url",
    )


@pytest.fixture
def solr_index(chunk_window_config):
    """SolrIndex created without connecting to Solr (initialize() not called)."""
    vector_store = VectorDB(
        identifier="test-store",
        embedding_dimension=EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
        provider_id="solr",
    )
    return SolrIndex(
        vector_store=vector_store,
        solr_url="http://localhost:8983/solr",
        collection_name="test",
        vector_field="chunk_vector",
        content_field="chunk",
        id_field="id",
        dimension=EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
        chunk_window_config=chunk_window_config,
    )


def _basic_doc(**extra):
    return {"id": "doc_chunk_0", "chunk": "Test content.", "parent_id": "doc", **extra}


class TestMetadataFields:
    def test_source_present(self, solr_index):
        chunk = solr_index._doc_to_chunk(_basic_doc())
        assert chunk is not None
        assert chunk.metadata["source"] == OKP_SOURCE
        assert chunk.chunk_metadata.source == OKP_SOURCE

    def test_metadata_and_chunk_metadata_both_set(self, solr_index):
        chunk = solr_index._doc_to_chunk(_basic_doc())
        assert chunk is not None
        assert chunk.metadata is not None
        assert chunk.chunk_metadata is not None

    def test_document_id_in_metadata(self, solr_index):
        chunk = solr_index._doc_to_chunk(_basic_doc())
        assert chunk is not None
        assert chunk.metadata["document_id"] == "doc"
        assert chunk.metadata["doc_id"] == "doc"

    def test_chunk_id_in_metadata(self, solr_index):
        chunk = solr_index._doc_to_chunk(_basic_doc())
        assert chunk is not None
        assert chunk.metadata["chunk_id"] == "doc_chunk_0"


class TestOptionalFields:
    def test_title_included_when_present(self, solr_index):
        chunk = solr_index._doc_to_chunk(_basic_doc(title="My Title"))
        assert chunk is not None
        assert chunk.metadata["title"] == "My Title"

    def test_title_absent_when_missing(self, solr_index):
        chunk = solr_index._doc_to_chunk(_basic_doc())
        assert chunk is not None
        assert "title" not in chunk.metadata

    def test_reference_url_included_when_present(self, solr_index):
        chunk = solr_index._doc_to_chunk(
            _basic_doc(reference_url="https://example.com/doc")
        )
        assert chunk is not None
        assert chunk.metadata["reference_url"] == "https://example.com/doc"

    def test_token_count_included_when_present(self, solr_index):
        chunk = solr_index._doc_to_chunk(_basic_doc(num_tokens=42))
        assert chunk is not None
        assert chunk.metadata["num_tokens"] == 42

    def test_chunk_index_included_when_present(self, solr_index):
        chunk = solr_index._doc_to_chunk(_basic_doc(chunk_index=3))
        assert chunk is not None
        assert chunk.metadata["chunk_index"] == 3

    def test_resource_name_used_as_chunk_id(self, solr_index):
        doc = {"resourceName": "res_chunk_1", "chunk": "Content.", "parent_id": "res"}
        chunk = solr_index._doc_to_chunk(doc)
        assert chunk is not None
        assert chunk.chunk_id == "res_chunk_1"

    def test_parent_id_derived_from_chunk_id_pattern(self, solr_index):
        doc = {"id": "mydoc_chunk_2", "chunk": "Content."}
        chunk = solr_index._doc_to_chunk(doc)
        assert chunk is not None
        assert chunk.metadata["document_id"] == "mydoc"


class TestGuardClauses:
    def test_missing_content_returns_none(self, solr_index):
        chunk = solr_index._doc_to_chunk({"id": "doc_chunk_0", "parent_id": "doc"})
        assert chunk is None

    def test_non_chunk_doc_returns_none(self, solr_index):
        chunk = solr_index._doc_to_chunk(_basic_doc(is_chunk=False))
        assert chunk is None

    def test_missing_chunk_id_returns_none(self, solr_index):
        chunk = solr_index._doc_to_chunk({"chunk": "Content."})
        assert chunk is None
