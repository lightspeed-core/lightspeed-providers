"""
Unit tests for SolrIndex._doc_to_chunk.

Tests the chunk-building logic directly without requiring a running Solr
instance.
"""

from typing import Any

import pytest
from llama_stack_api.vector_stores import VectorStore as VectorDB

# pylint: disable=line-too-long
from lightspeed_stack_providers.providers.remote.solr_vector_io.solr_vector_io.src.solr_vector_io.config import (
    ChunkWindowConfig,
)

# pylint: disable=line-too-long
from lightspeed_stack_providers.providers.remote.solr_vector_io.solr_vector_io.src.solr_vector_io.solr import (
    OKP_SOURCE,
    SolrIndex,
)

EMBEDDING_DIM = 384
EMBEDDING_MODEL = "ibm-granite/granite-embedding-30m-english"


@pytest.fixture(name="chunk_window_config")
def chunk_window_config_fixture() -> ChunkWindowConfig:
    """
    Create a ChunkWindowConfig with explicit field names used by the tests.

    Returns:
        ChunkWindowConfig: Configuration mapping chunk fields (content, index, token count)
        and parent document fields (id, title, url, totals) to the names expected
        by the SolrIndex test fixtures.
    """
    return ChunkWindowConfig(
        chunk_parent_id_field="parent_id",
        chunk_index_field="chunk_index",
        chunk_content_field="chunk",
        chunk_token_count_field="num_tokens",
        chunk_online_source_url_field="online_source_url",
        chunk_source_path_field="source_path",
        parent_total_chunks_field="total_chunks",
        parent_total_tokens_field="total_tokens",
        parent_content_id_field="doc_id",
        parent_content_title_field="title",
    )


@pytest.fixture(name="solr_index")
def solr_index_fixture(chunk_window_config: ChunkWindowConfig) -> SolrIndex:
    """
    Create a SolrIndex configured for tests without calling initialize().

    Parameters:
        chunk_window_config (ChunkWindowConfig): Configuration that maps parent
        document and chunk field names used by the index.

    Returns:
        SolrIndex: A SolrIndex instance pointing to a local test Solr
        collection and using the provided chunk window configuration; the
        instance is not initialized.
    """
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


def _basic_doc(**extra: Any) -> dict[str, Any]:
    """
    Return a baseline document dictionary representing a chunk, with sensible defaults.

    Parameters:
        **extra: Additional key-value pairs to merge into the returned
        document; provided keys override defaults.

    Returns:
        dict: Document containing "id" (default "doc_chunk_0"), "chunk"
        (default "Test content."), and "parent_id" (default "doc"), merged with
        any entries from `extra`.
    """
    return {"id": "doc_chunk_0", "chunk": "Test content.", "parent_id": "doc", **extra}


# pylint: disable=protected-access
class TestMetadataFields:
    """Class containing tests for metadata fields."""

    def test_source_present(self, solr_index: SolrIndex) -> None:
        """Test for source field."""
        chunk = solr_index._doc_to_chunk(_basic_doc())
        assert chunk is not None
        assert chunk.metadata["source"] == OKP_SOURCE
        assert chunk.chunk_metadata.source == OKP_SOURCE

    def test_metadata_and_chunk_metadata_both_set(self, solr_index: SolrIndex) -> None:
        """Test for chunk metadata field."""
        chunk = solr_index._doc_to_chunk(_basic_doc())
        assert chunk is not None
        assert chunk.metadata is not None
        assert chunk.chunk_metadata is not None

    def test_document_id_in_metadata(self, solr_index: SolrIndex) -> None:
        """Test for document_id and doc_id fields."""
        chunk = solr_index._doc_to_chunk(_basic_doc())
        assert chunk is not None
        assert chunk.metadata["document_id"] == "doc"
        assert chunk.metadata["doc_id"] == "doc"

    def test_chunk_id_in_metadata(self, solr_index: SolrIndex) -> None:
        """Test for chunk_id field."""
        chunk = solr_index._doc_to_chunk(_basic_doc())
        assert chunk is not None
        assert chunk.metadata["chunk_id"] == "doc_chunk_0"


# pylint: disable=protected-access
class TestOptionalFields:
    """Class containing tests for optional fields."""

    def test_title_included_when_present(self, solr_index: SolrIndex) -> None:
        """Test for title field."""
        chunk = solr_index._doc_to_chunk(_basic_doc(title="My Title"))
        assert chunk is not None
        assert chunk.metadata["title"] == "My Title"

    def test_title_absent_when_missing(self, solr_index: SolrIndex) -> None:
        """Test for title field."""
        chunk = solr_index._doc_to_chunk(_basic_doc())
        assert chunk is not None
        assert "title" not in chunk.metadata

    def test_online_source_url_mapped_to_reference_url(
        self, solr_index: SolrIndex
    ) -> None:
        """Test for reference_url field."""
        chunk = solr_index._doc_to_chunk(
            _basic_doc(online_source_url="https://example.com/doc#chunk1")
        )
        assert chunk is not None
        assert chunk.metadata["reference_url"] == "https://example.com/doc#chunk1"

    def test_source_path_included_when_present(self, solr_index: SolrIndex) -> None:
        """Test for source_path field."""
        chunk = solr_index._doc_to_chunk(
            _basic_doc(source_path="/docs/install.html#step-3")
        )
        assert chunk is not None
        assert chunk.metadata["source_path"] == "/docs/install.html#step-3"

    def test_token_count_included_when_present(self, solr_index: SolrIndex) -> None:
        """Test for num_tokens field."""
        chunk = solr_index._doc_to_chunk(_basic_doc(num_tokens=42))
        assert chunk is not None
        assert chunk.metadata["num_tokens"] == 42

    def test_chunk_index_included_when_present(self, solr_index: SolrIndex) -> None:
        """Test for chunk_index field."""
        chunk = solr_index._doc_to_chunk(_basic_doc(chunk_index=3))
        assert chunk is not None
        assert chunk.metadata["chunk_index"] == 3

    def test_resource_name_used_as_chunk_id(self, solr_index: SolrIndex) -> None:
        """Test for chunk_id field."""
        doc = {"resourceName": "res_chunk_1", "chunk": "Content.", "parent_id": "res"}
        chunk = solr_index._doc_to_chunk(doc)
        assert chunk is not None
        assert chunk.chunk_id == "res_chunk_1"

    def test_parent_id_derived_from_chunk_id_pattern(
        self, solr_index: SolrIndex
    ) -> None:
        """Test for document_id field."""
        doc = {"id": "mydoc_chunk_2", "chunk": "Content."}
        chunk = solr_index._doc_to_chunk(doc)
        assert chunk is not None
        assert chunk.metadata["document_id"] == "mydoc"


# pylint: disable=protected-access
class TestGuardClauses:
    """Tests for guard clauses."""

    def test_missing_content_returns_none(self, solr_index: SolrIndex) -> None:
        """Test is missing content is checked."""
        chunk = solr_index._doc_to_chunk({"id": "doc_chunk_0", "parent_id": "doc"})
        assert chunk is None

    def test_non_chunk_doc_returns_none(self, solr_index: SolrIndex) -> None:
        """Test is missing doc_to_chunk is checked."""
        chunk = solr_index._doc_to_chunk(_basic_doc(is_chunk=False))
        assert chunk is None

    def test_missing_chunk_id_returns_none(self, solr_index: SolrIndex) -> None:
        """Test is missing chunk_id is checked."""
        chunk = solr_index._doc_to_chunk({"chunk": "Content."})
        assert chunk is None
