"""
Unit tests for the private helper methods extracted from _apply_chunk_window_expansion.

Covers _get_chunk_boundary_and_budget, _assemble_expanded_chunk, and
_select_context_chunks_in_window without requiring a running Solr instance.
"""

from typing import Any

from unittest.mock import AsyncMock

import pytest
from llama_stack_api.vector_io import EmbeddedChunk
from llama_stack_api.vector_stores import VectorStore as VectorDB
from pytest_mock import MockerFixture

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
    Create a ChunkWindowConfig with all optional fields populated.

    Returns:
        ChunkWindowConfig: Configuration used across chunk window expansion tests.
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
        family_token_budget=3072,
        orphan_token_budget=1536,
    )


@pytest.fixture(name="chunk_window_config_with_family")
def chunk_window_config_with_family_fixture(
    chunk_window_config: ChunkWindowConfig,
) -> ChunkWindowConfig:
    """
    Return a ChunkWindowConfig with chunk_family_fields set to ['heading'].

    Parameters:
        chunk_window_config (ChunkWindowConfig): Base configuration to extend.

    Returns:
        ChunkWindowConfig: Configuration with heading as a family field.
    """
    return chunk_window_config.model_copy(update={"chunk_family_fields": ["heading"]})


@pytest.fixture(name="solr_index")
def solr_index_fixture(chunk_window_config: ChunkWindowConfig) -> SolrIndex:
    """
    Return a SolrIndex configured for tests without calling initialize().

    Parameters:
        chunk_window_config (ChunkWindowConfig): Configuration injected into the index.

    Returns:
        SolrIndex: Test instance with no Solr connection required.
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


@pytest.fixture(name="solr_index_with_family")
def solr_index_with_family_fixture(
    chunk_window_config_with_family: ChunkWindowConfig,
) -> SolrIndex:
    """
    Return a SolrIndex with chunk_family_fields=['heading'].

    Parameters:
        chunk_window_config_with_family (ChunkWindowConfig): Config with family fields set.

    Returns:
        SolrIndex: Test instance configured with family field support.
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
        chunk_window_config=chunk_window_config_with_family,
    )


def _make_chunk(metadata: dict[str, Any]) -> EmbeddedChunk:
    """
    Return a minimal EmbeddedChunk populated with the given metadata.

    Parameters:
        metadata (dict): Metadata dict to attach to the chunk.

    Returns:
        EmbeddedChunk: Test chunk with fixed chunk_id and content.
    """
    return EmbeddedChunk(
        chunk_id="doc_chunk_0",
        content="original content",
        metadata=metadata,
        chunk_metadata=metadata,
        embedding=[],
        embedding_model=EMBEDDING_MODEL,
        embedding_dimension=EMBEDDING_DIM,
        metadata_token_count=None,
    )


# pylint: disable=protected-access
class TestGetChunkBoundaryAndBudget:
    """Tests for _get_chunk_boundary_and_budget."""

    def test_no_family_fields_returns_none_boundary_and_family_budget(
        self, solr_index: SolrIndex
    ) -> None:
        """Test that without chunk_family_fields, boundary is None and family budget is used."""
        chunk = _make_chunk({"parent_id": "doc1", "chunk_index": 2})
        assert solr_index.chunk_window_config is not None
        boundary, budget, is_orphan = solr_index._get_chunk_boundary_and_budget(
            chunk, solr_index.chunk_window_config
        )
        assert boundary is None
        assert budget == 3072
        assert is_orphan is False

    def test_family_chunk_returns_boundary_values_and_family_budget(
        self, solr_index_with_family: SolrIndex
    ) -> None:
        """Test that a chunk with a family field value gets its boundary dict and family budget."""
        chunk = _make_chunk({"parent_id": "doc1", "chunk_index": 2, "heading": "Intro"})
        schema = solr_index_with_family.chunk_window_config
        assert schema is not None
        boundary, budget, is_orphan = (
            solr_index_with_family._get_chunk_boundary_and_budget(chunk, schema)
        )
        assert boundary == {"heading": "Intro"}
        assert budget == 3072
        assert is_orphan is False

    def test_orphan_chunk_uses_orphan_budget(
        self, solr_index_with_family: SolrIndex
    ) -> None:
        """Test that a chunk missing all family field values is flagged as orphan and gets orphan budget."""
        chunk = _make_chunk({"parent_id": "doc1", "chunk_index": 5})
        schema = solr_index_with_family.chunk_window_config
        assert schema is not None
        boundary, budget, is_orphan = (
            solr_index_with_family._get_chunk_boundary_and_budget(chunk, schema)
        )
        assert boundary == {}
        assert budget == 1536
        assert is_orphan is True

    def test_chunk_with_any_family_field_value_is_not_orphan(
        self, solr_index_with_family: SolrIndex
    ) -> None:
        """Test that a chunk with at least one family field value is not treated as an orphan."""
        chunk = _make_chunk({"parent_id": "doc1", "chunk_index": 2, "heading": "Ch1"})
        schema = solr_index_with_family.chunk_window_config
        assert schema is not None
        _, _, is_orphan = solr_index_with_family._get_chunk_boundary_and_budget(
            chunk, schema
        )
        assert is_orphan is False


class TestAssembleExpandedChunk:
    """Tests for _assemble_expanded_chunk."""

    def test_content_joined_with_double_newline(self, solr_index: SolrIndex) -> None:
        """Test that selected chunk content is concatenated with double newlines."""
        chunk = _make_chunk({"parent_id": "doc1", "chunk_index": 1})
        selected = [
            {"chunk": "First.", "num_tokens": 5},
            {"chunk": "Second.", "num_tokens": 5},
        ]
        assert solr_index.chunk_window_config is not None
        result = solr_index._assemble_expanded_chunk(
            chunk, selected, {}, solr_index.chunk_window_config, matched_chunk_index=1
        )
        assert result.content == "First.\n\nSecond."

    def test_empty_content_chunks_are_skipped(self, solr_index: SolrIndex) -> None:
        """Test that chunks with an empty content field are excluded from the joined output."""
        chunk = _make_chunk({"parent_id": "doc1", "chunk_index": 0})
        selected = [
            {"chunk": "", "num_tokens": 0},
            {"chunk": "Only this.", "num_tokens": 5},
        ]
        assert solr_index.chunk_window_config is not None
        result = solr_index._assemble_expanded_chunk(
            chunk, selected, {}, solr_index.chunk_window_config, matched_chunk_index=0
        )
        assert result.content == "Only this."

    def test_expansion_metadata_flags_set(self, solr_index: SolrIndex) -> None:
        """Test that chunk_window_expanded, chunk_window_size, and matched_chunk_index are written."""
        chunk = _make_chunk({"parent_id": "doc1", "chunk_index": 2})
        selected = [{"chunk": "a", "num_tokens": 5}, {"chunk": "b", "num_tokens": 5}]
        assert solr_index.chunk_window_config is not None
        result = solr_index._assemble_expanded_chunk(
            chunk, selected, {}, solr_index.chunk_window_config, matched_chunk_index=2
        )
        assert result.metadata["chunk_window_expanded"] is True
        assert result.metadata["chunk_window_size"] == 2
        assert result.metadata["matched_chunk_index"] == 2

    def test_parent_doc_id_written_to_metadata(self, solr_index: SolrIndex) -> None:
        """Test that doc_id from the parent document is included when the field is configured."""
        chunk = _make_chunk({"parent_id": "doc1", "chunk_index": 0})
        assert solr_index.chunk_window_config is not None
        result = solr_index._assemble_expanded_chunk(
            chunk,
            [{"chunk": "x", "num_tokens": 5}],
            {"doc_id": "content-abc", "title": "My Doc"},
            solr_index.chunk_window_config,
            matched_chunk_index=0,
        )
        assert result.metadata["doc_id"] == "content-abc"

    def test_parent_title_written_to_metadata(self, solr_index: SolrIndex) -> None:
        """Test that title from the parent document is included when the field is configured."""
        chunk = _make_chunk({"parent_id": "doc1", "chunk_index": 0})
        assert solr_index.chunk_window_config is not None
        result = solr_index._assemble_expanded_chunk(
            chunk,
            [{"chunk": "x", "num_tokens": 5}],
            {"doc_id": "content-abc", "title": "My Doc"},
            solr_index.chunk_window_config,
            matched_chunk_index=0,
        )
        assert result.metadata["title"] == "My Doc"

    def test_source_always_set_to_okp(self, solr_index: SolrIndex) -> None:
        """Test that source is always set to OKP_SOURCE regardless of parent doc content."""
        chunk = _make_chunk({"parent_id": "doc1", "chunk_index": 0})
        assert solr_index.chunk_window_config is not None
        result = solr_index._assemble_expanded_chunk(
            chunk,
            [{"chunk": "x", "num_tokens": 5}],
            {},
            solr_index.chunk_window_config,
            matched_chunk_index=0,
        )
        assert result.metadata["source"] == OKP_SOURCE

    def test_chunk_id_preserved_from_original(self, solr_index: SolrIndex) -> None:
        """Test that chunk_id on the result matches the original chunk's chunk_id."""
        chunk = _make_chunk({"parent_id": "doc1", "chunk_index": 0})
        assert solr_index.chunk_window_config is not None
        result = solr_index._assemble_expanded_chunk(
            chunk,
            [{"chunk": "x", "num_tokens": 5}],
            {},
            solr_index.chunk_window_config,
            matched_chunk_index=0,
        )
        assert result.chunk_id == "doc_chunk_0"


class TestSelectContextChunksInWindow:
    """Tests for _select_context_chunks_in_window (HTTP calls mocked via mocker)."""

    @pytest.mark.asyncio
    async def test_short_doc_fetches_all_chunks(
        self, solr_index: SolrIndex, mocker: MockerFixture
    ) -> None:
        """Test that total_chunks < min_chunk_window triggers a full fetch of all chunks."""
        all_chunks = [
            {"chunk_index": i, "chunk": f"c{i}", "num_tokens": 10} for i in range(3)
        ]
        mocker.patch.object(
            solr_index, "_fetch_context_chunks", AsyncMock(return_value=all_chunks)
        )
        assert solr_index.chunk_window_config is not None
        result = await solr_index._select_context_chunks_in_window(
            client=mocker.MagicMock(),
            parent_id="doc1",
            matched_chunk_index=1,
            total_chunks=3,
            total_tokens=30,
            token_budget=3072,
            boundary_values=None,
            schema=solr_index.chunk_window_config,
            min_chunk_window=4,
        )
        assert result == all_chunks

    @pytest.mark.asyncio
    async def test_window_fitting_budget_returned_directly(
        self, solr_index: SolrIndex, mocker: MockerFixture
    ) -> None:
        """Test that context chunks fitting within the token budget are returned without expansion."""
        window = [
            {"chunk_index": i, "chunk": f"c{i}", "num_tokens": 50} for i in range(5)
        ]
        mocker.patch.object(
            solr_index, "_fetch_context_chunks", AsyncMock(return_value=window)
        )
        assert solr_index.chunk_window_config is not None
        result = await solr_index._select_context_chunks_in_window(
            client=mocker.MagicMock(),
            parent_id="doc1",
            matched_chunk_index=10,
            total_chunks=100,
            total_tokens=5000,
            token_budget=3072,
            boundary_values=None,
            schema=solr_index.chunk_window_config,
            min_chunk_window=4,
        )
        assert result == window

    @pytest.mark.asyncio
    async def test_over_budget_delegates_to_expand_chunk_window(
        self, solr_index: SolrIndex, mocker: MockerFixture
    ) -> None:
        """Test that an over-budget window calls _expand_chunk_window and returns its result."""
        big_window = [
            {"chunk_index": i, "chunk": f"c{i}", "num_tokens": 1000} for i in range(5)
        ]
        mocker.patch.object(
            solr_index, "_fetch_context_chunks", AsyncMock(return_value=big_window)
        )
        mocker.patch.object(
            solr_index, "_expand_chunk_window", return_value=big_window[:2]
        )
        assert solr_index.chunk_window_config is not None
        result = await solr_index._select_context_chunks_in_window(
            client=mocker.MagicMock(),
            parent_id="doc1",
            matched_chunk_index=2,
            total_chunks=100,
            total_tokens=5000,
            token_budget=3072,
            boundary_values=None,
            schema=solr_index.chunk_window_config,
            min_chunk_window=4,
        )
        assert result == big_window[:2]

    @pytest.mark.asyncio
    async def test_empty_fetch_returns_none(
        self, solr_index: SolrIndex, mocker: MockerFixture
    ) -> None:
        """Test that an empty context chunk fetch returns None to signal fallback to original chunk."""
        mocker.patch.object(
            solr_index, "_fetch_context_chunks", AsyncMock(return_value=[])
        )
        assert solr_index.chunk_window_config is not None
        result = await solr_index._select_context_chunks_in_window(
            client=mocker.MagicMock(),
            parent_id="doc1",
            matched_chunk_index=10,
            total_chunks=100,
            total_tokens=5000,
            token_budget=3072,
            boundary_values=None,
            schema=solr_index.chunk_window_config,
            min_chunk_window=4,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_match_index_absent_from_window_returns_none(
        self, solr_index: SolrIndex, mocker: MockerFixture
    ) -> None:
        """Test that a matched chunk absent from the fetched window returns None to signal fallback."""
        chunks = [{"chunk_index": 99, "chunk": "x", "num_tokens": 1000}]
        mocker.patch.object(
            solr_index, "_fetch_context_chunks", AsyncMock(return_value=chunks)
        )
        assert solr_index.chunk_window_config is not None
        result = await solr_index._select_context_chunks_in_window(
            client=mocker.MagicMock(),
            parent_id="doc1",
            matched_chunk_index=2,
            total_chunks=100,
            total_tokens=5000,
            token_budget=500,
            boundary_values=None,
            schema=solr_index.chunk_window_config,
            min_chunk_window=4,
        )
        assert result is None
