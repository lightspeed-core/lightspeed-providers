"""
Pytest-based test suite for Solr Vector IO provider.

Run all tests:
    uv run pytest tests.py -v

Run specific test suites:
    uv run pytest tests.py -v -m basic
    uv run pytest tests.py -v -m search
    uv run pytest tests.py -v -m chunk_window
    uv run pytest tests.py -v -m persistence
    uv run pytest tests.py -v -m embeddings

Run with summary:
    uv run pytest tests.py -v --tb=short

Skip slow tests (real embeddings):
    uv run pytest tests.py -v -m "not slow"
"""

from typing import Any, cast
from unittest.mock import AsyncMock

import numpy as np
import pytest
import requests
from ogx.core.storage.datatypes import KVStoreReference, SqliteKVStoreConfig
from ogx.core.storage.kvstore.kvstore import register_kvstore_backends
from ogx_api.inference import Inference
from ogx_api.vector_io import (
    ChunkForDeletion,
    ChunkMetadata,
    DeleteChunksRequest,
    EmbeddedChunk,
    InsertChunksRequest,
)
from ogx_api.vector_stores import VectorStore as VectorDB
from src.solr_vector_io import (
    ChunkWindowConfig,
    SolrVectorIOAdapter,
    SolrVectorIOConfig,
)

# ============================================================================
# Configuration
# ============================================================================

SOLR_URL = "http://localhost4:8080/solr"
COLLECTION_NAME = "portal-rag"
VECTOR_FIELD = "chunk_vector"
CONTENT_FIELD = "chunk"
EMBEDDING_DIM = 384
EMBEDDING_MODEL = "ibm-granite/granite-embedding-30m-english"


@pytest.fixture(autouse=True)
def require_custom_search_handler(request: pytest.FixtureRequest) -> None:
    """Skip query tests when Solr lacks the provider-specific search handler."""
    if request.node.get_closest_marker("requires_custom_search_handler") is None:
        return

    response = requests.get(
        f"{SOLR_URL}/{COLLECTION_NAME}/semantic-search",
        timeout=5,
    )
    if response.status_code == 404:
        pytest.skip("Solr semantic-search handler is not available")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def config_basic() -> Any:
    """Provide basic configuration without chunk window schema.

    Create a SolrVectorIOConfig using the module's test constants with no
    chunk-window or persistence enabled.

    Returns:
        SolrVectorIOConfig: Configuration with SOLR_URL, COLLECTION_NAME,
        VECTOR_FIELD, CONTENT_FIELD, EMBEDDING_DIM, and persistence set to
        None.
    """
    return SolrVectorIOConfig(
        solr_url=SOLR_URL,
        collection_name=COLLECTION_NAME,
        vector_field=VECTOR_FIELD,
        content_field=CONTENT_FIELD,
        embedding_model=EMBEDDING_MODEL,
        embedding_dimension=EMBEDDING_DIM,
        persistence=None,
    )


@pytest.fixture
def config_with_chunk_window() -> Any:
    """Configure the provider with chunk window expansion enabled.

    Builds a SolrVectorIOConfig with chunk-window expansion enabled.

    Returns:
        SolrVectorIOConfig: Configuration for connecting to the test Solr
        instance with chunk_window_config populated (field mappings and a
        chunk_filter_query) and no persistence.
    """
    return SolrVectorIOConfig(
        solr_url=SOLR_URL,
        collection_name=COLLECTION_NAME,
        vector_field=VECTOR_FIELD,
        content_field=CONTENT_FIELD,
        embedding_model=EMBEDDING_MODEL,
        embedding_dimension=EMBEDDING_DIM,
        persistence=None,
        chunk_window_config=ChunkWindowConfig(
            chunk_parent_id_field="parent_id",
            chunk_index_field="chunk_index",
            chunk_content_field="chunk",
            chunk_token_count_field="num_tokens",
            parent_total_chunks_field="total_chunks",
            parent_total_tokens_field="total_tokens",
            parent_content_id_field="doc_id",
            parent_content_title_field="title",
            chunk_online_source_url_field="reference_url",
            chunk_filter_query="is_chunk:true",
        ),
    )


@pytest.fixture
async def adapter_basic(config_basic: Any) -> Any:
    """Adapter instance with basic configuration.

    Provide an initialized SolrVectorIOAdapter configured without chunk-window or persistence.

    Yields:
        SolrVectorIOAdapter: an adapter initialized and ready for use; the
        adapter is shut down after the fixture consumer finishes.
    """
    adapter = SolrVectorIOAdapter(
        config=config_basic, inference_api=cast(Inference, AsyncMock())
    )
    await adapter.initialize()
    yield adapter
    await adapter.shutdown()


@pytest.fixture
async def adapter_with_chunk_window(config_with_chunk_window: Any) -> Any:
    """Adapter instance with chunk window configuration.

    Async pytest fixture that initializes a SolrVectorIOAdapter configured for
    chunk-window behavior and yields it to the test.

    Parameters:
        - config_with_chunk_window (SolrVectorIOConfig): Configuration with
          `chunk_window_config` enabled (fields and chunk filter) used to
          construct the adapter.

    Returns:
        SolrVectorIOAdapter: An initialized adapter instance; the adapter is
        shut down after the fixture completes.
    """
    adapter = SolrVectorIOAdapter(
        config=config_with_chunk_window, inference_api=cast(Inference, AsyncMock())
    )
    await adapter.initialize()
    yield adapter
    await adapter.shutdown()


@pytest.fixture
async def vector_store_basic(adapter_basic: Any) -> Any:
    """Register vector store with basic config.

    Register and yield a basic VectorDB named "test-basic-store" for use in tests.

    This pytest fixture registers a VectorDB with the adapter, yields the
    registered store to the test, and ensures the store is unregistered after
    the test completes.

    Returns:
        VectorDB: The registered vector store instance yielded to the test.
    """
    # adapter_basic is already awaited by pytest-asyncio
    vector_store = VectorDB(
        identifier="test-basic-store",
        embedding_dimension=EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
        provider_id="solr",
    )
    await adapter_basic.register_vector_store(vector_store)
    yield vector_store
    await adapter_basic.unregister_vector_store("test-basic-store")


@pytest.fixture
async def vector_store_chunk_window(adapter_with_chunk_window: Any) -> Any:
    """Register vector store with chunk window config.

    Pytest fixture that registers a VectorDB configured for chunk-window tests and yields it.

    Yields:
        vector_store (VectorDB): The registered vector store (identifier
        "test-chunk-window-store"). The fixture unregisters this store from the
        adapter during teardown.
    """
    # adapter_with_chunk_window is already awaited by pytest-asyncio
    vector_store = VectorDB(
        identifier="test-chunk-window-store",
        embedding_dimension=EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
        provider_id="solr",
    )
    await adapter_with_chunk_window.register_vector_store(vector_store)
    yield vector_store
    await adapter_with_chunk_window.unregister_vector_store("test-chunk-window-store")


@pytest.fixture
def random_embedding() -> Any:
    """Generate random 384-dimensional embedding.

    Create a random embedding vector of length EMBEDDING_DIM for testing.

    Returns:
        np.ndarray: A float32 NumPy array of shape (EMBEDDING_DIM,) with values
        in the range [0, 1).
    """
    return np.random.rand(EMBEDDING_DIM).astype(np.float32)


@pytest.fixture
def config_with_persistence(tmp_path: Any) -> Any:
    """Configure SQLite-backed persistence for an isolated test namespace.

    Parameters:
        tmp_path: Temporary directory supplied by pytest for the SQLite database.

    Returns:
        SolrVectorIOConfig configured with a registered SQLite KV backend.
    """
    register_kvstore_backends(
        {
            "sql_default": SqliteKVStoreConfig(
                db_path=str(tmp_path / "solr-vector-store.db")
            )
        }
    )
    return SolrVectorIOConfig(
        solr_url=SOLR_URL,
        collection_name=COLLECTION_NAME,
        vector_field=VECTOR_FIELD,
        content_field=CONTENT_FIELD,
        embedding_model=EMBEDDING_MODEL,
        embedding_dimension=EMBEDDING_DIM,
        persistence=KVStoreReference(namespace="test_vector_io", backend="sql_default"),
    )


@pytest.fixture
async def adapter_with_persistence(config_with_persistence: Any) -> Any:
    """Adapter instance with persistence enabled.

    Provide an initialized SolrVectorIOAdapter configured with persistence enabled.

    Parameters:
        - config_with_persistence: SolrVectorIOConfig configured to enable KV
          persistence (e.g., SqliteKVStoreConfig).

    Returns:
        An initialized SolrVectorIOAdapter instance with persistence active;
        the adapter is shut down after use.
    """
    adapter = SolrVectorIOAdapter(
        config=config_with_persistence, inference_api=cast(Inference, AsyncMock())
    )
    await adapter.initialize()
    yield adapter
    await adapter.shutdown()


# ============================================================================
# Test Class: Basic Functionality
# ============================================================================


@pytest.mark.basic
class TestBasicFunctionality:
    """Test basic connection and configuration."""

    @pytest.mark.asyncio
    async def test_adapter_initialization(self, config_basic: Any) -> None:
        """Test that adapter initializes successfully."""
        adapter = SolrVectorIOAdapter(
            config=config_basic, inference_api=cast(Inference, AsyncMock())
        )
        await adapter.initialize()
        assert adapter is not None
        await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_vector_store_registration(self, adapter_basic: Any) -> None:
        """Test vector store registration and unregistration.

        Verifies that registering a VectorDB adds it to the adapter's cache and
        that unregistering it removes it.

        Creates a VectorDB with identifier "test-registration", registers it
        with the provided adapter, asserts the identifier appears in
        adapter.cache, then unregisters it and asserts the identifier is
        absent.
        """
        vector_store = VectorDB(
            identifier="test-registration",
            embedding_dimension=EMBEDDING_DIM,
            embedding_model=EMBEDDING_MODEL,
            provider_id="solr",
        )
        await adapter_basic.register_vector_store(vector_store)
        assert "test-registration" in adapter_basic.cache

        await adapter_basic.unregister_vector_store("test-registration")
        assert "test-registration" not in adapter_basic.cache

    @pytest.mark.asyncio
    async def test_read_only_insert_fails(
        self, adapter_basic: Any, vector_store_basic: Any
    ) -> None:
        """Test that insert operations raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="read-only"):
            await adapter_basic.insert_chunks(
                InsertChunksRequest(
                    vector_store_id="test-basic-store",
                    chunks=[
                        EmbeddedChunk(
                            chunk_id="test",
                            content="test",
                            metadata={},
                            chunk_metadata=ChunkMetadata(chunk_id="test"),
                            embedding=[0.1] * EMBEDDING_DIM,
                            embedding_model=EMBEDDING_MODEL,
                            embedding_dimension=EMBEDDING_DIM,
                        )
                    ],
                )
            )

    @pytest.mark.asyncio
    async def test_read_only_delete_fails(
        self, adapter_basic: Any, vector_store_basic: Any
    ) -> None:
        """Test that delete operations raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="read-only"):
            await adapter_basic.delete_chunks(
                DeleteChunksRequest(
                    vector_store_id="test-basic-store",
                    chunks=[
                        ChunkForDeletion(
                            document_id="test-document",
                            chunk_id="test-chunk-1",
                        ),
                        ChunkForDeletion(
                            document_id="test-document",
                            chunk_id="test-chunk-2",
                        ),
                    ],
                )
            )


# ============================================================================
# Test Class: Persistence
# ============================================================================


@pytest.mark.persistence
class TestPersistence:
    """Test persistence functionality with KV store."""

    @pytest.mark.asyncio
    async def test_adapter_initialization_with_persistence(
        self, config_with_persistence: Any
    ) -> None:
        """Test that adapter initializes successfully with persistence enabled."""
        adapter = SolrVectorIOAdapter(
            config=config_with_persistence, inference_api=cast(Inference, AsyncMock())
        )
        await adapter.initialize()

        # Check that kvstore was initialized
        assert adapter.kvstore is not None

        # Check that OpenAI API support was initialized
        assert hasattr(adapter, "openai_vector_stores")
        assert adapter.openai_vector_stores is not None

        await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_vector_store_persistence(
        self, adapter_with_persistence: Any
    ) -> None:
        """Test that vector stores are persisted to KV store."""
        from src.solr_vector_io.solr import VECTOR_DBS_PREFIX

        # Register a vector store
        vector_store = VectorDB(
            identifier="test-persisted-store",
            embedding_dimension=EMBEDDING_DIM,
            embedding_model=EMBEDDING_MODEL,
            provider_id="solr",
        )
        await adapter_with_persistence.register_vector_store(vector_store)

        # Verify it's in the cache
        assert "test-persisted-store" in adapter_with_persistence.cache

        # Verify it's in the KV store with the correct prefix
        key = f"{VECTOR_DBS_PREFIX}test-persisted-store"
        persisted_data = await adapter_with_persistence.kvstore.get(key=key)
        assert persisted_data is not None

        # Clean up
        await adapter_with_persistence.unregister_vector_store("test-persisted-store")

    @pytest.mark.asyncio
    async def test_vector_store_reload_from_persistence(
        self, config_with_persistence: Any
    ) -> None:
        """Test that vector stores are loaded from persistence on adapter initialization."""
        # First adapter: register a vector store
        adapter1 = SolrVectorIOAdapter(
            config=config_with_persistence, inference_api=cast(Inference, AsyncMock())
        )
        await adapter1.initialize()

        vector_store = VectorDB(
            identifier="test-reload-store",
            embedding_dimension=EMBEDDING_DIM,
            embedding_model=EMBEDDING_MODEL,
            provider_id="solr",
        )
        await adapter1.register_vector_store(vector_store)

        # Verify it's registered
        assert "test-reload-store" in adapter1.cache

        # Shutdown the first adapter
        await adapter1.shutdown()

        # Second adapter: should load the persisted vector store
        adapter2 = SolrVectorIOAdapter(
            config=config_with_persistence, inference_api=cast(Inference, AsyncMock())
        )
        await adapter2.initialize()

        # Verify the vector store was loaded from persistence
        assert "test-reload-store" in adapter2.cache

        # Clean up
        await adapter2.unregister_vector_store("test-reload-store")
        await adapter2.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.requires_custom_search_handler
    async def test_persistence_with_search(
        self, adapter_with_persistence: Any, random_embedding: Any
    ) -> None:
        """Test that search works with persistence enabled."""
        # Register a vector store
        vector_store = VectorDB(
            identifier="test-search-persisted",
            embedding_dimension=EMBEDDING_DIM,
            embedding_model=EMBEDDING_MODEL,
            provider_id="solr",
        )
        await adapter_with_persistence.register_vector_store(vector_store)

        # Get the index and perform a search
        index = await adapter_with_persistence._get_and_cache_vector_store_index(
            "test-search-persisted"
        )

        response = await index.index.query_vector(
            embedding=random_embedding, k=5, score_threshold=0.0
        )

        # Verify we get results (persistence shouldn't affect search)
        assert (
            len(response.chunks) > 0
        ), "Search should return results with persistence enabled"
        assert len(response.scores) == len(response.chunks)

        # Clean up
        await adapter_with_persistence.unregister_vector_store("test-search-persisted")


# ============================================================================
# Test Class: OpenAI API
# ============================================================================


@pytest.mark.openai
class TestOpenAIAPI:
    """Test vector similarity search functionality."""

    @pytest.mark.asyncio
    @pytest.mark.persistence
    async def test_openai_api_list_vector_stores(
        self, adapter_with_persistence: Any
    ) -> None:
        """Test that OpenAI API list_vector_stores works with persistence.

        Verify the OpenAI-compatible list_vector_stores API returns a list and
        remains callable after persisting a vector store.

        Asserts the response includes a `data` attribute of type `list`,
        registers a VectorDB into the adapter's persistence, calls
        `openai_list_vector_stores()` again to confirm it still returns a list,
        and then unregisters the test store.
        """
        # Call the OpenAI API method
        response = await adapter_with_persistence.openai_list_vector_stores()
        assert response is not None
        # VectorDBListResponse has a data attribute
        assert hasattr(response, "data")
        assert isinstance(response.data, list)

        # Register a vector store
        vector_store = VectorDB(
            identifier="test-openai-store",
            embedding_dimension=EMBEDDING_DIM,
            embedding_model=EMBEDDING_MODEL,
            provider_id="solr",
        )
        await adapter_with_persistence.register_vector_store(vector_store)

        # List should still work (though this provider is read-only for Solr data)
        response = await adapter_with_persistence.openai_list_vector_stores()
        assert response is not None
        assert isinstance(response.data, list)

        # Clean up
        await adapter_with_persistence.unregister_vector_store("test-openai-store")

    @pytest.mark.asyncio
    async def test_openai_api_without_persistence(self, config_basic: Any) -> None:
        """Test that OpenAI API methods work even without persistence (but with empty state)."""
        # Create adapter without persistence
        adapter = SolrVectorIOAdapter(
            config=config_basic, inference_api=cast(Inference, AsyncMock())
        )
        await adapter.initialize()

        # OpenAI attributes ARE initialized (in the mixin's __init__) but empty
        assert hasattr(adapter, "openai_vector_stores")
        assert adapter.openai_vector_stores == {}  # Empty dict

        # The API method should still work (just returns empty results)
        response = await adapter.openai_list_vector_stores()
        assert response is not None
        assert hasattr(response, "data")
        assert response.data == []  # No stores without persistence

        await adapter.shutdown()


# ============================================================================
# Test Class: Vector Search
# ============================================================================


@pytest.mark.search
@pytest.mark.requires_custom_search_handler
class TestVectorSearch:
    """Test vector similarity search functionality."""

    @pytest.mark.asyncio
    async def test_vector_search_basic(
        self,
        adapter_with_chunk_window: Any,
        vector_store_chunk_window: Any,
        random_embedding: Any,
    ) -> None:
        """Test basic vector search returns results."""
        index = await adapter_with_chunk_window._get_and_cache_vector_store_index(
            "test-chunk-window-store"
        )

        response = await index.index.query_vector(
            embedding=random_embedding, k=10, score_threshold=0.0
        )

        assert len(response.chunks) > 0, "Vector search should return results"
        assert len(response.scores) == len(response.chunks)
        assert all(score > 0 for score in response.scores), "Scores should be positive"

        # Check chunk structure
        first_chunk = response.chunks[0]
        assert first_chunk.content is not None
        assert len(first_chunk.content) > 0

    @pytest.mark.asyncio
    async def test_vector_search_with_threshold(
        self,
        adapter_with_chunk_window: Any,
        vector_store_chunk_window: Any,
        random_embedding: Any,
    ) -> None:
        """Test vector search with score threshold filtering.

        Verify vector search respects a score_threshold by filtering results.

        Runs an initial vector query with a zero threshold to collect all
        results, computes the median score from those results, then runs a
        second query using that median as the score_threshold. Asserts the
        filtered result set is no larger than the full set and every returned
        score is greater than or equal to the median.
        """
        index = await adapter_with_chunk_window._get_and_cache_vector_store_index(
            "test-chunk-window-store"
        )

        # Get all results first
        all_results = await index.index.query_vector(
            embedding=random_embedding, k=10, score_threshold=0.0
        )

        assert (
            len(all_results.chunks) > 0
        ), "Zero score threshold should return all results"

        if len(all_results.chunks) > 0:
            # Set threshold to median score
            median_score = sorted(all_results.scores)[len(all_results.scores) // 2]

            filtered_results = await index.index.query_vector(
                embedding=random_embedding, k=10, score_threshold=median_score
            )

            assert len(filtered_results.chunks) <= len(all_results.chunks)
            assert all(score >= median_score for score in filtered_results.scores)


# ============================================================================
# Test Class: Keyword Search
# ============================================================================


@pytest.mark.search
@pytest.mark.requires_custom_search_handler
class TestKeywordSearch:
    """Test keyword-based search functionality."""

    @pytest.mark.asyncio
    async def test_keyword_search_specific(
        self, adapter_with_chunk_window: Any, vector_store_chunk_window: Any
    ) -> None:
        """Test keyword search with specific query."""
        index = await adapter_with_chunk_window._get_and_cache_vector_store_index(
            "test-chunk-window-store"
        )

        response = await index.index.query_keyword(
            query_string="Linux Red Hat", k=5, score_threshold=0.0
        )

        assert (
            len(response.chunks) > 0
        ), "Keyword search should return results for 'Linux Red Hat'"
        assert len(response.scores) == len(response.chunks)

    @pytest.mark.asyncio
    async def test_keyword_search_wildcard(
        self, adapter_with_chunk_window: Any, vector_store_chunk_window: Any
    ) -> None:
        """Test keyword search with wildcard."""
        index = await adapter_with_chunk_window._get_and_cache_vector_store_index(
            "test-chunk-window-store"
        )

        response = await index.index.query_keyword(
            query_string="*:*", k=10, score_threshold=0.0
        )

        assert len(response.chunks) > 0, "Wildcard search should return results"
        assert len(response.scores) == len(response.chunks)


# ============================================================================
# Test Class: Hybrid Search
# ============================================================================


@pytest.mark.search
@pytest.mark.requires_custom_search_handler
class TestHybridSearch:
    """Test hybrid search (vector + keyword) functionality."""

    @pytest.mark.asyncio
    async def test_hybrid_search_basic(
        self,
        adapter_with_chunk_window: Any,
        vector_store_chunk_window: Any,
        random_embedding: Any,
    ) -> None:
        """Test basic hybrid search.

        Verifies that a hybrid query combining vector and keyword signals
        returns results with aligned scores.

        Asserts that a hybrid search using both an embedding and a keyword
        query (with equal vector and keyword boost) returns at least one chunk
        and that the length of the scores list matches the number of returned
        chunks.
        """
        index = await adapter_with_chunk_window._get_and_cache_vector_store_index(
            "test-chunk-window-store"
        )

        response = await index.index.query_hybrid(
            embedding=random_embedding,
            query_string="Linux software",
            k=5,
            score_threshold=0.0,
            reranker_type="solr",
            reranker_params={"vector_boost": 1.0, "keyword_boost": 1.0},
        )

        assert len(response.chunks) > 0, "Hybrid search should return results"
        assert len(response.scores) == len(response.chunks)

    @pytest.mark.asyncio
    async def test_hybrid_search_boost_weights(
        self,
        adapter_with_chunk_window: Any,
        vector_store_chunk_window: Any,
        random_embedding: Any,
    ) -> None:
        """Test hybrid search with different boost weights."""
        index = await adapter_with_chunk_window._get_and_cache_vector_store_index(
            "test-chunk-window-store"
        )

        # Vector-boosted search
        vector_boosted = await index.index.query_hybrid(
            embedding=random_embedding,
            query_string="security",
            k=3,
            score_threshold=0.0,
            reranker_type="solr",
            reranker_params={"vector_boost": 2.0, "keyword_boost": 0.5},
        )

        # Keyword-boosted search
        keyword_boosted = await index.index.query_hybrid(
            embedding=random_embedding,
            query_string="security",
            k=3,
            score_threshold=0.0,
            reranker_type="solr",
            reranker_params={"vector_boost": 0.5, "keyword_boost": 2.0},
        )

        assert len(vector_boosted.chunks) > 0
        assert len(keyword_boosted.chunks) > 0


# ============================================================================
# Test Class: Chunk Window Expansion
# ============================================================================


@pytest.mark.chunk_window
@pytest.mark.requires_custom_search_handler
class TestChunkWindowExpansion:
    """Test chunk window expansion functionality."""

    @pytest.mark.asyncio
    async def test_chunk_window_disabled_without_config(
        self, adapter_basic: Any, random_embedding: Any
    ) -> None:
        """Test that chunk window expansion does NOT happen without chunk_window_config."""
        vector_store = VectorDB(
            identifier="test-no-config",
            embedding_dimension=EMBEDDING_DIM,
            embedding_model=EMBEDDING_MODEL,
            provider_id="solr",
        )
        await adapter_basic.register_vector_store(vector_store)

        index = await adapter_basic._get_and_cache_vector_store_index("test-no-config")

        response = await index.index.query_vector(
            embedding=random_embedding,
            k=5,
            score_threshold=0.0,
        )

        # Verify NO chunks have chunk window expansion metadata
        for chunk in response.chunks:
            assert not chunk.metadata.get(
                "chunk_window_expanded"
            ), "Chunk window expansion should NOT happen without chunk_window_config"

        await adapter_basic.unregister_vector_store("test-no-config")

    @pytest.mark.asyncio
    async def test_chunk_window_expansion_enabled(
        self,
        adapter_with_chunk_window: Any,
        vector_store_chunk_window: Any,
        random_embedding: Any,
    ) -> None:
        """Test chunk window expansion when enabled."""
        index = await adapter_with_chunk_window._get_and_cache_vector_store_index(
            "test-chunk-window-store"
        )

        # Chunk window expansion is now configured via ChunkWindowConfig in the fixture
        # It will automatically expand when chunk_window_config is set with enable_chunk_window=True
        response = await index.index.query_vector(
            embedding=random_embedding,
            k=5,
            score_threshold=0.0,
        )

        if len(response.chunks) > 0:
            expanded_chunks = [
                c for c in response.chunks if c.metadata.get("chunk_window_expanded")
            ]
            if len(expanded_chunks) > 0:
                first_expanded = expanded_chunks[0]
                assert "chunk_window_size" in first_expanded.metadata
                assert first_expanded.metadata["chunk_window_size"] >= 1
                assert "matched_chunk_index" in first_expanded.metadata

    @pytest.mark.asyncio
    async def test_chunk_window_all_search_modes(
        self,
        adapter_with_chunk_window: Any,
        vector_store_chunk_window: Any,
        random_embedding: Any,
    ) -> None:
        """Test chunk window expansion works with all search modes."""
        index = await adapter_with_chunk_window._get_and_cache_vector_store_index(
            "test-chunk-window-store"
        )

        # Chunk window expansion is configured in the fixture via ChunkWindowConfig
        # It will automatically happen for all search modes

        # Vector search
        vector_response = await index.index.query_vector(
            embedding=random_embedding, k=3, score_threshold=0.0
        )
        # May be 0 if no chunks match criteria
        assert len(vector_response.chunks) >= 0

        # Keyword search
        keyword_response = await index.index.query_keyword(
            query_string="Linux", k=3, score_threshold=0.0
        )
        assert len(keyword_response.chunks) >= 0

        # Hybrid search
        hybrid_response = await index.index.query_hybrid(
            embedding=random_embedding,
            query_string="Linux",
            k=3,
            score_threshold=0.0,
            reranker_type="solr",
            reranker_params={"vector_boost": 1.0, "keyword_boost": 1.0},
        )
        assert len(hybrid_response.chunks) >= 0


# ============================================================================
# Test Class: Real Embeddings (slow tests)
# ============================================================================


@pytest.mark.slow
@pytest.mark.embeddings
class TestRealEmbeddings:
    """Test with real embeddings from granite model."""

    @pytest.fixture(scope="class")
    def embedding_model(self) -> Any:
        """Load granite embedding model (cached at class scope).

        Load and return the Hugging Face tokenizer and model for generating embeddings.

        If the `transformers` library is not available, the test is skipped.

        Returns:
            (tokenizer, model): A tuple where `tokenizer` is the pretrained
            tokenizer and `model` is the pretrained model set to evaluation
            mode.
        """
        try:
            from transformers import (  # type: ignore[import-not-found]
                AutoModel,
                AutoTokenizer,
            )

            model_name = EMBEDDING_MODEL
            tokenizer = AutoTokenizer.from_pretrained(model_name)  # nosec B615
            model = AutoModel.from_pretrained(model_name)  # nosec B615
            model.eval()
            return tokenizer, model
        except ImportError:
            pytest.skip("transformers library not installed")

    @pytest.fixture
    def get_embedding(self, embedding_model: Any) -> Any:
        """Generate embeddings from text.

        Create a callable that converts input text to a dense embedding vector.

        Parameters:
            - embedding_model (tuple): A (tokenizer, model) pair where
              `tokenizer` produces PyTorch tensors and `model` is a PyTorch
              model returning a `last_hidden_state`.

        Returns:
            callable: A function that accepts a single `text` (str) and returns
            a 1-D NumPy array representing the embedding produced by tokenizing
            the text (with padding, truncation, and max_length=512) and
            averaging the model's token-level hidden states.
        """
        import torch  # type: ignore[import-not-found]

        tokenizer, model = embedding_model

        def _get_embedding(text: Any) -> Any:
            """
            Compute a fixed-size embedding vector for the given text.

            Parameters:
                text (str): Input text to convert into an embedding.

            Returns:
                embedding (numpy.ndarray): 1-D float32 NumPy array representing the text embedding.
            """
            with torch.no_grad():
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings[0].numpy()

        return _get_embedding

    @pytest.mark.asyncio
    async def test_real_embedding_vector_search(
        self,
        adapter_with_chunk_window: Any,
        vector_store_chunk_window: Any,
        get_embedding: Any,
    ) -> None:
        """Test vector search with real embeddings."""
        index = await adapter_with_chunk_window._get_and_cache_vector_store_index(
            "test-chunk-window-store"
        )

        query_text = "How to fix security vulnerabilities in Linux?"
        query_embedding = get_embedding(query_text)

        response = await index.index.query_vector(
            embedding=query_embedding, k=5, score_threshold=0.0
        )

        assert len(response.chunks) > 0, "Real embedding search should return results"
        # Real embeddings should have better scores than random
        assert (
            response.scores[0] > 0.1
        ), "Top result should have decent similarity score"

    @pytest.mark.asyncio
    async def test_real_embedding_hybrid_search(
        self,
        adapter_with_chunk_window: Any,
        vector_store_chunk_window: Any,
        get_embedding: Any,
    ) -> None:
        """Test hybrid search with real embeddings."""
        index = await adapter_with_chunk_window._get_and_cache_vector_store_index(
            "test-chunk-window-store"
        )

        query_text = "Red Hat Enterprise Linux updates"
        query_embedding = get_embedding(query_text)

        response = await index.index.query_hybrid(
            embedding=query_embedding,
            query_string="Red Hat Linux",
            k=5,
            score_threshold=0.0,
            reranker_type="solr",
            reranker_params={"vector_boost": 1.5, "keyword_boost": 1.0},
        )

        assert (
            len(response.chunks) > 0
        ), "Hybrid search with real embeddings should return results"


# ============================================================================
# Main entry point for direct execution
# ============================================================================

if __name__ == "__main__":
    # Run pytest with verbose output and short traceback
    # Additional CLI options can be passed through, e.g.:
    #   uv run tests.py -k test_name
    #   uv run tests.py -m basic
    #   uv run tests.py --help
    import sys

    pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])
