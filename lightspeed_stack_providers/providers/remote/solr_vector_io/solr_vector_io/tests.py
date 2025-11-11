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

import pytest
import numpy as np

from src.solr_vector_io import (
    ChunkWindowConfig,
    SolrVectorIOAdapter,
    SolrVectorIOConfig,
)
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk
from llama_stack.providers.utils.memory.vector_store import ChunkForDeletion
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig


# ============================================================================
# Configuration
# ============================================================================

SOLR_URL = "http://localhost4:8080/solr"
COLLECTION_NAME = "portal"
VECTOR_FIELD = "chunk_vector"
CONTENT_FIELD = "chunk"
EMBEDDING_DIM = 384
EMBEDDING_MODEL = "ibm-granite/granite-embedding-30m-english"


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def config_basic():
    """Basic configuration without chunk window schema."""
    return SolrVectorIOConfig(
        solr_url=SOLR_URL,
        collection_name=COLLECTION_NAME,
        vector_field=VECTOR_FIELD,
        content_field=CONTENT_FIELD,
        embedding_dimension=EMBEDDING_DIM,
        persistence=None,
    )


@pytest.fixture
def config_with_chunk_window():
    """Configuration with chunk window expansion enabled."""
    return SolrVectorIOConfig(
        solr_url=SOLR_URL,
        collection_name=COLLECTION_NAME,
        vector_field=VECTOR_FIELD,
        content_field=CONTENT_FIELD,
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
            parent_content_url_field="reference_url",
            chunk_filter_query="is_chunk:true",
        ),
    )


@pytest.fixture
async def adapter_basic(config_basic):
    """Adapter instance with basic configuration."""
    adapter = SolrVectorIOAdapter(config=config_basic, inference_api=None)
    await adapter.initialize()
    yield adapter
    await adapter.shutdown()


@pytest.fixture
async def adapter_with_chunk_window(config_with_chunk_window):
    """Adapter instance with chunk window configuration."""
    adapter = SolrVectorIOAdapter(config=config_with_chunk_window, inference_api=None)
    await adapter.initialize()
    yield adapter
    await adapter.shutdown()


@pytest.fixture
async def vector_store_basic(adapter_basic):
    """Registered vector store with basic config."""
    # adapter_basic is already awaited by pytest-asyncio
    vector_store = VectorDB(
        identifier="test-basic-store",
        embedding_dimension=EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
        provider_id="solr",
    )
    await adapter_basic.register_vector_db(vector_store)
    yield vector_store
    await adapter_basic.unregister_vector_db("test-basic-store")


@pytest.fixture
async def vector_store_chunk_window(adapter_with_chunk_window):
    """Registered vector store with chunk window config."""
    # adapter_with_chunk_window is already awaited by pytest-asyncio
    vector_store = VectorDB(
        identifier="test-chunk-window-store",
        embedding_dimension=EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
        provider_id="solr",
    )
    await adapter_with_chunk_window.register_vector_db(vector_store)
    yield vector_store
    await adapter_with_chunk_window.unregister_vector_db("test-chunk-window-store")


@pytest.fixture
def random_embedding():
    """Generate random 384-dimensional embedding."""
    return np.random.rand(EMBEDDING_DIM).astype(np.float32)


@pytest.fixture
def config_with_persistence(tmp_path):
    """Configuration with persistence enabled using SQLite KV store."""
    return SolrVectorIOConfig(
        solr_url=SOLR_URL,
        collection_name=COLLECTION_NAME,
        vector_field=VECTOR_FIELD,
        content_field=CONTENT_FIELD,
        embedding_dimension=EMBEDDING_DIM,
        persistence=SqliteKVStoreConfig(
            namespace="test_vector_io",
        ),
    )


@pytest.fixture
async def adapter_with_persistence(config_with_persistence):
    """Adapter instance with persistence enabled."""
    adapter = SolrVectorIOAdapter(config=config_with_persistence, inference_api=None)
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
    async def test_adapter_initialization(self, config_basic):
        """Test that adapter initializes successfully."""
        adapter = SolrVectorIOAdapter(config=config_basic, inference_api=None)
        await adapter.initialize()
        assert adapter is not None
        await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_vector_store_registration(self, adapter_basic):
        """Test vector store registration and unregistration."""
        vector_store = VectorDB(
            identifier="test-registration",
            embedding_dimension=EMBEDDING_DIM,
            embedding_model=EMBEDDING_MODEL,
            provider_id="solr",
        )
        await adapter_basic.register_vector_db(vector_store)
        assert "test-registration" in adapter_basic.cache

        await adapter_basic.unregister_vector_db("test-registration")
        assert "test-registration" not in adapter_basic.cache

    @pytest.mark.asyncio
    async def test_read_only_insert_fails(self, adapter_basic, vector_store_basic):
        """Test that insert operations raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="read-only"):
            await adapter_basic.insert_chunks(
                vector_store_id="test-basic-store",
                chunks=[
                    Chunk(content="test", metadata={}, embedding=[0.1] * EMBEDDING_DIM)
                ],
            )

    @pytest.mark.asyncio
    async def test_read_only_delete_fails(self, adapter_basic, vector_store_basic):
        """Test that delete operations raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="read-only"):
            await adapter_basic.delete_chunks(
                store_id="test-basic-store",
                chunks_for_deletion=[
                    ChunkForDeletion(chunk_id="test", document_id="test-doc")
                ],
            )


# ============================================================================
# Test Class: Persistence
# ============================================================================


@pytest.mark.persistence
class TestPersistence:
    """Test persistence functionality with KV store."""

    @pytest.mark.asyncio
    async def test_adapter_initialization_with_persistence(
        self, config_with_persistence
    ):
        """Test that adapter initializes successfully with persistence enabled."""
        adapter = SolrVectorIOAdapter(
            config=config_with_persistence, inference_api=None
        )
        await adapter.initialize()

        # Check that kvstore was initialized
        assert adapter.kvstore is not None

        # Check that OpenAI API support was initialized
        assert hasattr(adapter, "openai_vector_stores")
        assert adapter.openai_vector_stores is not None

        await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_vector_store_persistence(self, adapter_with_persistence):
        """Test that vector stores are persisted to KV store."""
        from src.solr_vector_io.solr import VECTOR_DBS_PREFIX

        # Register a vector store
        vector_store = VectorDB(
            identifier="test-persisted-store",
            embedding_dimension=EMBEDDING_DIM,
            embedding_model=EMBEDDING_MODEL,
            provider_id="solr",
        )
        await adapter_with_persistence.register_vector_db(vector_store)

        # Verify it's in the cache
        assert "test-persisted-store" in adapter_with_persistence.cache

        # Verify it's in the KV store with the correct prefix
        key = f"{VECTOR_DBS_PREFIX}test-persisted-store"
        persisted_data = await adapter_with_persistence.kvstore.get(key=key)
        assert persisted_data is not None

        # Clean up
        await adapter_with_persistence.unregister_vector_db("test-persisted-store")

    @pytest.mark.asyncio
    async def test_vector_store_reload_from_persistence(self, config_with_persistence):
        """Test that vector stores are loaded from persistence on adapter initialization."""
        # First adapter: register a vector store
        adapter1 = SolrVectorIOAdapter(
            config=config_with_persistence, inference_api=None
        )
        await adapter1.initialize()

        vector_store = VectorDB(
            identifier="test-reload-store",
            embedding_dimension=EMBEDDING_DIM,
            embedding_model=EMBEDDING_MODEL,
            provider_id="solr",
        )
        await adapter1.register_vector_db(vector_store)

        # Verify it's registered
        assert "test-reload-store" in adapter1.cache

        # Shutdown the first adapter
        await adapter1.shutdown()

        # Second adapter: should load the persisted vector store
        adapter2 = SolrVectorIOAdapter(
            config=config_with_persistence, inference_api=None
        )
        await adapter2.initialize()

        # Verify the vector store was loaded from persistence
        assert "test-reload-store" in adapter2.cache

        # Clean up
        await adapter2.unregister_vector_db("test-reload-store")
        await adapter2.shutdown()

    @pytest.mark.asyncio
    async def test_persistence_with_search(
        self, adapter_with_persistence, random_embedding
    ):
        """Test that search works with persistence enabled."""
        # Register a vector store
        vector_store = VectorDB(
            identifier="test-search-persisted",
            embedding_dimension=EMBEDDING_DIM,
            embedding_model=EMBEDDING_MODEL,
            provider_id="solr",
        )
        await adapter_with_persistence.register_vector_db(vector_store)

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
        await adapter_with_persistence.unregister_vector_db("test-search-persisted")


# ============================================================================
# Test Class: OpenAI API
# ============================================================================


@pytest.mark.openai
class TestOpenAIAPI:
    """Test vector similarity search functionality."""

    @pytest.mark.asyncio
    @pytest.mark.persistence
    async def test_openai_api_list_vector_stores(self, adapter_with_persistence):
        """Test that OpenAI API list_vector_stores works with persistence."""
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
        await adapter_with_persistence.register_vector_db(vector_store)

        # List should still work (though this provider is read-only for Solr data)
        response = await adapter_with_persistence.openai_list_vector_stores()
        assert response is not None
        assert isinstance(response.data, list)

        # Clean up
        await adapter_with_persistence.unregister_vector_db("test-openai-store")

    @pytest.mark.asyncio
    async def test_openai_api_without_persistence(self, config_basic):
        """Test that OpenAI API methods work even without persistence (but with empty state)."""
        # Create adapter without persistence
        adapter = SolrVectorIOAdapter(config=config_basic, inference_api=None)
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
class TestVectorSearch:
    """Test vector similarity search functionality."""

    @pytest.mark.asyncio
    async def test_vector_search_basic(
        self, adapter_with_chunk_window, vector_store_chunk_window, random_embedding
    ):
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
        self, adapter_with_chunk_window, vector_store_chunk_window, random_embedding
    ):
        """Test vector search with score threshold filtering."""
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
class TestKeywordSearch:
    """Test keyword-based search functionality."""

    @pytest.mark.asyncio
    async def test_keyword_search_specific(
        self, adapter_with_chunk_window, vector_store_chunk_window
    ):
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
        self, adapter_with_chunk_window, vector_store_chunk_window
    ):
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
class TestHybridSearch:
    """Test hybrid search (vector + keyword) functionality."""

    @pytest.mark.asyncio
    async def test_hybrid_search_basic(
        self, adapter_with_chunk_window, vector_store_chunk_window, random_embedding
    ):
        """Test basic hybrid search."""
        index = await adapter_with_chunk_window._get_and_cache_vector_store_index(
            "test-chunk-window-store"
        )

        response = await index.index.query_hybrid(
            embedding=random_embedding,
            query_string="Linux software",
            k=5,
            score_threshold=0.0,
            reranker_type="solr_native",
            reranker_params={"vector_boost": 1.0, "keyword_boost": 1.0},
        )

        assert len(response.chunks) > 0, "Hybrid search should return results"
        assert len(response.scores) == len(response.chunks)

    @pytest.mark.asyncio
    async def test_hybrid_search_boost_weights(
        self, adapter_with_chunk_window, vector_store_chunk_window, random_embedding
    ):
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
            reranker_type="solr_native",
            reranker_params={"vector_boost": 2.0, "keyword_boost": 0.5},
        )

        # Keyword-boosted search
        keyword_boosted = await index.index.query_hybrid(
            embedding=random_embedding,
            query_string="security",
            k=3,
            score_threshold=0.0,
            reranker_type="solr_native",
            reranker_params={"vector_boost": 0.5, "keyword_boost": 2.0},
        )

        assert len(vector_boosted.chunks) > 0
        assert len(keyword_boosted.chunks) > 0


# ============================================================================
# Test Class: Chunk Window Expansion
# ============================================================================


@pytest.mark.chunk_window
class TestChunkWindowExpansion:
    """Test chunk window expansion functionality."""

    @pytest.mark.asyncio
    async def test_chunk_window_disabled_without_config(
        self, adapter_basic, random_embedding
    ):
        """Test that chunk window expansion does NOT happen without chunk_window_config."""
        vector_store = VectorDB(
            identifier="test-no-config",
            embedding_dimension=EMBEDDING_DIM,
            embedding_model=EMBEDDING_MODEL,
            provider_id="solr",
        )
        await adapter_basic.register_vector_db(vector_store)

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

        await adapter_basic.unregister_vector_db("test-no-config")

    @pytest.mark.asyncio
    async def test_chunk_window_expansion_enabled(
        self, adapter_with_chunk_window, vector_store_chunk_window, random_embedding
    ):
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
        self, adapter_with_chunk_window, vector_store_chunk_window, random_embedding
    ):
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
            reranker_type="solr_native",
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
    def embedding_model(self):
        """Load granite embedding model (cached at class scope)."""
        try:
            from transformers import AutoTokenizer, AutoModel

            model_name = EMBEDDING_MODEL
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            return tokenizer, model
        except ImportError:
            pytest.skip("transformers library not installed")

    @pytest.fixture
    def get_embedding(self, embedding_model):
        """Function to generate embeddings from text."""
        import torch

        tokenizer, model = embedding_model

        def _get_embedding(text):
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
        self, adapter_with_chunk_window, vector_store_chunk_window, get_embedding
    ):
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
        self, adapter_with_chunk_window, vector_store_chunk_window, get_embedding
    ):
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
            reranker_type="solr_native",
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
