"""Solr VectorIO adapter implementation."""

from typing import Any, Optional

import httpx
from ogx.core.storage.kvstore import kvstore_impl
from ogx.log import get_logger
from ogx.providers.utils.memory.openai_vector_store_mixin import (
    OpenAIVectorStoreMixin,
)
from ogx.providers.utils.memory.vector_store import VectorStoreWithIndex
from ogx_api.common.errors import VectorStoreNotFoundError
from ogx_api.datatypes import VectorStoresProtocolPrivate
from ogx_api.files import Files
from ogx_api.inference import Inference
from ogx_api.vector_io import (
    DeleteChunksRequest,
    InsertChunksRequest,
    QueryChunksRequest,
    QueryChunksResponse,
    VectorIO,
)
from ogx_api.vector_stores import VectorStore

from .config import SolrVectorIOConfig
from .constants import VECTOR_DBS_PREFIX
from .solr_index import SolrIndex

log = get_logger(name=__name__, category="vector_io::solr")


class SolrVectorIOAdapter(
    OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate
):
    """
    Read-only Solr VectorIO adapter.

    This adapter provides read-only access to Solr collections for vector search.
    Write operations (insert_chunks, delete_chunks, etc.) are not supported.
    """

    def __init__(
        self,
        config: SolrVectorIOConfig,
        inference_api: Inference,
        files_api: Optional[Files] = None,
    ) -> None:
        """
        Initialize the Solr-backed read-only VectorIO adapter and prepare internal cache/state.

        Parameters:
            - config (SolrVectorIOConfig): Configuration for Solr connections,
              collection, schema fields, and chunk-window expansion.
            - inference_api (Inference): Inference service used for
              embedding/reranking operations.
            - files_api (Optional[Files]): Optional file management API used by
              higher-level utilities; may be None.

        Notes:
            - This adapter is read-only: KV persistence (if used) is managed
              separately and `kvstore` is not provided here.
            - Creates an empty in-memory cache for vector store indexes and
              leaves `vector_store_table` uninitialized.
        """
        super().__init__(inference_api=inference_api, files_api=files_api, kvstore=None)
        self.config = config
        self.inference_api = inference_api
        self.cache: dict[Any, Any] = {}
        self.vector_store_table = None
        log.info("SolrVectorIOAdapter instance created")

    async def initialize(self) -> None:
        """
        Initialize the Solr-backed VectorIO adapter and load any persisted vector store def.

        If persistence is configured, initializes the KV store and the
        read-only OpenAI vector store support. Then loads all stored vector
        store metadata from the KV range prefixed by VECTOR_DBS_PREFIX,
        constructs a SolrIndex for each entry, calls its initialize method, and
        caches the resulting VectorStoreWithIndex instances for runtime use.
        Logs progress and skips KV/openai initialization when persistence is
        not configured.
        """
        log.info("Initializing Solr vector_io adapter")
        log.info(
            f"Configuration: solr_url={self.config.solr_url}, "
            f"collection={self.config.collection_name}, "
            f"vector_field={self.config.vector_field}, "
            f"dimension={self.config.embedding_dimension}"
        )

        if self.config.persistence is not None:
            self.kvstore = await kvstore_impl(self.config.persistence)
            log.info("KV store initialized")

            # Initialize OpenAI vector stores support (read-only) - requires kvstore
            await self.initialize_openai_vector_stores()
            log.info("OpenAI vector stores initialized")
        else:
            log.info(
                "No persistence configured, skipping KV store and OpenAI vector "
                "store initialization"
            )

        # Load any persisted vector stores
        if self.kvstore is not None:
            start_key = VECTOR_DBS_PREFIX
            end_key = f"{VECTOR_DBS_PREFIX}\xff"
            stored_vector_stores = await self.kvstore.values_in_range(
                start_key, end_key
            )

            log.info(f"Loading {
                    len(stored_vector_stores)
                } persisted vector stores from KV store")
            for vector_store_data in stored_vector_stores:
                vector_store = VectorStore.model_validate_json(vector_store_data)
                log.info(f"Loading vector store: {vector_store.identifier}")

                index = SolrIndex(vector_store=vector_store, config=self.config)
                await index.initialize()
                self.cache[vector_store.identifier] = VectorStoreWithIndex(
                    vector_store, index, self.inference_api
                )

        log.info("Solr vector_io adapter initialization complete")

    async def shutdown(self) -> None:
        """
        Shuts down the SolrVectorIOAdapter and releases mixin-managed resources.

        Performs cleanup of resources managed by the adapter's mixins (for
        example, file batch tasks) and completes adapter shutdown.
        """
        log.info("Shutting down Solr vector_io adapter")
        # Clean up mixin resources (file batch tasks)
        await super().shutdown()
        log.info("Shutdown complete")

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        """Register a vector store (read-only, just caches the metadata).

        Parameters:
            - vector_store (VectorStore): Vector store metadata to register and
              cache. The function will persist this metadata to the configured
              KV store (keyed under VECTOR_DBS_PREFIX + identifier) if a KV
              store is available, create and initialize a SolrIndex for the
              store, and store a VectorStoreWithIndex in the adapter's cache.
        """
        log.info(f"Registering vector store: {vector_store.identifier}")
        if self.kvstore is not None:
            key = f"{VECTOR_DBS_PREFIX}{vector_store.identifier}"
            await self.kvstore.set(key=key, value=vector_store.model_dump_json())
            log.info(f"Persisted vector store metadata to KV store: {key}")
        else:
            log.info("No KV store configured, skipping persistence")

        index = SolrIndex(vector_store=vector_store, config=self.config)
        await index.initialize()
        self.cache[vector_store.identifier] = VectorStoreWithIndex(
            vector_store, index, self.inference_api
        )
        log.info(f"Successfully registered vector store: {vector_store.identifier}")

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        """Unregister a vector store (removes from cache and KV store).

        Parameters:
            - vector_store_id (str): Identifier of the vector store; used as
              the suffix for the KV key when deleting persisted metadata.
        """
        log.info(f"Unregistering vector store: {vector_store_id}")

        if vector_store_id in self.cache:
            del self.cache[vector_store_id]
            log.info(f"Removed vector store from cache: {vector_store_id}")

        if self.kvstore is not None:
            await self.kvstore.delete(key=f"{VECTOR_DBS_PREFIX}{vector_store_id}")
            log.info("Removed from KV store")

        log.info(f"Successfully unregistered vector store: {vector_store_id}")

    async def insert_chunks(self, request: InsertChunksRequest) -> None:
        """Reject chunk insertion because this provider is read-only.

        Parameters:
            request: Insertion request containing the vector store identifier,
                chunks, and optional time-to-live.

        Raises:
            NotImplementedError: Always raised to indicate that write
            operations are not supported.
        """
        log.warning(
            "Attempted to insert %d chunks into read-only provider "
            "(vector_store_id=%s)",
            len(request.chunks),
            request.vector_store_id,
        )
        raise NotImplementedError("SolrVectorIO is read-only.")

    async def query_chunks(
        self,
        request: QueryChunksRequest,
    ) -> QueryChunksResponse:
        """Query chunks from the Solr collection.

        Retrieve matching chunks from the Solr-backed vector store identified
        by `request.vector_store_id`.

        The `request.query` may be a search string, a single content item, or a list of
        content items; the adapter delegates retrieval to the underlying Solr
        index which performs vector, keyword, or hybrid search as appropriate.
        The optional `request.params` dictionary supplies provider-specific query
        options (for example: `k`, `score_threshold`, `reranker_type`,
        `reranker_params`) that control result count, filtering, and reranking.

        Returns:
            QueryChunksResponse: Search results containing a list of chunks and
            their corresponding similarity scores. Results may include
            chunk-window expansions when the vector store is configured to
            expand matches into larger contextual windows.
        """
        log.info(f"Query chunks request for vector_store_id={request.vector_store_id}")
        try:
            index = await self._get_and_cache_vector_store_index(
                request.vector_store_id
            )
            result = await index.query_chunks(request)
            log.info(f"Query returned {len(result.chunks)} chunks")
            return result
        except httpx.TransportError as e:
            log.warning(
                "OKP/Solr unreachable for %s, returning empty results: %s",
                request.vector_store_id,
                e,
            )
            return QueryChunksResponse(chunks=[], scores=[])

    async def delete_chunks(self, request: DeleteChunksRequest) -> None:
        """Reject chunk deletion because this provider is read-only.

        Parameters:
            request: Deletion request containing the vector store identifier and
                chunks to remove.

        Raises:
            NotImplementedError: Always, because SolrVectorIO is read-only.
        """
        log.warning(
            "Attempted to delete %d chunks from read-only provider (store_id=%s)",
            len(request.chunks),
            request.vector_store_id,
        )
        raise NotImplementedError("SolrVectorIO is read-only.")

    async def _get_and_cache_vector_store_index(
        self, vector_store_id: str
    ) -> VectorStoreWithIndex:
        """
        Retrieve the cached VectorStoreWithIndex.

        Retrieve the cached VectorStoreWithIndex for a given vector store ID,
        loading it from the vector store table and caching it if not already
        present.

        Parameters:
            vector_store_id (str): Identifier of the vector store to retrieve.

        Returns:
            VectorStoreWithIndex: The cached or newly loaded vector store
            paired with its Solr index and inference API.

        Raises:
            VectorStoreNotFoundError: If the vector store table is not
            configured or the requested vector store does not exist.
        """
        if vector_store_id in self.cache:
            log.debug(f"Retrieved vector store from cache: {vector_store_id}")
            return self.cache[vector_store_id]

        log.info(f"Vector store not in cache, loading from table: {vector_store_id}")

        if self.vector_store_table is None:
            log.error(f"Vector store table not set, cannot find: {vector_store_id}")
            raise VectorStoreNotFoundError(vector_store_id)

        vector_store = self.vector_store_table.get_vector_store(vector_store_id)
        if not vector_store:
            log.error(f"Vector store not found: {vector_store_id}")
            raise VectorStoreNotFoundError(vector_store_id)

        log.info(f"Loaded vector store from table: {vector_store_id}")
        index = SolrIndex(vector_store=vector_store, config=self.config)
        await index.initialize()
        self.cache[vector_store_id] = VectorStoreWithIndex(
            vector_store, index, self.inference_api
        )
        return self.cache[vector_store_id]
