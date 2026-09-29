"""Solr embedding-index implementation."""

from typing import Any, Optional

import httpx
from numpy.typing import NDArray
from ogx.log import get_logger
from ogx.providers.utils.memory.vector_store import ChunkForDeletion, EmbeddingIndex
from ogx.providers.utils.vector_io.filters import ComparisonFilter, Filter
from ogx_api.vector_io import (
    Chunk,
    ChunkMetadata,
    EmbeddedChunk,
    QueryChunksResponse,
)
from ogx_api.vector_stores import VectorStore

from .chunk_window import ChunkWindowMixin
from .config import SolrVectorIOConfig
from .constants import OKP_SOURCE
from .filter_helpers import build_solr_filter_query

log = get_logger(name=__name__, category="vector_io::solr")

# Maintain backward compatibility with private function names
_build_solr_filter_query = build_solr_filter_query


class SolrIndex(  # pylint: disable=too-many-instance-attributes
    ChunkWindowMixin, EmbeddingIndex
):
    """
    Read-only Solr vector index implementation using DenseVectorField and KNN search.

    Supports hybrid search using Solr's native query reranking capabilities.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        config: SolrVectorIOConfig,
    ):
        """
        Initialize a SolrIndex with configuration for read-only searches.

        Parameters:
            - vector_store (VectorStore): Metadata describing the vector store this index serves.
            - config (SolrVectorIOConfig): Solr connection, schema, embedding,
              timeout, and optional chunk-window settings.
        """
        self.vector_store = vector_store
        self.solr_url = config.solr_url.rstrip("/")
        self.collection_name = config.collection_name
        self.vector_field = config.vector_field
        self.content_field = config.content_field
        self.id_field = config.id_field
        self.dimension = config.embedding_dimension
        self.embedding_model = config.embedding_model
        self.request_timeout = config.request_timeout
        self.chunk_window_config = config.chunk_window_config
        self.base_url = f"{self.solr_url}/{self.collection_name}"
        log.info(
            f"Initialized SolrIndex for collection '{self.collection_name}' at {
                self.base_url
            }, "
            f"vector_field='{self.vector_field}', content_field='{
                self.content_field
            }', dimension={self.dimension}, "
            f"chunk_window_enabled={self.chunk_window_config is not None}"
        )

    def _create_http_client(self) -> httpx.AsyncClient:
        """Create an HTTP client configured for Solr connections.

        Uses IPv4 by binding to 0.0.0.0. When Solr runs in a podman container,
        IPv4 is required unless podman has been explicitly configured to support IPv6.

        Returns:
            httpx.AsyncClient: An async HTTP client with the instance's request
            timeout and an IPv4-bound transport.
        """
        return httpx.AsyncClient(
            timeout=self.request_timeout,
            transport=httpx.AsyncHTTPTransport(local_address="0.0.0.0"),  # nosec B104
        )

    async def initialize(self) -> None:
        """Verify connection to Solr and collection exists.

        Check that the configured Solr collection is reachable.

        Raises:
            RuntimeError: If the Solr collection is unavailable or an HTTP
            error occurs while verifying the connection.
        """
        log.info(f"Initializing connection to Solr collection: {self.collection_name}")
        async with self._create_http_client() as client:
            try:
                # Check if collection exists
                response = await client.get(f"{self.base_url}/select?q=*:*&rows=0")
                response.raise_for_status()
                log.info(
                    f"Successfully connected to Solr collection: {self.collection_name}"
                )
            except httpx.HTTPStatusError as e:
                log.error(
                    f"HTTP error connecting to Solr collection {self.collection_name}: "
                    f"status={e.response.status_code}"
                )
                raise RuntimeError(f"Failed to connect to Solr collection {
                        self.collection_name
                    }: HTTP {e.response.status_code}") from e
            except Exception as e:
                log.exception(
                    f"Error connecting to Solr collection {self.collection_name}"
                )
                raise RuntimeError(
                    f"Error connecting to Solr collection {self.collection_name}: {e}"
                ) from e

    async def add_chunks(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Not implemented - this is a read-only provider.

        Attempting to add chunks to this read-only SolrIndex is not supported.

        Parameters:
            embedded_chunks: Chunks and their embeddings provided for insertion
                (ignored).

        Raises:
            NotImplementedError: Always raised because SolrIndex is read-only.
        """
        log.warning(
            "Attempted to add %d chunks to read-only SolrIndex",
            len(embedded_chunks),
        )
        raise NotImplementedError("SolrVectorIO is read-only.")

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Not implemented - this is a read-only provider.

        Rejects attempts to delete chunks from the Solr-backed index because
        the store is read-only.

        Raises:
            NotImplementedError: always raised with message "SolrVectorIO is read-only."
        """
        log.warning(f"Attempted to delete {
                len(chunks_for_deletion)
            } chunks from read-only SolrIndex")
        raise NotImplementedError("SolrVectorIO is read-only.")

    async def query_vector(
        self,
        embedding: NDArray[Any],
        k: int,
        score_threshold: float,
        filters: Optional[Filter] = None,
    ) -> QueryChunksResponse:
        """
        Perform vector similarity search using Solr's KNN query.

        Parameters:
            embedding: The query embedding vector
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Optional filters to apply to the search results

        Returns:
            QueryChunksResponse with matching chunks and scores

        """
        log.info(
            f"Performing vector search: k={k}, score_threshold={score_threshold}, "
            f"embedding_dim={len(embedding)}, filters_present={bool(filters)}"
        )
        log.debug(f"Vector search filters: {filters}")

        async with self._create_http_client() as client:
            # The SearchHandler executes the KNN expression supplied in q.
            # Solr expects the vector literal in [f1,f2,f3] format.
            vector_str = "[" + ",".join(str(v) for v in embedding.tolist()) + "]"

            params = {
                "q": f"{{!knn f={self.vector_field} topK={k}}}{vector_str}",
                "rows": k,
                "fl": "*,score",
                "wt": "json",
            }

            # Build combined filter query from static config and dynamic filters
            chunk_filter = (
                self.chunk_window_config.chunk_filter_query
                if self.chunk_window_config
                else None
            )
            combined_filter = _build_solr_filter_query(chunk_filter, filters)
            if combined_filter:
                params["fq"] = combined_filter
                if isinstance(combined_filter, list):
                    log.info(
                        f"Applying {len(combined_filter)} filter queries (multiple fq params)"
                    )
                    log.debug(f"Filter queries: {combined_filter}")
                else:
                    log.info("Applying 1 filter query (single fq param)")
                    log.debug(f"Filter query: {combined_filter}")
            else:
                log.info("No filter query applied (fq parameter not set)")

            try:
                response = await client.post(
                    f"{self.base_url}/semantic-search",
                    data=params,
                )
                response.raise_for_status()
                return self._build_vector_query_response(
                    response.json(), score_threshold
                )

            except httpx.HTTPStatusError as e:
                log.error(
                    f"semantic-search failed: status={e.response.status_code}, "
                    f"body={e.response.text[:500]}"
                )
                raise

    def _build_vector_query_response(
        self, data: dict[str, Any], score_threshold: float
    ) -> QueryChunksResponse:
        """Convert Solr vector-search documents into a query response.

        Parameters:
            data: Solr JSON response body.
            score_threshold: Minimum score for returned chunks.

        Returns:
            Query response containing chunks that meet the score threshold.
        """
        chunks = []
        scores = []
        documents = data.get("response", {}).get("docs", [])

        for document in documents:
            score = float(document.get("score", 0.0))
            if score < score_threshold:
                continue

            chunk = self._doc_to_chunk(document)
            if chunk is None:
                continue

            chunks.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    chunk_metadata=chunk.chunk_metadata,
                    metadata=chunk.metadata or {},
                    embedding=[],
                    embedding_model=self.embedding_model,
                    embedding_dimension=self.dimension,
                )
            )
            scores.append(score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_keyword(
        self,
        query_string: str,
        k: int,
        score_threshold: float,
        filters: Optional[Filter] = None,
    ) -> QueryChunksResponse:
        """
        Perform keyword-based search using Solr's text search.

        Parameters:
            query_string: The text query for keyword search
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Optional filters to apply to the search results

        Returns:
            QueryChunksResponse with matching chunks and scores

        """
        log.info(
            f"Performing keyword search: query='{query_string}', k={k}, "
            f"score_threshold={score_threshold}, filters_present={bool(filters)}"
        )
        log.debug(f"Keyword search filters: {filters}")

        # Replace ? and * because when the edismax text parser is enabled, they
        # are evaluated as lucene wildcards
        if query_string != "*:*":
            query_string = query_string.replace("?", "").replace("*", "")

        async with self._create_http_client() as client:
            solr_params = {
                "q": query_string,
                "rows": k,
                "fl": "*,score",
                "wt": "json",
            }

            # Build combined filter query from static config and dynamic filters
            chunk_filter = (
                self.chunk_window_config.chunk_filter_query
                if self.chunk_window_config
                else None
            )
            combined_filter = _build_solr_filter_query(chunk_filter, filters)
            if combined_filter:
                solr_params["fq"] = combined_filter
                if isinstance(combined_filter, list):
                    log.info(
                        f"Applying {len(combined_filter)} filter queries (multiple fq params)"
                    )
                    log.debug(f"Filter queries: {combined_filter}")
                else:
                    log.info("Applying 1 filter query (single fq param)")
                    log.debug(f"Filter query: {combined_filter}")
            else:
                log.info("No filter query applied (fq parameter not set)")

            log.debug(f"Final fq param in request: {solr_params.get('fq')}")

            try:
                log.info("Sending keyword query to Solr")
                response = await client.get(
                    f"{self.base_url}/hybrid-search",
                    params=solr_params,  # type: ignore
                )
                response.raise_for_status()
                query_response = self._build_keyword_query_response(
                    response.json(), score_threshold
                )

                # Apply chunk window expansion if configured
                if self.chunk_window_config is not None:
                    return await self._apply_chunk_window_expansion(
                        initial_response=query_response,
                        min_chunk_gap=self.chunk_window_config.min_chunk_gap,
                        min_chunk_window=self.chunk_window_config.min_chunk_window,
                    )

                return query_response

            except httpx.HTTPStatusError as e:
                log.error(
                    f"HTTP error during keyword search: status={e.response.status_code}"
                )
                raise
            except Exception as e:
                log.exception(f"Error querying Solr with keyword search: {e}")
                raise

    def _build_keyword_query_response(
        self, data: dict[str, Any], score_threshold: float
    ) -> QueryChunksResponse:
        """Convert Solr keyword-search documents into a query response.

        Parameters:
            data: Solr JSON response body.
            score_threshold: Minimum score for returned chunks.

        Returns:
            Query response containing chunks that meet the score threshold.
        """
        documents = data.get("response", {}).get("docs", [])
        num_documents = data.get("response", {}).get("numFound", 0)
        chunks = []
        scores = []
        log.info("Solr returned %d documents for keyword search", num_documents)

        for document in documents:
            score = float(document.get("score", 0))
            if score < score_threshold:
                log.debug(
                    "Filtering out document with score %s < threshold %s",
                    score,
                    score_threshold,
                )
                continue

            chunk = self._doc_to_chunk(document)
            if chunk is not None:
                chunks.append(chunk)
                scores.append(score)

        log.debug(
            "Keyword search returned %d chunks (filtered from %d by score threshold)",
            len(chunks),
            num_documents,
        )
        return QueryChunksResponse(chunks=chunks, scores=scores)

    # The EmbeddingIndex interface requires these independent query controls.
    # The Solr request and result diagnostics must remain together to correlate
    # a reranked query with the returned documents.
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements
    async def query_hybrid(
        self,
        embedding: NDArray[Any],
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: Optional[dict[str, Any]] = None,
        filters: Optional[Filter] = None,
    ) -> QueryChunksResponse:
        """
        Hybrid search combining vector similarity and keyword search using Solr's native reranking.

        Parameters:
            - embedding: The query embedding vector
            - query_string: The text query for keyword search
            - k: Number of results to return
            - score_threshold: Minimum similarity score threshold
            - reranker_type: Type of reranker (ignored, uses Solr's native capabilities)
            - reranker_params: Parameters for reranking (e.g., boost values)
            - filters: Optional filters to apply to the search results

        Returns:
            QueryChunksResponse with combined results

        """
        if reranker_params is None:
            reranker_params = {}

        # Get boost parameters, defaulting to equal weighting
        vector_boost = reranker_params.get("vector_boost", 8.0)

        # Replace ? and * because when the edismax text parser is enabled, they
        # are evaluated as lucene wildcards
        query_string = query_string.replace("?", "").replace("*", "")

        log.info(
            f"Performing hybrid search: query='{query_string}', k={k}, "
            f"score_threshold={score_threshold}, vector_boost={vector_boost}, "
            f"filters_present={bool(filters)}"
        )
        log.debug(f"Hybrid search filters: {filters}")

        async with self._create_http_client() as client:
            # Use POST to avoid URI length limits with large embeddings
            # Solr expects format: [f1,f2,f3]
            vector_str = "[" + ",".join(str(v) for v in embedding.tolist()) + "]"

            # Construct hybrid query using Solr's query boosting
            # This uses both KNN and text search with configurable boosts
            # The keyword_boost is applied via the reRankWeight for the text query
            # and vector_boost is applied via reRankWeight for the KNN reranking
            data_params = {
                "q": query_string,
                "rq": f"{{!rerank reRankQuery=$rqq reRankDocs=100 reRankWeight={vector_boost}}}",
                "rqq": f"{{!knn f={self.vector_field} topK=100}}{vector_str}",
                "rows": k,
                "fl": "*,score,originalScore()",
                "wt": "json",
            }

            # Build combined filter query from static config and dynamic filters
            chunk_filter = (
                self.chunk_window_config.chunk_filter_query
                if self.chunk_window_config
                else None
            )
            combined_filter = _build_solr_filter_query(chunk_filter, filters)
            if combined_filter:
                data_params["fq"] = combined_filter
                if isinstance(combined_filter, list):
                    log.info(
                        f"Applying {len(combined_filter)} filter queries (multiple fq params)"
                    )
                    log.debug(f"Filter queries: {combined_filter}")
                else:
                    log.info("Applying 1 filter query (single fq param)")
                    log.debug(f"Filter query: {combined_filter}")
            else:
                log.info("No filter query applied (fq parameter not set)")

            log.debug(f"Final fq param in request: {data_params.get('fq')}")

            try:
                log.info(
                    f"Sending hybrid query to Solr with reranking: reRankDocs={k * 2}, "
                    f"reRankWeight={vector_boost}"
                )
                response = await client.post(
                    f"{self.base_url}/select",
                    data=data_params,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()
                data = response.json()

                chunks = []
                scores = []

                num_docs = data.get("response", {}).get("numFound", 0)
                log.info(f"Solr returned {num_docs} documents for hybrid search")

                for idx, doc in enumerate(data.get("response", {}).get("docs", [])):
                    score = float(doc.get("score", 0))

                    # Log first few documents for debugging filters
                    if filters and idx < 3:
                        log.info(f"Doc {idx} id={doc.get('id')}: score={score}")
                        log.info(f"Doc {idx} available fields: {list(doc.keys())[:10]}")
                        # Try to log the filter field value
                        if isinstance(filters, ComparisonFilter):
                            filter_key = filters.key
                            filter_value = doc.get(filter_key, "FIELD_NOT_FOUND")
                            log.info(f"Doc {idx} {filter_key}={filter_value}")

                    # Apply score threshold
                    if score < score_threshold:
                        log.debug(
                            f"Filtering out document with score {score} < threshold {
                                score_threshold
                            }"
                        )
                        continue

                    chunk = self._doc_to_chunk(doc)
                    if chunk:
                        chunks.append(chunk)
                        scores.append(score)

                log.debug(f"Hybrid search returned {len(chunks)} chunks (filtered from {
                        num_docs
                    } by score threshold)")
                query_chunks_response = QueryChunksResponse(
                    chunks=chunks, scores=scores
                )

                # Apply chunk window expansion if configured
                if self.chunk_window_config is not None:
                    return await self._apply_chunk_window_expansion(
                        initial_response=query_chunks_response,
                        min_chunk_gap=self.chunk_window_config.min_chunk_gap,
                        min_chunk_window=self.chunk_window_config.min_chunk_window,
                    )

                return query_chunks_response

            except httpx.HTTPStatusError as e:
                log.error(
                    f"HTTP error during hybrid search: status={e.response.status_code}"
                )
                try:
                    error_data = e.response.json()
                    log.error(f"Solr error response: {error_data}")
                except ValueError:
                    log.error(f"Solr error response (text): {e.response.text[:500]}")
                raise
            except Exception as e:
                log.exception(f"Error querying Solr with hybrid search: {e}")
                raise

    async def delete(self) -> None:
        """Not implemented - this is a read-only provider."""
        log.warning("Attempted to delete SolrIndex")
        raise NotImplementedError("SolrVectorIO is read-only.")

    def _build_doc_metadata(
        self,
        doc: dict[str, Any],
        chunk_id: Any,
        parent_id: Optional[str],
    ) -> dict[str, Any]:
        """
        Build the full metadata dict for a Solr document.

        Parameters:
            doc: The Solr document dict to read field values from.
            chunk_id: The chunk identifier value.
            parent_id: The parent document identifier, or None if unavailable.

        Returns:
            A metadata dict populated with standard document fields and any
                configured window metadata.

        """
        metadata: dict[str, Any] = {
            "document_id": parent_id,
            "doc_id": parent_id,
            "chunk_id": chunk_id,
            "source": OKP_SOURCE,
        }
        for field in ("title", "resourceName", "chunk_index", "parent_id"):
            if field in doc:
                metadata[field] = doc[field]
        self._add_window_config_metadata(doc, metadata)
        return metadata

    def _create_chunk_metadata(self, metadata: dict[str, Any]) -> ChunkMetadata:
        """Build API chunk metadata from a provider metadata dictionary."""
        return ChunkMetadata(
            chunk_id=(
                str(metadata.get("chunk_id")) if metadata.get("chunk_id") else None
            ),
            document_id=metadata.get("document_id"),
            source=metadata.get("source"),
        )

    def _doc_to_chunk(self, doc: dict[str, Any]) -> Optional[Chunk]:
        """
        Convert a Solr document dictionary into an EmbeddedChunk suitable for search responses.

        Parameters:
            doc (dict[str, Any]): A Solr document dictionary as returned by a Solr JSON query.

        Returns:
            Optional[EmbeddedChunk]: An EmbeddedChunk populated with `chunk_id`,
            `content`, and `metadata` (including parent/document identifiers
            and any configured family/token fields), with embedding metadata
            (model and dimension) set but an empty `embedding` list; returns
            `None` if the document is not a chunk, lacks required content or
            identifiers, or cannot be converted.
        """
        try:
            if not doc.get("is_chunk", True):
                log.info("Skipping non-chunk document")
                return None

            content = doc.get(self.content_field)
            if not content:
                log.warning(
                    f"Content field '{self.content_field}' not found. "
                    f"Available fields: {list(doc.keys())}"
                )
                return None

            chunk_id = (
                doc.get(self.id_field) or doc.get("resourceName") or doc.get("id")
            )
            if not chunk_id:
                log.error("No chunk_id found in Solr document")
                return None

            parent_id = (
                doc.get("parent_id")
                or doc.get("doc_id")
                or (
                    str(chunk_id).rsplit("_chunk_", 1)[0]
                    if "_chunk_" in str(chunk_id)
                    else None
                )
            )

            metadata = self._build_doc_metadata(doc, chunk_id, parent_id)

            embedding = doc.get(self.vector_field)
            if not isinstance(embedding, list):
                embedding = []
            else:
                embedding = [float(x) for x in embedding]

            return EmbeddedChunk(
                chunk_id=str(chunk_id),
                content=content,
                metadata=metadata,
                chunk_metadata=self._create_chunk_metadata(metadata),
                embedding=[],  # can be None
                embedding_model=self.embedding_model,
                embedding_dimension=self.dimension,
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            log.exception(f"Error converting Solr document to Chunk: {e}")
            return None
