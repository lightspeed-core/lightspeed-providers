from typing import Any, Optional

import httpx
from llama_stack.core.storage.kvstore import kvstore_impl
from llama_stack.log import get_logger
from llama_stack.providers.utils.memory.openai_vector_store_mixin import (
    OpenAIVectorStoreMixin,
)
from llama_stack.providers.utils.memory.vector_store import (
    ChunkForDeletion,
    EmbeddingIndex,
    VectorStoreWithIndex,
)
from llama_stack_api.common.content_types import ImageContentItem, TextContentItem
from llama_stack_api.common.errors import VectorStoreNotFoundError
from llama_stack_api.datatypes import VectorStoresProtocolPrivate
from llama_stack_api.files import Files
from llama_stack_api.inference import Inference
from llama_stack_api.vector_io import (
    Chunk,
    EmbeddedChunk,
    QueryChunksResponse,
    VectorIO,
)
from llama_stack_api.vector_stores import VectorStore
from numpy.typing import NDArray

from .config import ChunkWindowConfig, SolrVectorIOConfig

log = get_logger(name=__name__, category="vector_io::solr")

VERSION = "v1"
VECTOR_DBS_PREFIX = f"vector_stores:solr:{VERSION}::"
OKP_SOURCE = "okp"


class SolrIndex(EmbeddingIndex):
    """
    Read-only Solr vector index implementation using DenseVectorField and KNN search.
    Supports hybrid search using Solr's native query reranking capabilities.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        solr_url: str,
        collection_name: str,
        vector_field: str,
        content_field: str,
        id_field: str,
        dimension: int,
        embedding_model: str,
        request_timeout: int = 30,
        chunk_window_config: Optional[ChunkWindowConfig] = None,
    ):
        self.vector_store = vector_store
        self.solr_url = solr_url.rstrip("/")
        self.collection_name = collection_name
        self.vector_field = vector_field
        self.content_field = content_field
        self.id_field = id_field
        self.dimension = dimension
        self.embedding_model = embedding_model
        self.request_timeout = request_timeout
        self.chunk_window_config = chunk_window_config
        self.base_url = f"{self.solr_url}/{self.collection_name}"
        log.info(
            f"Initialized SolrIndex for collection '{collection_name}' at {
                self.base_url
            }, "
            f"vector_field='{vector_field}', content_field='{
                content_field
            }', dimension={dimension}, "
            f"chunk_window_enabled={chunk_window_config is not None}"
        )

    def _create_http_client(self) -> httpx.AsyncClient:
        """Create an HTTP client configured for Solr connections.

        Uses IPv4 by binding to 0.0.0.0. When Solr runs in a podman container,
        IPv4 is required unless podman has been explicitly configured to support IPv6.
        """
        return httpx.AsyncClient(
            timeout=self.request_timeout,
            transport=httpx.AsyncHTTPTransport(local_address="0.0.0.0"),
        )

    async def initialize(self) -> None:
        """Verify connection to Solr and collection exists."""
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

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray) -> None:
        """Not implemented - this is a read-only provider."""
        log.warning(f"Attempted to add {len(chunks)} chunks to read-only SolrIndex")
        raise NotImplementedError("SolrVectorIO is read-only.")

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Not implemented - this is a read-only provider."""
        log.warning(f"Attempted to delete {
                len(chunks_for_deletion)
            } chunks from read-only SolrIndex")
        raise NotImplementedError("SolrVectorIO is read-only.")

    async def query_vector(
        self,
        embedding: NDArray,
        k: int,
        score_threshold: float,
    ) -> QueryChunksResponse:
        """
        Performs vector similarity search using Solr's KNN query.

        Args:
            embedding: The query embedding vector
            k: Number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            QueryChunksResponse with matching chunks and scores

        """
        log.info(
            f"Performing vector search: k={k}, score_threshold={score_threshold}, "
            f"embedding_dim={len(embedding)}"
        )

        async with self._create_http_client() as client:
            # Solr KNN query using the dense vector field
            # Use knn-search endpoint with JSON body
            # Solr expects format: [f1,f2,f3]
            # Build Solr vector literal
            vector_str = ",".join(str(v) for v in embedding.tolist())

            params = {
                "q": "*:*",  # or query_string if hybrid
                "vector": vector_str,
                "topK": k,
                "rows": k,
                "fl": "*,score",
                "wt": "json",
            }

            if self.chunk_window_config and self.chunk_window_config.chunk_filter_query:
                params["fq"] = self.chunk_window_config.chunk_filter_query

            try:
                response = await client.post(
                    f"{self.base_url}/semantic-search",
                    data=params,  # ✅ form-encoded
                )
                response.raise_for_status()
                data = response.json()

                chunks = []
                scores = []

                for doc in data.get("response", {}).get("docs", []):
                    score = float(doc.get("score", 0.0))
                    if score < score_threshold:
                        continue

                    chunk = self._doc_to_chunk(doc)
                    if not chunk:
                        continue

                    embedded_chunk = EmbeddedChunk(
                        chunk_id=chunk.chunk_id,
                        content=chunk.content,
                        chunk_metadata=chunk.metadata or {},
                        metadata=chunk.metadata or {},
                        embedding=[],  # can be None
                        embedding_model=self.embedding_model,
                        embedding_dimension=self.dimension,
                        metadata_token_count=None,  # optional but required by schema
                    )

                    chunks.append(embedded_chunk)
                    scores.append(score)

                return QueryChunksResponse(chunks=chunks, scores=scores)

            except httpx.HTTPStatusError as e:
                log.error(
                    f"semantic-search failed: status={e.response.status_code}, "
                    f"body={e.response.text[:500]}"
                )
                raise

    async def query_keyword(
        self,
        query_string: str,
        k: int,
        score_threshold: float,
    ) -> QueryChunksResponse:
        """
        Performs keyword-based search using Solr's text search.

        Args:
            query_string: The text query for keyword search
            k: Number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            QueryChunksResponse with matching chunks and scores

        """
        log.info(
            f"Performing keyword search: query='{query_string}', k={k}, "
            f"score_threshold={score_threshold}"
        )

        # Replace ? and * because when the edismax text parser is enabled, they are evaluated as lucene wildcards
        query_string = query_string.replace("?", "").replace("*", "")

        async with self._create_http_client() as client:
            solr_params = {
                "q": query_string,
                "rows": k,
                "fl": "*,score",
                "wt": "json",
            }

            # Add filter query for chunk documents if schema is configured
            if self.chunk_window_config and self.chunk_window_config.chunk_filter_query:
                solr_params["fq"] = self.chunk_window_config.chunk_filter_query
                log.info(f"Applying chunk filter: {
                        self.chunk_window_config.chunk_filter_query
                    }")

            try:
                log.info("Sending keyword query to Solr")
                response = await client.get(
                    f"{self.base_url}/select", params=solr_params
                )
                response.raise_for_status()
                data = response.json()

                chunks = []
                scores = []

                num_docs = data.get("response", {}).get("numFound", 0)
                log.info(f"Solr returned {num_docs} documents for keyword search")

                for doc in data.get("response", {}).get("docs", []):
                    score = float(doc.get("score", 0))

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

                log.debug(
                    f"Keyword search returned {len(chunks)} chunks (filtered from {
                        num_docs
                    } by score threshold)"
                )
                response = QueryChunksResponse(chunks=chunks, scores=scores)

                # Apply chunk window expansion if configured
                if self.chunk_window_config is not None:
                    return await self._apply_chunk_window_expansion(
                        initial_response=response,
                        min_chunk_gap=self.chunk_window_config.min_chunk_gap,
                        min_chunk_window=self.chunk_window_config.min_chunk_window,
                    )

                return response

            except httpx.HTTPStatusError as e:
                log.error(
                    f"HTTP error during keyword search: status={e.response.status_code}"
                )
                raise
            except Exception as e:
                log.exception(f"Error querying Solr with keyword search: {e}")
                raise

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        """
        Hybrid search combining vector similarity and keyword search using Solr's native reranking.

        Args:
            embedding: The query embedding vector
            query_string: The text query for keyword search
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            reranker_type: Type of reranker (ignored, uses Solr's native capabilities)
            reranker_params: Parameters for reranking (e.g., boost values)

        Returns:
            QueryChunksResponse with combined results

        """
        if reranker_params is None:
            reranker_params = {}

        # Get boost parameters, defaulting to equal weighting
        vector_boost = reranker_params.get("vector_boost", 8.0)

        # Replace ? and * because when the edismax text parser is enabled, they are evaluated as lucene wildcards
        query_string = query_string.replace("?", "").replace("*", "")

        log.info(
            f"Performing hybrid search: query='{query_string}', k={k}, "
            f"score_threshold={score_threshold}, vector_boost={vector_boost}, "
        )

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

            # Add filter query for chunk documents if schema is configured
            if self.chunk_window_config and self.chunk_window_config.chunk_filter_query:
                data_params["fq"] = self.chunk_window_config.chunk_filter_query
                log.info(f"Applying chunk filter: {
                        self.chunk_window_config.chunk_filter_query
                    }")

            try:
                log.info(
                    f"Sending hybrid query to Solr with reranking: reRankDocs={k * 2}, "
                    f"reRankWeight={vector_boost}"
                )
                response = await client.post(
                    f"{self.base_url}/hybrid-search",
                    data=data_params,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()
                data = response.json()

                chunks = []
                scores = []

                num_docs = data.get("response", {}).get("numFound", 0)
                log.info(f"Solr returned {num_docs} documents for hybrid search")

                for doc in data.get("response", {}).get("docs", []):
                    score = float(doc.get("score", 0))

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
                except Exception:
                    log.error(f"Solr error response (text): {e.response.text[:500]}")
                raise
            except Exception as e:
                log.exception(f"Error querying Solr with hybrid search: {e}")
                raise

    async def delete(self) -> None:
        """Not implemented - this is a read-only provider."""
        log.warning("Attempted to delete SolrIndex")
        raise NotImplementedError("SolrVectorIO is read-only.")

    async def _fetch_parent_metadata(
        self, client: httpx.AsyncClient, parent_id: str
    ) -> dict[str, Any] | None:
        """
        Fetch parent document metadata using configured field names.

        Args:
            client: HTTP client for making requests
            parent_id: ID of the parent document

        Returns:
            Parent document metadata dict, or None if not found

        """
        schema = self.chunk_window_config

        # Build field list from configured field names
        fields = [
            schema.parent_id_field,
            schema.parent_total_chunks_field,
            schema.parent_total_tokens_field,
        ]

        if schema.parent_content_id_field:
            fields.append(schema.parent_content_id_field)
        if schema.parent_content_title_field:
            fields.append(schema.parent_content_title_field)
        if schema.parent_content_url_field:
            fields.append(schema.parent_content_url_field)

        try:
            log.info(f"Fetching parent metadata for parent_id={parent_id}")
            response = await client.get(
                f"{self.base_url}/select",
                params={
                    "q": f'{schema.parent_id_field}:"{parent_id}"',
                    "fl": ",".join(fields),
                    "wt": "json",
                    "rows": "1",
                },
            )
            response.raise_for_status()
            data = response.json()

            docs = data.get("response", {}).get("docs", [])
            if not docs:
                log.warning(f"Parent document not found: {parent_id}")
                return None

            parent_doc = docs[0]
            log.info(f"Found parent document: total_chunks={
                    parent_doc.get(schema.parent_total_chunks_field)
                }, " f"total_tokens={parent_doc.get(schema.parent_total_tokens_field)}")
            return parent_doc

        except Exception as e:
            log.error(f"Error fetching parent metadata for {parent_id}: {e}")
            return None

    async def _fetch_context_chunks(
        self,
        client: httpx.AsyncClient,
        parent_id: str,
        window_start: int,
        window_end: int,
        boundary_values: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch chunks within a specified index range for a parent document.

        Args:
            client: HTTP client for making requests
            parent_id: ID of the parent document
            window_start: Start index (inclusive)
            window_end: End index (inclusive)
            boundary_values: Optional dict of field name -> value pairs that
                chunks must match (e.g. {'parent_id': 'doc1', 'heading': 'Intro'})

        Returns:
            List of chunk documents sorted by chunk_index

        """
        schema = self.chunk_window_config

        # Build field list for chunks
        fields = [
            schema.chunk_index_field,
            self.content_field,
            schema.chunk_token_count_field,
            schema.chunk_parent_id_field,
        ]

        # Add boundary fields to the field list so they're returned
        if schema.chunk_family_fields:
            for field in schema.chunk_family_fields:
                if field not in fields:
                    fields.append(field)

        # Build query
        query_parts = [
            f'{schema.chunk_parent_id_field}:"{parent_id}"',
            f"{schema.chunk_index_field}:[{window_start} TO {window_end}]",
        ]

        # Add boundary field filters
        if boundary_values:
            for field_name, field_value in boundary_values.items():
                # Skip parent_id since it's already in the query
                if field_name == schema.chunk_parent_id_field:
                    continue
                query_parts.append(f'{field_name}:"{field_value}"')

        # Add filter query if configured
        if schema.chunk_filter_query:
            query_parts.append(schema.chunk_filter_query)

        query = " AND ".join(query_parts)

        try:
            log.info(
                f"Fetching context chunks: parent_id={parent_id}, "
                f"range=[{window_start}, {window_end}], "
                f"boundary_values={boundary_values}"
            )
            response = await client.get(
                f"{self.base_url}/select",
                params={
                    "q": query,
                    "fl": ",".join(fields),
                    # Add buffer for safety
                    "rows": str(window_end - window_start + 20),
                    "sort": f"{schema.chunk_index_field} asc",
                    "wt": "json",
                },
            )
            response.raise_for_status()
            data = response.json()

            chunks = data.get("response", {}).get("docs", [])
            log.info(f"Fetched {len(chunks)} context chunks")
            return chunks

        except Exception as e:
            log.error(f"Error fetching context chunks for {parent_id}: {e}")
            return []

    async def _apply_chunk_window_expansion(
        self,
        initial_response: QueryChunksResponse,
        min_chunk_gap: int,
        min_chunk_window: int,
    ) -> QueryChunksResponse:
        """
        Apply chunk window expansion to query results.

        This method processes the initial query results, fetches parent documents,
        expands context windows around matched chunks, and returns expanded results.

        Uses family_token_budget for chunks that have values for any configured
        chunk_family_fields, and orphan_token_budget for chunks missing all of
        those field values.

        Args:
            initial_response: Initial query response with matched chunks
            min_chunk_gap: Minimum spacing between chunks from same parent
            min_chunk_window: Minimum chunks before windowing applies

        Returns:
            QueryChunksResponse with expanded context windows

        """
        from collections import defaultdict

        schema = self.chunk_window_config
        expanded_chunks = []
        expanded_scores = []

        # Track kept indices by parent to prevent duplicates
        kept_indices_by_parent = defaultdict(list)

        async with self._create_http_client() as client:
            for chunk, score in zip(initial_response.chunks, initial_response.scores):
                # Extract parent_id and chunk_index from metadata
                if not chunk.metadata:
                    log.warning(
                        "Chunk missing metadata, skipping chunk window expansion"
                    )
                    expanded_chunks.append(chunk)
                    expanded_scores.append(score)
                    continue

                parent_id = chunk.metadata.get(schema.chunk_parent_id_field)
                matched_chunk_index = chunk.metadata.get(schema.chunk_index_field)

                if parent_id is None or matched_chunk_index is None:
                    log.warning(
                        "Chunk missing parent_id or chunk_index fields, "
                        "skipping chunk window expansion"
                    )
                    expanded_chunks.append(chunk)
                    expanded_scores.append(score)
                    continue

                # Skip if too close to any already-kept anchor in this parent
                if any(
                    abs(matched_chunk_index - kept) < min_chunk_gap
                    for kept in kept_indices_by_parent[parent_id]
                ):
                    log.debug(
                        f"Skipping chunk at index {matched_chunk_index} "
                        f"(too close to existing anchor)"
                    )
                    continue

                # Keep this anchor
                kept_indices_by_parent[parent_id].append(matched_chunk_index)

                # Fetch parent metadata
                parent_doc = await self._fetch_parent_metadata(client, parent_id)
                if not parent_doc:
                    log.warning(f"Parent document not found for {
                            parent_id
                        }, using original chunk")
                    expanded_chunks.append(chunk)
                    expanded_scores.append(score)
                    continue

                total_chunks = parent_doc.get(schema.parent_total_chunks_field, 0)
                total_tokens = parent_doc.get(schema.parent_total_tokens_field, 0)

                # Build boundary values from matched chunk metadata and
                # determine whether this chunk is an orphan (missing family fields)
                boundary_values = None
                is_orphan = False
                if schema.chunk_family_fields:
                    boundary_values = {}
                    for field in schema.chunk_family_fields:
                        value = chunk.metadata.get(field)
                        if value is not None:
                            boundary_values[field] = value

                    # mark the chunk as an orphan if it is missing values for ALL the family fields.
                    is_orphan = all(
                        chunk.metadata.get(field) is None
                        for field in schema.chunk_family_fields
                    )
                    if is_orphan:
                        log.debug(
                            f"Chunk at index {matched_chunk_index} is an orphan "
                            f"(missing family field values), using orphan_token_budget"
                        )

                token_budget = (
                    schema.orphan_token_budget
                    if is_orphan
                    else schema.family_token_budget
                )

                # If short doc, return all chunks
                if (
                    total_chunks < min_chunk_window
                    or total_tokens <= schema.family_token_budget
                ):
                    log.info(
                        f"Document is short (total_chunks={total_chunks}, "
                        f"total_tokens={total_tokens}), fetching all chunks"
                    )
                    context_chunks = await self._fetch_context_chunks(
                        client, parent_id, 0, max(0, total_chunks - 1)
                    )
                    selected_chunks = context_chunks
                else:
                    log.info(
                        f"Document exceeds token budget (total_chunks={total_chunks}, "
                        f"total_tokens={total_tokens}, "
                        f"{'orphan' if is_orphan else 'family'}_token_budget={token_budget}), "
                        f"expanding window around match"
                    )

                    # Fetch bounded window around match (±10 chunks)
                    window_start = max(0, matched_chunk_index - 10)
                    window_end = (
                        min(total_chunks - 1, matched_chunk_index + 10)
                        if total_chunks > 0
                        else 0
                    )

                    log.info(
                        f"Fetching bounded window: [{window_start}, {window_end}] "
                        f"around match at index {matched_chunk_index}"
                    )
                    context_chunks = await self._fetch_context_chunks(
                        client,
                        parent_id,
                        window_start,
                        window_end,
                        boundary_values=boundary_values,
                    )

                    if not context_chunks:
                        log.warning("No context chunks fetched, using original chunk")
                        expanded_chunks.append(chunk)
                        expanded_scores.append(score)
                        continue

                    # If all fetched chunks fit in the token budget, use them
                    # all directly (no need to run expansion loop)
                    window_tokens = sum(
                        c.get(schema.chunk_token_count_field, 0) for c in context_chunks
                    )
                    if window_tokens <= token_budget:
                        log.info(
                            f"All {len(context_chunks)} context chunks fit in "
                            f"token budget ({window_tokens}/{token_budget}), "
                            f"skipping expansion loop"
                        )
                        selected_chunks = context_chunks
                    else:
                        # Find local match index in the fetched window
                        match_pos = None
                        for i, c in enumerate(context_chunks):
                            if c.get(schema.chunk_index_field) == matched_chunk_index:
                                match_pos = i
                                break

                        if match_pos is None:
                            log.warning(
                                "Matched chunk not found in context window, "
                                "using original chunk"
                            )
                            expanded_chunks.append(chunk)
                            expanded_scores.append(score)
                            continue

                        # Apply token budget expansion
                        selected_chunks = self._expand_chunk_window(
                            context_chunks, match_pos, token_budget
                        )

                # Concatenate selected chunks into final content
                content_parts = []
                for selected_chunk in selected_chunks:
                    content = selected_chunk.get(self.content_field, "")
                    if content:
                        content_parts.append(content)

                final_content = "\n\n".join(content_parts)

                # Build expanded chunk metadata
                expanded_metadata = dict(chunk.metadata) if chunk.metadata else {}
                expanded_metadata["chunk_window_expanded"] = True
                expanded_metadata["chunk_window_size"] = len(selected_chunks)
                expanded_metadata["matched_chunk_index"] = matched_chunk_index

                # Add optional parent metadata if available
                if schema.parent_content_id_field:
                    doc_id = parent_doc.get(schema.parent_content_id_field)
                    if doc_id:
                        expanded_metadata["doc_id"] = doc_id

                if schema.parent_content_title_field:
                    title = parent_doc.get(schema.parent_content_title_field)
                    if title:
                        expanded_metadata["title"] = title

                if schema.parent_content_url_field:
                    url = parent_doc.get(schema.parent_content_url_field)
                    if url:
                        expanded_metadata["reference_url"] = url

                expanded_metadata["source"] = OKP_SOURCE

                # Create expanded chunk
                expanded_chunk = EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    content=final_content,
                    metadata=expanded_metadata,
                    chunk_metadata=expanded_metadata,
                    embedding=[],
                    embedding_model=self.embedding_model,
                    embedding_dimension=self.dimension,
                    metadata_token_count=None,
                )

                expanded_chunks.append(expanded_chunk)
                expanded_scores.append(score)

        log.info(f"Chunk window expansion complete: {
                len(initial_response.chunks)
            } initial chunks -> " f"{len(expanded_chunks)} expanded chunks")

        return QueryChunksResponse(chunks=expanded_chunks, scores=expanded_scores)

    def _expand_chunk_window(
        self, chunks: list[dict[str, Any]], match_index: int, token_budget: int
    ) -> list[dict[str, Any]]:
        """
        Expand context window bidirectionally from matched chunk within token budget.

        This algorithm starts with the matched chunk and expands to prev/next chunks,
        adding adjacent chunks until the token budget is exhausted.

        Args:
            chunks: List of chunk documents with token counts
            match_index: Index of the matched chunk in the list
            token_budget: Maximum total tokens to include

        Returns:
            List of selected chunks sorted by chunk_index

        """
        schema = self.chunk_window_config
        total_tokens = 0
        selected_chunks = []

        n = len(chunks)
        prev_idx = match_index
        next_idx = match_index + 1

        # Always include the matched chunk first
        center_chunk = chunks[match_index]
        total_tokens += center_chunk.get(schema.chunk_token_count_field, 0)
        selected_chunks.append(center_chunk)

        log.info(
            f"Starting chunk window expansion: match_index={match_index}, "
            f"total_chunks={n}, token_budget={token_budget}"
        )

        # Expand bidirectionally
        while total_tokens < token_budget and (prev_idx > 0 or next_idx < n):
            added = False

            # Try to add previous chunk (earlier in document)
            if prev_idx > 0:
                next_chunk = chunks[prev_idx - 1]
                next_tokens = next_chunk.get(schema.chunk_token_count_field, 0)
                if total_tokens + next_tokens <= token_budget:
                    selected_chunks.insert(0, next_chunk)
                    total_tokens += next_tokens
                    prev_idx -= 1
                    added = True
                    log.debug(
                        f"Added prev chunk at index {prev_idx}, total_tokens={total_tokens}"
                    )

            # Try to add next chunk (later in document)
            if next_idx < n:
                next_chunk = chunks[next_idx]
                next_tokens = next_chunk.get(schema.chunk_token_count_field, 0)
                if total_tokens + next_tokens <= token_budget:
                    selected_chunks.append(next_chunk)
                    total_tokens += next_tokens
                    next_idx += 1
                    added = True
                    log.debug(f"Added next chunk at index {next_idx - 1}, total_tokens={
                            total_tokens
                        }")

            # If no chunks could be added, we're done
            if not added:
                break

        # Sort by chunk_index to maintain document order
        selected = sorted(
            selected_chunks, key=lambda c: c.get(schema.chunk_index_field, 0)
        )
        log.info(
            f"Chunk window expansion complete: selected {len(selected)} chunks, "
            f"total_tokens={total_tokens}/{token_budget}"
        )
        return selected

    def _doc_to_chunk(self, doc: dict[str, Any]) -> Chunk | None:
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

            metadata: dict[str, Any] = {
                "document_id": parent_id,
                "doc_id": parent_id,
                "chunk_id": chunk_id,
                "source": OKP_SOURCE,
            }

            # helpful extras if present
            if "title" in doc:
                metadata["title"] = doc["title"]
            if "reference_url" in doc:
                metadata["reference_url"] = doc["reference_url"]
            if "resourceName" in doc:
                metadata["resourceName"] = doc["resourceName"]
            if "chunk_index" in doc:
                metadata["chunk_index"] = doc["chunk_index"]
            if "parent_id" in doc:
                metadata["parent_id"] = doc["parent_id"]
            if self.chunk_window_config.chunk_token_count_field in doc:
                metadata[self.chunk_window_config.chunk_token_count_field] = doc[
                    self.chunk_window_config.chunk_token_count_field
                ]
            # add family fields to the metadata since we need to access them for comparison with other chunks later on
            if self.chunk_window_config.chunk_family_fields:
                for field in self.chunk_window_config.chunk_family_fields:
                    if field in doc:
                        metadata[field] = doc[field]

            embedding = doc.get(self.vector_field)
            if isinstance(embedding, list):
                embedding = [float(x) for x in embedding]
            else:
                embedding = []

            return EmbeddedChunk(
                chunk_id=str(chunk_id),
                content=content,
                metadata=metadata,
                chunk_metadata=metadata,
                embedding=[],  # can be None
                embedding_model=self.embedding_model,
                embedding_dimension=self.dimension,
                metadata_token_count=None,  # optional but required by schema
            )

        except Exception as e:
            log.exception(f"Error converting Solr document to Chunk: {e}")
            return None


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
        files_api: Files | None = None,
    ) -> None:
        super().__init__(inference_api=inference_api, files_api=files_api, kvstore=None)
        self.config = config
        self.inference_api = inference_api
        self.cache = {}
        self.vector_store_table = None
        log.info("SolrVectorIOAdapter instance created")

    async def initialize(self) -> None:
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
                "No persistence configured, skipping KV store and OpenAI vector store initialization"
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

                index = SolrIndex(
                    vector_store=vector_store,
                    solr_url=self.config.solr_url,
                    collection_name=self.config.collection_name,
                    vector_field=self.config.vector_field,
                    content_field=self.config.content_field,
                    id_field=self.config.id_field,
                    embedding_model=self.config.embedding_model,
                    dimension=self.config.embedding_dimension,
                    request_timeout=self.config.request_timeout,
                    chunk_window_config=self.config.chunk_window_config,
                )
                await index.initialize()
                self.cache[vector_store.identifier] = VectorStoreWithIndex(
                    vector_store, index, self.inference_api
                )

        log.info("Solr vector_io adapter initialization complete")

    async def shutdown(self) -> None:
        log.info("Shutting down Solr vector_io adapter")
        # Clean up mixin resources (file batch tasks)
        await super().shutdown()
        log.info("Shutdown complete")

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        """Register a vector store (read-only, just caches the metadata)."""
        log.info(f"Registering vector store: {vector_store.identifier}")
        if self.kvstore is not None:
            key = f"{VECTOR_DBS_PREFIX}{vector_store.identifier}"
            await self.kvstore.set(key=key, value=vector_store.model_dump_json())
            log.info(f"Persisted vector store metadata to KV store: {key}")
        else:
            log.info("No KV store configured, skipping persistence")

        index = SolrIndex(
            vector_store=vector_store,
            solr_url=self.config.solr_url,
            collection_name=self.config.collection_name,
            vector_field=self.config.vector_field,
            content_field=self.config.content_field,
            id_field=self.config.id_field,
            dimension=self.config.embedding_dimension,
            embedding_model=self.config.embedding_model,
            request_timeout=self.config.request_timeout,
            chunk_window_config=self.config.chunk_window_config,
        )
        await index.initialize()
        self.cache[vector_store.identifier] = VectorStoreWithIndex(
            vector_store, index, self.inference_api
        )
        log.info(f"Successfully registered vector store: {vector_store.identifier}")

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        """Unregister a vector store (removes from cache and KV store)."""
        log.info(f"Unregistering vector store: {vector_store_id}")

        if vector_store_id in self.cache:
            del self.cache[vector_store_id]
            log.info(f"Removed vector store from cache: {vector_store_id}")

        if self.kvstore is not None:
            await self.kvstore.delete(key=f"{VECTOR_DBS_PREFIX}{vector_store_id}")
            log.info("Removed from KV store")

        log.info(f"Successfully unregistered vector store: {vector_store_id}")

    async def insert_chunks(
        self,
        vector_store_id: str,
        chunks: list[EmbeddedChunk],
        ttl_seconds: int | None = None,
    ) -> None:
        """Not implemented - this is a read-only provider."""
        log.warning(
            f"Attempted to insert {len(chunks)} chunks into read-only provider "
            f"(vector_store_id={vector_store_id})"
        )
        raise NotImplementedError("SolrVectorIO is read-only.")

    async def query_chunks(
        self,
        vector_store_id: str,
        query: (
            str
            | TextContentItem
            | ImageContentItem
            | list[TextContentItem | ImageContentItem]
        ),
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        """Query chunks from the Solr collection."""
        log.info(f"Query chunks request for vector_store_id={vector_store_id}")
        index = await self._get_and_cache_vector_store_index(vector_store_id)
        result = await index.query_chunks(query, params)
        log.info(f"Query returned {len(result.chunks)} chunks")
        return result

    async def delete_chunks(
        self, store_id: str, chunks_for_deletion: list[ChunkForDeletion]
    ) -> None:
        """Not implemented - this is a read-only provider."""
        log.warning(f"Attempted to delete {
                len(chunks_for_deletion)
            } chunks from read-only provider " f"(store_id={store_id})")
        raise NotImplementedError("SolrVectorIO is read-only.")

    async def _get_and_cache_vector_store_index(
        self, vector_store_id: str
    ) -> VectorStoreWithIndex:
        if vector_store_id in self.cache:
            log.debug(f"Retrieved vector store from cache: {vector_store_id}")
            return self.cache[vector_store_id]

        log.info(f"Vector store not in cache, loading from table: {vector_store_id}")

        if self.vector_store_table is None:
            log.error(f"Vector store table not set, cannot find: {vector_store_id}")
            raise VectorStoreNotFoundError(vector_store_id)

        vector_store = await self.vector_store_table.get_vector_store(vector_store_id)
        if not vector_store:
            log.error(f"Vector store not found: {vector_store_id}")
            raise VectorStoreNotFoundError(vector_store_id)

        log.info(f"Loaded vector store from table: {vector_store_id}")
        index = SolrIndex(
            vector_store=vector_store,
            solr_url=self.config.solr_url,
            collection_name=self.config.collection_name,
            vector_field=self.config.vector_field,
            content_field=self.config.content_field,
            id_field=self.config.id_field,
            dimension=self.config.embedding_dimension,
            embedding_model=self.config.embedding_model,
            request_timeout=self.config.request_timeout,
            chunk_window_config=self.config.chunk_window_config,
        )
        await index.initialize()
        self.cache[vector_store_id] = VectorStoreWithIndex(
            vector_store, index, self.inference_api
        )
        return self.cache[vector_store_id]
