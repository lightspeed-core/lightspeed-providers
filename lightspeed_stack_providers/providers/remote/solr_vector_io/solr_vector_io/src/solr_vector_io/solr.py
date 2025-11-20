from typing import Any

import httpx
import numpy as np
from llama_stack.apis.common.errors import VectorStoreNotFoundError
from llama_stack.apis.files.files import Files
from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import (
    Chunk,
    ChunkMetadata,
    QueryChunksResponse,
    VectorIO,
)
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import Api, VectorDBsProtocolPrivate
from llama_stack.providers.utils.inference.prompt_adapter import (
    interleaved_content_as_str,
)
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.memory.openai_vector_store_mixin import (
    OpenAIVectorStoreMixin,
)
from llama_stack.providers.utils.memory.vector_store import (
    ChunkForDeletion,
    EmbeddingIndex,
    VectorDBWithIndex,
)
from numpy.typing import NDArray

from .config import SolrVectorIOConfig

log = get_logger(name=__name__, category="vector_io::solr")

VERSION = "v1"
VECTOR_DBS_PREFIX = f"vector_dbs:solr:{VERSION}::"


class SolrIndex(EmbeddingIndex):
    """
    Read-only Solr vector index implementation using DenseVectorField and KNN search.
    Supports hybrid search using Solr's native query reranking capabilities.
    """

    def __init__(
        self,
        vector_db: VectorDB,
        solr_url: str,
        collection_name: str,
        vector_field: str,
        content_field: str,
        id_field: str,
        dimension: int,
        request_timeout: int = 30,
        chunk_window_config=None,
    ):
        self.vector_db = vector_db
        self.solr_url = solr_url.rstrip("/")
        self.collection_name = collection_name
        self.vector_field = vector_field
        self.content_field = content_field
        self.id_field = id_field
        self.dimension = dimension
        self.request_timeout = request_timeout
        self.chunk_window_config = chunk_window_config
        self.base_url = f"{self.solr_url}/{self.collection_name}"
        log.debug(
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
                raise RuntimeError(
                    f"Failed to connect to Solr collection {
                        self.collection_name
                    }: HTTP {e.response.status_code}"
                ) from e
            except Exception as e:
                log.exception(
                    f"Error connecting to Solr collection {self.collection_name}"
                )
                raise RuntimeError(
                    f"Error connecting to Solr collection {self.collection_name}: {e}"
                ) from e

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray):
        """Not implemented - this is a read-only provider."""
        log.warning(f"Attempted to add {len(chunks)} chunks to read-only SolrIndex")
        raise NotImplementedError("SolrVectorIO is read-only.")

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Not implemented - this is a read-only provider."""
        log.warning(
            f"Attempted to delete {
                len(chunks_for_deletion)
            } chunks from read-only SolrIndex"
        )
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
        # log.debug(
        #     f"Performing vector search: k={k}, score_threshold={score_threshold}, "
        #     f"embedding_dim={len(embedding)}"
        # )

        async with self._create_http_client() as client:
            # Solr KNN query using the dense vector field
            # Use knn-search endpoint with JSON body
            # Solr expects format: [f1,f2,f3]
            vector_list = embedding.tolist()

            # Build params for knn-search endpoint
            solr_params = {
                "q": f"{{!knn f={self.vector_field} topK={k}}}{vector_list}",
                "rows": str(k),
                "fl": "*, score",
                "fq": ["product:*openshift*"],
            }

            # Add filter query for chunk documents if schema is configured
            if self.chunk_window_config and self.chunk_window_config.chunk_filter_query:
                solr_params["fq"].append(self.chunk_window_config.chunk_filter_query)
                log.debug(
                    f"Applying chunk filter: {
                        self.chunk_window_config.chunk_filter_query
                    }"
                )

            # Wrap in params structure for knn-search endpoint
            json_body = {"params": solr_params}

            try:
                log.debug(
                    f"Sending KNN query to Solr: topK={k}, field={self.vector_field}"
                )
                response = await client.post(
                    f"{self.base_url}/semantic-search?wt=json",
                    json=json_body,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()

                chunks = []
                scores = []

                num_docs = data.get("response", {}).get("numFound", 0)
                log.debug(f"Solr returned {num_docs} documents for vector search")

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

                    # Extract chunk from document
                    chunk = self._doc_to_chunk(doc)
                    if chunk:
                        chunks.append(chunk)
                        scores.append(score)

                log.info(
                    f"Vector search returned {len(chunks)} chunks (filtered from {
                        num_docs
                    } by score threshold)"
                )
                response = QueryChunksResponse(chunks=chunks, scores=scores)

                # Apply chunk window expansion if configured
                if self.chunk_window_config is not None:
                    return await self._apply_chunk_window_expansion(
                        initial_response=response,
                        token_budget=self.chunk_window_config.token_budget,
                        min_chunk_gap=self.chunk_window_config.min_chunk_gap,
                        min_chunk_window=self.chunk_window_config.min_chunk_window,
                    )

                return response

            except httpx.HTTPStatusError as e:
                log.error(
                    f"HTTP error during vector search: status={e.response.status_code}"
                )
                raise
            except Exception as e:
                log.exception(f"Error querying Solr with vector search: {e}")
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
        log.debug(
            f"Performing keyword search: query='{query_string}', k={k}, "
            f"score_threshold={score_threshold}"
        )

        async with self._create_http_client() as client:
            solr_params = {
                "q": query_string,
                "rows": k,
                "fl": "*, score",
                "fq": ["product:*openshift*"],
                "wt": "json",
                "defType": "edismax",  # Use extended DisMax for better text search
            }

            # Add filter query for chunk documents if schema is configured
            if self.chunk_window_config and self.chunk_window_config.chunk_filter_query:
                solr_params["fq"].append(self.chunk_window_config.chunk_filter_query)
                log.debug(
                    f"Applying chunk filter: {
                        self.chunk_window_config.chunk_filter_query
                    }"
                )

            try:
                log.debug("Sending keyword query to Solr using edismax parser")
                response = await client.get(
                    f"{self.base_url}/select", params=solr_params
                )
                response.raise_for_status()
                data = response.json()

                chunks = []
                scores = []

                num_docs = data.get("response", {}).get("numFound", 0)
                log.debug(f"Solr returned {num_docs} documents for keyword search")

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

                log.info(
                    f"Keyword search returned {len(chunks)} chunks (filtered from {
                        num_docs
                    } by score threshold)"
                )
                response = QueryChunksResponse(chunks=chunks, scores=scores)

                # Apply chunk window expansion if configured
                if self.chunk_window_config is not None:
                    return await self._apply_chunk_window_expansion(
                        initial_response=response,
                        token_budget=self.chunk_window_config.token_budget,
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
        vector_boost = reranker_params.get("vector_boost", 1.0)
        keyword_boost = reranker_params.get("keyword_boost", 1.0)

        log.debug(
            f"Performing hybrid search: query='{query_string}', k={k}, "
            f"score_threshold={score_threshold}, vector_boost={vector_boost}, "
            f"keyword_boost={keyword_boost}"
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
                "rq": f"{{!rerank reRankQuery=$rqq reRankDocs={k * 2} reRankWeight={vector_boost}}}",
                "rqq": f"{{!knn f={self.vector_field} topK={k * 2}}}{vector_str}",
                "rows": k,
                "fl": "*, score",
                "fq": ["product:*openshift*"],
                "wt": "json",
                "defType": "edismax",
            }
            # Note: keyword_boost can be incorporated in future by discovering schema fields

            # Add filter query for chunk documents if schema is configured
            if self.chunk_window_config and self.chunk_window_config.chunk_filter_query:
                solr_params["fq"].append(self.chunk_window_config.chunk_filter_query)
                log.debug(
                    f"Applying chunk filter: {
                        self.chunk_window_config.chunk_filter_query
                    }"
                )

            try:
                log.debug(
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
                log.debug(f"Solr returned {num_docs} documents for hybrid search")

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

                log.info(
                    f"Hybrid search returned {len(chunks)} chunks (filtered from {
                        num_docs
                    } by score threshold)"
                )
                response = QueryChunksResponse(chunks=chunks, scores=scores)

                # Apply chunk window expansion if configured
                if self.chunk_window_config is not None:
                    return await self._apply_chunk_window_expansion(
                        initial_response=response,
                        token_budget=self.chunk_window_config.token_budget,
                        min_chunk_gap=self.chunk_window_config.min_chunk_gap,
                        min_chunk_window=self.chunk_window_config.min_chunk_window,
                    )

                return response

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

    async def delete(self):
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
            log.debug(f"Fetching parent metadata for parent_id={parent_id}")
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
            log.debug(
                f"Found parent document: total_chunks={
                    parent_doc.get(schema.parent_total_chunks_field)
                }, "
                f"total_tokens={parent_doc.get(schema.parent_total_tokens_field)}"
            )
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
    ) -> list[dict[str, Any]]:
        """
        Fetch chunks within a specified index range for a parent document.

        Args:
            client: HTTP client for making requests
            parent_id: ID of the parent document
            window_start: Start index (inclusive)
            window_end: End index (inclusive)

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

        # Build query
        query_parts = [
            f'{schema.chunk_parent_id_field}:"{parent_id}"',
            f"{schema.chunk_index_field}:[{window_start} TO {window_end}]",
        ]

        # Add filter query if configured
        if schema.chunk_filter_query:
            query_parts.append(schema.chunk_filter_query)

        query = " AND ".join(query_parts)

        try:
            log.debug(
                f"Fetching context chunks: parent_id={parent_id}, "
                f"range=[{window_start}, {window_end}]"
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
            log.debug(f"Fetched {len(chunks)} context chunks")
            return chunks

        except Exception as e:
            log.error(f"Error fetching context chunks for {parent_id}: {e}")
            return []

    async def _apply_chunk_window_expansion(
        self,
        initial_response: QueryChunksResponse,
        token_budget: int,
        min_chunk_gap: int,
        min_chunk_window: int,
    ) -> QueryChunksResponse:
        """
        Apply chunk window expansion to query results.

        This method processes the initial query results, fetches parent documents,
        expands context windows around matched chunks, and returns expanded results.

        Args:
            initial_response: Initial query response with matched chunks
            token_budget: Maximum tokens per context window
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
                    log.warning(
                        f"Parent document not found for {
                            parent_id
                        }, using original chunk"
                    )
                    expanded_chunks.append(chunk)
                    expanded_scores.append(score)
                    continue

                total_chunks = parent_doc.get(schema.parent_total_chunks_field, 0)
                total_tokens = parent_doc.get(schema.parent_total_tokens_field, 0)

                # If short doc, return all chunks
                if total_chunks < min_chunk_window or total_tokens <= token_budget:
                    log.debug(
                        f"Document is short (total_chunks={total_chunks}, "
                        f"total_tokens={total_tokens}), fetching all chunks"
                    )
                    context_chunks = await self._fetch_context_chunks(
                        client, parent_id, 0, max(0, total_chunks - 1)
                    )
                    selected_chunks = context_chunks
                else:
                    # Fetch bounded window around match (±10 chunks)
                    window_start = max(0, matched_chunk_index - 10)
                    window_end = (
                        min(total_chunks - 1, matched_chunk_index + 10)
                        if total_chunks > 0
                        else 0
                    )

                    log.debug(
                        f"Fetching bounded window: [{window_start}, {window_end}] "
                        f"around match at index {matched_chunk_index}"
                    )
                    context_chunks = await self._fetch_context_chunks(
                        client, parent_id, window_start, window_end
                    )

                    if not context_chunks:
                        log.warning("No context chunks fetched, using original chunk")
                        expanded_chunks.append(chunk)
                        expanded_scores.append(score)
                        continue

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

                # Create ChunkMetadata for expanded chunk
                expanded_chunk_metadata = ChunkMetadata(
                    chunk_id=chunk.stored_chunk_id,
                    document_id=expanded_metadata.get("parent_id"),
                    source=expanded_metadata.get("reference_url"),
                )

                # Create expanded chunk
                expanded_chunk = Chunk(
                    content=final_content,
                    metadata=expanded_metadata,
                    stored_chunk_id=chunk.stored_chunk_id,
                    chunk_metadata=expanded_chunk_metadata,
                )

                expanded_chunks.append(expanded_chunk)
                expanded_scores.append(score)

        log.info(
            f"Chunk window expansion complete: {
                len(initial_response.chunks)
            } initial chunks -> "
            f"{len(expanded_chunks)} expanded chunks"
        )

        return QueryChunksResponse(chunks=expanded_chunks, scores=expanded_scores)

    def _expand_chunk_window(
        self, chunks: list[dict[str, Any]], match_index: int, token_budget: int
    ) -> list[dict[str, Any]]:
        """
        Expand context window bidirectionally from matched chunk within token budget.

        This algorithm starts with the matched chunk and expands left/right,
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
        left = match_index
        right = match_index + 1

        # Always include the matched chunk first
        center_chunk = chunks[match_index]
        total_tokens += center_chunk.get(schema.chunk_token_count_field, 0)
        selected_chunks.append(center_chunk)

        log.debug(
            f"Starting chunk window expansion: match_index={match_index}, "
            f"total_chunks={n}, token_budget={token_budget}"
        )

        # Expand bidirectionally
        while total_tokens < token_budget and (left > 0 or right < n):
            added = False

            # Try to add chunk to the left
            if left > 0:
                next_chunk = chunks[left - 1]
                next_tokens = next_chunk.get(schema.chunk_token_count_field, 0)
                if total_tokens + next_tokens <= token_budget:
                    selected_chunks.insert(0, next_chunk)
                    total_tokens += next_tokens
                    left -= 1
                    added = True
                    log.debug(
                        f"Added left chunk at index {left}, total_tokens={total_tokens}"
                    )

            # Try to add chunk to the right
            if right < n:
                next_chunk = chunks[right]
                next_tokens = next_chunk.get(schema.chunk_token_count_field, 0)
                if total_tokens + next_tokens <= token_budget:
                    selected_chunks.append(next_chunk)
                    total_tokens += next_tokens
                    right += 1
                    added = True
                    log.debug(
                        f"Added right chunk at index {right - 1}, total_tokens={
                            total_tokens
                        }"
                    )

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
        """
        Convert a Solr document to a Chunk object.

        This expects documents to have fields compatible with the Chunk schema.
        The exact mapping may need to be customized based on your Solr schema.
        """
        try:
            # Remove Solr-specific fields
            clean_doc = {
                k: v for k, v in doc.items() if not k.startswith("_") and k != "score"
            }

            # Extract content from configured field
            content = clean_doc.pop(self.content_field, None)

            if content is None:
                log.warning(
                    f"Content field '{self.content_field}' not found in document. "
                    f"Available fields: {list(clean_doc.keys())}"
                )
                return None

            # Extract embedding if present (for potential future use)
            embedding = clean_doc.pop(self.vector_field, None)
            if isinstance(embedding, list):
                embedding = [float(x) for x in embedding]

            # Extract chunk_id from configured id_field
            chunk_id = clean_doc.pop(self.id_field, None)

            # Remaining fields become metadata
            metadata = clean_doc

            # Create ChunkMetadata with available information
            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=metadata.get("parent_id"),
                source=metadata.get("reference_url"),
            )

            log.debug(f"Converted Solr document to chunk: chunk_id={chunk_id}")
            return Chunk(
                content=content,
                metadata=metadata,
                embedding=embedding,
                stored_chunk_id=chunk_id,
                chunk_metadata=chunk_metadata,
            )

        except Exception as e:
            log.exception(f"Error converting Solr document to Chunk: {e}")
            return None


class SolrVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorDBsProtocolPrivate):
    """
    Read-only Solr VectorIO adapter.

    This adapter provides read-only access to Solr collections for vector search.
    Write operations (insert_chunks, delete_chunks, etc.) are not supported.
    """

    def __init__(
        self,
        config: SolrVectorIOConfig,
        inference_api: Api.inference,
        files_api: Files | None = None,
    ) -> None:
        self.config = config
        self.inference_api = inference_api
        self.files_api = files_api
        self.kvstore = None
        self.cache = {}
        self.openai_vector_stores = {}
        self.vector_db_store = None
        log.debug("SolrVectorIOAdapter instance created")

    async def initialize(self) -> None:
        log.info("Initializing Solr vector_io adapter")
        log.debug(
            f"Configuration: solr_url={self.config.solr_url}, "
            f"collection={self.config.collection_name}, "
            f"vector_field={self.config.vector_field}, "
            f"dimension={self.config.embedding_dimension}"
        )

        if self.config.persistence is not None:
            self.kvstore = await kvstore_impl(self.config.persistence)
            log.debug("KV store initialized")

            # Initialize OpenAI vector stores support (read-only) - requires kvstore
            await self.initialize_openai_vector_stores()
            log.debug("OpenAI vector stores initialized")
        else:
            log.debug(
                "No persistence configured, skipping KV store and OpenAI vector store initialization"
            )

        # Load any persisted vector DBs
        if self.kvstore is not None:
            start_key = VECTOR_DBS_PREFIX
            end_key = f"{VECTOR_DBS_PREFIX}\xff"
            stored_vector_dbs = await self.kvstore.values_in_range(start_key, end_key)

            log.info(
                f"Loading {len(stored_vector_dbs)} persisted vector DBs from KV store"
            )
            for vector_db_data in stored_vector_dbs:
                vector_db = VectorDB.model_validate_json(vector_db_data)
                log.debug(f"Loading vector DB: {vector_db.identifier}")

                index = SolrIndex(
                    vector_db=vector_db,
                    solr_url=self.config.solr_url,
                    collection_name=self.config.collection_name,
                    vector_field=self.config.vector_field,
                    content_field=self.config.content_field,
                    id_field=self.config.id_field,
                    dimension=self.config.embedding_dimension,
                    request_timeout=self.config.request_timeout,
                    chunk_window_config=self.config.chunk_window_config,
                )
                await index.initialize()
                self.cache[vector_db.identifier] = VectorDBWithIndex(
                    vector_db, index, self.inference_api
                )

        log.info("Solr vector_io adapter initialization complete")

    async def shutdown(self) -> None:
        log.info("Shutting down Solr vector_io adapter")
        # Clean up any resources if needed
        # (No parent class shutdown to call in 0.2.22)
        log.debug("Shutdown complete")

    async def register_vector_db(self, vector_db: VectorDB) -> None:
        """Register a vector DB (read-only, just caches the metadata)."""
        log.info(f"Registering vector DB: {vector_db.identifier}")
        if self.kvstore is not None:
            key = f"{VECTOR_DBS_PREFIX}{vector_db.identifier}"
            await self.kvstore.set(key=key, value=vector_db.model_dump_json())
            log.debug(f"Persisted vector DB metadata to KV store: {key}")
        else:
            log.debug("No KV store configured, skipping persistence")

        index = SolrIndex(
            vector_db=vector_db,
            solr_url=self.config.solr_url,
            collection_name=self.config.collection_name,
            vector_field=self.config.vector_field,
            content_field=self.config.content_field,
            id_field=self.config.id_field,
            dimension=self.config.embedding_dimension,
            request_timeout=self.config.request_timeout,
            chunk_window_config=self.config.chunk_window_config,
        )
        await index.initialize()
        self.cache[vector_db.identifier] = VectorDBWithIndex(
            vector_db, index, self.inference_api
        )
        log.info(f"Successfully registered vector DB: {vector_db.identifier}")

    async def unregister_vector_db(self, vector_db_id: str) -> None:
        """Unregister a vector DB (removes from cache and KV store)."""
        log.info(f"Unregistering vector DB: {vector_db_id}")

        if vector_db_id in self.cache:
            del self.cache[vector_db_id]
            log.debug(f"Removed vector DB from cache: {vector_db_id}")

        if self.kvstore is not None:
            await self.kvstore.delete(key=f"{VECTOR_DBS_PREFIX}{vector_db_id}")
            log.debug("Removed from KV store")

        log.info(f"Successfully unregistered vector DB: {vector_db_id}")

    async def insert_chunks(
        self, vector_db_id: str, chunks: list[Chunk], ttl_seconds: int | None = None
    ) -> None:
        """Not implemented - this is a read-only provider."""
        log.warning(
            f"Attempted to insert {len(chunks)} chunks into read-only provider "
            f"(vector_db_id={vector_db_id})"
        )
        raise NotImplementedError("SolrVectorIO is read-only.")

    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        """Query chunks from the Solr collection."""
        log.debug(f"Query chunks request for vector_db_id={vector_db_id}")
        index = await self._get_and_cache_vector_db_index(vector_db_id)

        if params is None:
            params = {}
        k = params.get("max_chunks", 3)
        mode = params.get("mode")
        embedding = params.get("embedding")

        score_threshold = params.get("score_threshold", 0.0)
        vector_boost = params.get("rerank_vector_boost", 1.0)
        keyword_boost = params.get("rerank_keyword_boost", 1.0)

        score_threshold = 0.0
        vector_boost = 1.0
        keyword_boost = 1.0

        # see docstring for query_hybrid. tl;dr: the "reranker type" is a llama-stack thing and may not be applicable to Solr (if it is applicable, we will implement it later)
        solr_reranker_type = "doesntmatter"
        solr_reranker_params = {
            vector_boost: vector_boost,
            keyword_boost: keyword_boost,
        }

        query_string = interleaved_content_as_str(query)

        if mode == "keyword":
            result = await index.index.query_keyword(query_string, k, score_threshold)
        else:
            # auto-generate the embedding
            embeddings_response = await index.inference_api.openai_embeddings(
                index.vector_db.embedding_model, [query_string]
            )
            embedding = embeddings_response.data[0].embedding

            query_vector = np.array(embedding, dtype=np.float32)

            if mode == "hybrid":
                # self,
                # embedding: NDArray,
                # query_string: str,
                # k: int,
                # score_threshold: float,
                # reranker_type: str,
                # reranker_params: dict[str, Any] | None = None,

                result = await index.index.query_hybrid(
                    embedding=query_vector,
                    query_string=query_string,
                    k=k,
                    score_threshold=score_threshold,
                    reranker_type=solr_reranker_type,
                    reranker_params=solr_reranker_params,
                )
            else:
                result = await index.index.query_vector(
                    query_vector, k, score_threshold
                )

        log.debug(f"Query returned {len(result.chunks)} chunks")
        return result

    async def delete_chunks(
        self, store_id: str, chunks_for_deletion: list[ChunkForDeletion]
    ) -> None:
        """Not implemented - this is a read-only provider."""
        log.warning(
            f"Attempted to delete {
                len(chunks_for_deletion)
            } chunks from read-only provider "
            f"(store_id={store_id})"
        )
        raise NotImplementedError("SolrVectorIO is read-only.")

    async def _get_and_cache_vector_db_index(
        self, vector_db_id: str
    ) -> VectorDBWithIndex:
        if vector_db_id in self.cache:
            log.debug(f"Retrieved vector DB from cache: {vector_db_id}")
            return self.cache[vector_db_id]

        log.debug(f"Vector DB not in cache, loading from store: {vector_db_id}")

        if self.vector_db_store is None:
            log.error(f"Vector DB store not set, cannot find: {vector_db_id}")
            raise VectorStoreNotFoundError(vector_db_id)

        vector_db = await self.vector_db_store.get_vector_db(vector_db_id)
        if not vector_db:
            log.error(f"Vector DB not found: {vector_db_id}")
            raise VectorStoreNotFoundError(vector_db_id)

        log.info(f"Loaded vector DB from store: {vector_db_id}")
        index = SolrIndex(
            vector_db=vector_db,
            solr_url=self.config.solr_url,
            collection_name=self.config.collection_name,
            vector_field=self.config.vector_field,
            content_field=self.config.content_field,
            id_field=self.config.id_field,
            dimension=self.config.embedding_dimension,
            request_timeout=self.config.request_timeout,
            chunk_window_config=self.config.chunk_window_config,
        )
        await index.initialize()
        self.cache[vector_db_id] = VectorDBWithIndex(
            vector_db, index, self.inference_api
        )
        return self.cache[vector_db_id]
