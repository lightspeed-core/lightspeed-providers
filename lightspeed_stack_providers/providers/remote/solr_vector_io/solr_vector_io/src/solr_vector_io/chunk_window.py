"""Chunk-window expansion support for Solr query results."""

from collections import defaultdict
from typing import Any, Optional

import httpx
from ogx.log import get_logger
from ogx_api.vector_io import EmbeddedChunk, QueryChunksResponse

from .config import ChunkWindowConfig
from .constants import OKP_SOURCE

log = get_logger(name=__name__, category="vector_io::solr")


class ChunkWindowMixin:  # pylint: disable=too-few-public-methods
    """Expand matching Solr chunks with nearby context within a token budget."""

    async def _fetch_parent_metadata(
        self: Any, client: httpx.AsyncClient, parent_id: str
    ) -> Optional[dict[str, Any]]:
        """
        Fetch parent document metadata using configured field names.

        Parameters:
            client: HTTP client for making requests
            parent_id: ID of the parent document

        Returns:
            Parent document metadata dict, or None if not found

        """
        schema = self.chunk_window_config

        if schema is None:
            log.error("Missing chunk window configuration")
            return None

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

        except Exception as e:  # pylint: disable=broad-exception-caught
            log.error(f"Error fetching parent metadata for {parent_id}: {e}")
            return None

    async def _fetch_context_chunks(
        self: Any,
        client: httpx.AsyncClient,
        parent_id: str,
        window_start: int,
        window_end: int,
        boundary_values: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        # pylint: disable=too-many-locals
        """
        Fetch chunks within a specified index range for a parent document.

        Parameters:
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

        if schema is None:
            log.error("Missing chunk window configuration")
            return []

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

        except Exception as e:  # pylint: disable=broad-exception-caught
            log.error(f"Error fetching context chunks for {parent_id}: {e}")
            return []

    def _get_chunk_boundary_and_budget(
        self: Any,
        chunk: EmbeddedChunk,
        schema: ChunkWindowConfig,
    ) -> tuple[Optional[dict[str, Any]], int, bool]:
        """
        Return (boundary_values, token_budget, is_orphan) for a chunk.

        Parameters:
            chunk: The matched EmbeddedChunk to evaluate.
            schema: ChunkWindowConfig defining family fields and token budgets.

        Returns:
            A tuple of (boundary_values, token_budget, is_orphan): boundary_values
                is a dict of family field values used to scope sibling queries
                (None if no family fields are configured), token_budget is the
                applicable token ceiling, and is_orphan indicates whether the
                chunk is missing all configured family field values.

        """
        if not schema.chunk_family_fields:
            return None, schema.family_token_budget, False

        boundary_values: dict[str, Any] = {}
        for field in schema.chunk_family_fields:
            value = chunk.metadata.get(field)
            if value is not None:
                boundary_values[field] = value

        # Orphan: chunk is missing values for ALL configured family fields
        is_orphan = all(
            chunk.metadata.get(field) is None for field in schema.chunk_family_fields
        )
        if is_orphan:
            log.debug(
                f"Chunk at index {chunk.metadata.get(schema.chunk_index_field)} is an orphan "
                f"(missing family field values), using orphan_token_budget"
            )
        token_budget = (
            schema.orphan_token_budget if is_orphan else schema.family_token_budget
        )
        return boundary_values, token_budget, is_orphan

    async def _select_context_chunks_in_window(
        self: Any,
        client: httpx.AsyncClient,
        parent_id: str,
        matched_chunk_index: int,
        total_chunks: int,
        total_tokens: int,
        token_budget: int,
        boundary_values: Optional[dict[str, Any]],
        schema: ChunkWindowConfig,
        min_chunk_window: int,
    ) -> Optional[list[dict[str, Any]]]:
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        """
        Select context chunks for a matched chunk.

        Returns the selected chunk list, or None when the caller should fall back
        to the original chunk (empty fetch or match position not found).

        Parameters:
            client: The async HTTP client for Solr requests.
            parent_id: The parent document identifier used to scope the chunk query.
            matched_chunk_index: The index of the matched chunk within the document.
            total_chunks: Total number of chunks in the parent document.
            total_tokens: Total token count across all chunks in the document.
            token_budget: Maximum tokens allowed for the context window.
            boundary_values: Family field values used to filter sibling chunks, or None.
            schema: ChunkWindowConfig governing field names and expansion behavior.
            min_chunk_window: Minimum chunk count below which the full
                document is returned.

        Returns:
            The selected list of Solr chunk dicts, or None if the caller
                should fall back to the original chunk.

        """
        if (
            total_chunks < min_chunk_window
            or total_tokens <= schema.family_token_budget
        ):
            log.info(
                f"Document is short (total_chunks={total_chunks}, "
                f"total_tokens={total_tokens}), fetching all chunks"
            )
            return await self._fetch_context_chunks(
                client, parent_id, 0, max(0, total_chunks - 1)
            )

        window_start = max(0, matched_chunk_index - 10)
        window_end = (
            min(total_chunks - 1, matched_chunk_index + 10) if total_chunks > 0 else 0
        )
        log.info(
            f"Document exceeds token budget (total_chunks={total_chunks}, "
            f"total_tokens={total_tokens}, token_budget={token_budget}), "
            f"fetching bounded window: [{window_start}, {window_end}] "
            f"around match at index {matched_chunk_index}"
        )
        context_chunks = await self._fetch_context_chunks(
            client, parent_id, window_start, window_end, boundary_values=boundary_values
        )

        if not context_chunks:
            log.warning("No context chunks fetched, using original chunk")
            return None

        window_tokens = sum(
            c.get(schema.chunk_token_count_field, 0) for c in context_chunks
        )
        if window_tokens <= token_budget:
            log.info(
                f"All {len(context_chunks)} context chunks fit in "
                f"token budget ({window_tokens}/{token_budget}), skipping expansion loop"
            )
            return context_chunks

        match_pos = next(
            (
                i
                for i, c in enumerate(context_chunks)
                if c.get(schema.chunk_index_field) == matched_chunk_index
            ),
            None,
        )
        if match_pos is None:
            log.warning(
                "Matched chunk not found in context window, using original chunk"
            )
            return None

        return self._expand_chunk_window(context_chunks, match_pos, token_budget)

    def _assemble_expanded_chunk(
        self: Any,
        chunk: EmbeddedChunk,
        selected_chunks: list[dict[str, Any]],
        parent_doc: dict[str, Any],
        schema: ChunkWindowConfig,
        matched_chunk_index: int,
    ) -> EmbeddedChunk:
        """
        Build the final EmbeddedChunk from the selected context window.

        Parameters:
            chunk: The original matched EmbeddedChunk whose metadata is used
                as the base.
            selected_chunks: Ordered list of Solr chunk dicts forming the
                context window.
            parent_doc: The parent Solr document dict used to populate content
                ID metadata.
            schema: ChunkWindowConfig governing field names used during assembly.
            matched_chunk_index: The index of the matched chunk within the document.

        Returns:
            A new EmbeddedChunk with concatenated content from the window and
                expanded metadata.

        """
        content_parts = [
            c.get(self.content_field, "")
            for c in selected_chunks
            if c.get(self.content_field)
        ]
        final_content = "\n\n".join(content_parts)

        expanded_metadata = dict(chunk.metadata) if chunk.metadata else {}
        expanded_metadata["chunk_window_expanded"] = True
        expanded_metadata["chunk_window_size"] = len(selected_chunks)
        expanded_metadata["matched_chunk_index"] = matched_chunk_index

        if schema.parent_content_id_field:
            doc_id = parent_doc.get(schema.parent_content_id_field)
            if doc_id:
                expanded_metadata["doc_id"] = doc_id

        if schema.parent_content_title_field:
            title = parent_doc.get(schema.parent_content_title_field)
            if title:
                expanded_metadata["title"] = title

        expanded_metadata["source"] = OKP_SOURCE

        return EmbeddedChunk(
            chunk_id=chunk.chunk_id,
            content=final_content,
            metadata=expanded_metadata,
            chunk_metadata=self._create_chunk_metadata(expanded_metadata),
            embedding=[],
            embedding_model=self.embedding_model,
            embedding_dimension=self.dimension,
        )

    async def _apply_chunk_window_expansion(
        self: Any,
        initial_response: QueryChunksResponse,
        min_chunk_gap: int,
        min_chunk_window: int,
    ) -> QueryChunksResponse:
        # pylint: disable=too-many-locals
        """
        Apply chunk window expansion to query results.

        This method processes the initial query results, fetches parent documents,
        expands context windows around matched chunks, and returns expanded results.

        Uses family_token_budget for chunks that have values for any configured
        chunk_family_fields, and orphan_token_budget for chunks missing all of
        those field values.

        Parameters:
            initial_response: Initial query response with matched chunks
            min_chunk_gap: Minimum spacing between chunks from same parent
            min_chunk_window: Minimum chunks before windowing applies

        Returns:
            QueryChunksResponse with expanded context windows

        """
        schema = self.chunk_window_config

        if schema is None:
            log.error("Missing chunk window configuration")
            return QueryChunksResponse(chunks=[], scores=[])

        expanded_chunks = []
        expanded_scores = []

        # Track kept indices by parent to prevent duplicates
        kept_indices_by_parent: dict[Any, list[Any]] = defaultdict(list)

        async with self._create_http_client() as client:
            for chunk, score in zip(initial_response.chunks, initial_response.scores):
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

                kept_indices_by_parent[parent_id].append(matched_chunk_index)

                parent_doc = await self._fetch_parent_metadata(client, parent_id)
                if not parent_doc:
                    log.warning(
                        f"Parent document not found for {parent_id}, using original chunk"
                    )
                    expanded_chunks.append(chunk)
                    expanded_scores.append(score)
                    continue

                total_chunks = parent_doc.get(schema.parent_total_chunks_field, 0)
                total_tokens = parent_doc.get(schema.parent_total_tokens_field, 0)
                boundary_values, token_budget, _ = self._get_chunk_boundary_and_budget(
                    chunk, schema
                )

                selected_chunks = await self._select_context_chunks_in_window(
                    client,
                    parent_id,
                    matched_chunk_index,
                    total_chunks,
                    total_tokens,
                    token_budget,
                    boundary_values,
                    schema,
                    min_chunk_window,
                )

                if selected_chunks is None:
                    expanded_chunks.append(chunk)
                    expanded_scores.append(score)
                    continue

                expanded_chunk = self._assemble_expanded_chunk(
                    chunk, selected_chunks, parent_doc, schema, matched_chunk_index
                )
                expanded_chunks.append(expanded_chunk)
                expanded_scores.append(score)

        log.info(
            f"Chunk window expansion complete: {len(initial_response.chunks)} "
            f"initial chunks -> {len(expanded_chunks)} expanded chunks"
        )
        return QueryChunksResponse(chunks=expanded_chunks, scores=expanded_scores)

    def _expand_chunk_window(
        self: Any, chunks: list[dict[str, Any]], match_index: int, token_budget: int
    ) -> list[dict[str, Any]]:
        """
        Expand context window bidirectionally from matched chunk within token budget.

        This algorithm starts with the matched chunk and expands to prev/next chunks,
        adding adjacent chunks until the token budget is exhausted.

        Parameters:
            chunks: List of chunk documents with token counts
            match_index: Index of the matched chunk in the list
            token_budget: Maximum total tokens to include

        Returns:
            List of selected chunks sorted by chunk_index

        """
        schema = self.chunk_window_config

        if schema is None:
            log.error("Missing chunk window configuration")
            return []

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

    def _add_window_config_metadata(
        self: Any, doc: dict[str, Any], metadata: dict[str, Any]
    ) -> None:
        """
        Populate metadata with fields sourced from chunk_window_config.

        Parameters:
            doc: The Solr document dict to read field values from.
            metadata: The metadata dict to populate in place.

        """
        if not self.chunk_window_config:
            return
        schema = self.chunk_window_config
        if (
            schema.chunk_online_source_url_field
            and schema.chunk_online_source_url_field in doc
        ):
            metadata["reference_url"] = doc[schema.chunk_online_source_url_field]
        if schema.chunk_source_path_field and schema.chunk_source_path_field in doc:
            metadata["source_path"] = doc[schema.chunk_source_path_field]
        if schema.chunk_token_count_field in doc:
            metadata[schema.chunk_token_count_field] = doc[
                schema.chunk_token_count_field
            ]
        # Family fields are needed later for cross-chunk comparison
        for field in schema.chunk_family_fields or []:
            if field in doc:
                metadata[field] = doc[field]
