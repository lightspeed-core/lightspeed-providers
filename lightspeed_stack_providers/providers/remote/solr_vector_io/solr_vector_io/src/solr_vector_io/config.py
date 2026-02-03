from typing import Any

from pydantic import BaseModel, Field

from llama_stack.core.storage.datatypes import KVStoreReference
from llama_stack_api.schema_utils import json_schema_type


@json_schema_type
class ChunkWindowConfig(BaseModel):
    """
    Schema mapping + parameters for chunk window expansion.

    This tells the provider how to:
      - identify chunk docs vs parent docs
      - find neighboring chunks for a matched chunk
      - read token counts / ordering fields
    """

    # ---- Chunk document fields ----
    chunk_parent_id_field: str = Field(
        description="Field name for parent document ID in chunk documents (e.g. 'parent_id')"
    )
    chunk_index_field: str = Field(
        description="Field name for chunk index/position in chunk documents (e.g. 'chunk_index')"
    )
    chunk_content_field: str = Field(
        description="Field name for chunk text content in chunk documents (e.g. 'chunk')"
    )
    chunk_token_count_field: str = Field(
        description="Field name for token count in chunk documents (e.g. 'num_tokens')"
    )

    # Optional: if you want to explicitly detect chunk docs (recommended for mixed results handlers)
    chunk_is_chunk_field: str | None = Field(
        default=None,
        description="Optional field that marks chunk documents (e.g. 'is_chunk')",
    )

    # ---- Parent document fields ----
    parent_id_field: str = Field(
        default="id",
        description="Field name for document ID in parent documents (e.g. 'id' or 'parent_id')",
    )
    parent_total_chunks_field: str = Field(
        description="Field name for total chunk count in parent documents (e.g. 'total_chunks')"
    )
    parent_total_tokens_field: str = Field(
        description="Field name for total token count in parent documents (e.g. 'total_tokens')"
    )

    # Optional parent metadata fields
    parent_content_id_field: str | None = Field(
        default=None, description="Field name for content identifier in parent documents (e.g. 'doc_id')"
    )
    parent_content_title_field: str | None = Field(
        default=None, description="Field name for content title in parent documents (e.g. 'title')"
    )
    parent_content_url_field: str | None = Field(
        default=None, description="Field name for content URL in parent documents (e.g. 'reference_url')"
    )

    # ---- Query filters ----
    chunk_filter_query: str | None = Field(
        default="is_chunk:true",
        description="Filter query to restrict results to chunk documents (recommended).",
    )

    # ---- Chunk window expansion parameters ----
    token_budget: int = Field(default=2048, description="Max token budget per expanded context window")
    min_chunk_gap: int = Field(default=4, description="Min gap between anchors to avoid overlap")
    min_chunk_window: int = Field(default=4, description="Min number of chunks before windowing applies")


@json_schema_type
class SolrVectorIOConfig(BaseModel):
    """
    Configuration for Solr Vector IO provider.
    """

    # Solr connection
    solr_url: str = Field(description="Base URL of the Solr server (e.g. http://localhost:8983/solr)")
    collection_name: str = Field(description="Name of the Solr collection to use")

    # Fields
    vector_field: str = Field(description="DenseVectorField name (e.g. 'chunk_vector')")
    content_field: str = Field(description="Chunk content field name (e.g. 'chunk')")
    id_field: str = Field(
        default="id",
        description="Unique identifier field. For chunk docs this might be 'resourceName' or 'id'.",
    )

    # Embeddings (required by EmbeddedChunk in your 0.4.3 flow)
    embedding_model: str = Field(
        description="Embedding model identifier used to produce query embeddings (e.g. 'sentence-transformers/all-mpnet-base-v2')"
    )
    embedding_dimension: int = Field(description="Embedding vector dimension (e.g. 384)")

    # Optional: if your handler mixes parents/chunks, give your provider a clue
    chunk_only: bool = Field(
        default=True,
        description="If true, provider will try to return only chunk docs (uses chunk_filter_query when available).",
    )

    # Storage/persistence
    persistence: KVStoreReference | None = Field(default=None, description="KV store backend reference")

    # HTTP
    request_timeout: int = Field(default=30, description="Timeout for Solr requests (seconds)")

    # Chunk window expansion (optional)
    chunk_window_config: ChunkWindowConfig | None = Field(
        default=None,
        description="Schema mapping + params for chunk window expansion",
    )

    @classmethod
    def sample_run_config(
        cls,
        __distro_dir__: str,
        solr_url: str = "${env.SOLR_URL:=http://localhost:8983/solr}",
        collection_name: str = "${env.SOLR_COLLECTION:=portal-rag}",
        vector_field: str = "${env.SOLR_VECTOR_FIELD:=chunk_vector}",
        content_field: str = "${env.SOLR_CONTENT_FIELD:=chunk}",
        id_field: str = "${env.SOLR_ID_FIELD:=resourceName}",
        embedding_model: str = "${env.SOLR_EMBEDDING_MODEL:=sentence-transformers/all-mpnet-base-v2}",
        embedding_dimension: int = "${env.SOLR_EMBEDDING_DIM:=384}",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "solr_url": solr_url,
            "collection_name": collection_name,
            "vector_field": vector_field,
            "content_field": content_field,
            "id_field": id_field,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "persistence": {
                "namespace": "vector_io::solr",
                "backend": "kv_default",
            },
            # Optional chunk window expansion mapping example:
            # "chunk_window_config": {
            #   "chunk_parent_id_field": "parent_id",
            #   "chunk_index_field": "chunk_index",
            #   "chunk_content_field": "chunk",
            #   "chunk_token_count_field": "num_tokens",
            #   "chunk_is_chunk_field": "is_chunk",
            #   "parent_id_field": "id",
            #   "parent_total_chunks_field": "total_chunks",
            #   "parent_total_tokens_field": "total_tokens",
            #   "parent_content_id_field": "doc_id",
            #   "parent_content_title_field": "title",
            #   "parent_content_url_field": "reference_url",
            #   "chunk_filter_query": "is_chunk:true",
            #   "token_budget": 2048,
            #   "min_chunk_gap": 4,
            #   "min_chunk_window": 4,
            # }
        }
