from typing import Any

from pydantic import BaseModel, Field

from llama_stack.core.storage.datatypes import KVStoreReference
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class ChunkWindowConfig(BaseModel):
    """Configuration and schema mapping for chunk window expansion feature.

    When configured, enables chunk window expansion which retrieves neighboring chunks
    to provide additional context. This allows the provider to work with any Solr schema
    by mapping field names and controlling expansion behavior.
    """

    # Chunk document fields
    chunk_parent_id_field: str = Field(
        description="Field name for parent document ID in chunk documents"
    )
    chunk_index_field: str = Field(
        description="Field name for chunk index/position in chunk documents"
    )
    chunk_content_field: str = Field(
        description="Field name for chunk text content in chunk documents"
    )
    chunk_token_count_field: str = Field(
        description="Field name for token count in chunk documents"
    )

    # Parent document fields
    parent_id_field: str = Field(
        default="id", description="Field name for document ID in parent documents"
    )
    parent_total_chunks_field: str = Field(
        description="Field name for total chunk count in parent documents"
    )
    parent_total_tokens_field: str = Field(
        description="Field name for total token count in parent documents"
    )

    # Optional parent metadata fields
    parent_content_id_field: str | None = Field(
        default=None,
        description="Field name for content identifier in parent documents",
    )
    parent_content_title_field: str | None = Field(
        default=None, description="Field name for content title in parent documents"
    )
    parent_content_url_field: str | None = Field(
        default=None, description="Field name for content URL in parent documents"
    )

    # Query filters
    chunk_filter_query: str | None = Field(
        default=None,
        description="Filter query to identify chunk documents (e.g., 'is_chunk:true')",
    )

    # Chunk window expansion parameters
    token_budget: int = Field(
        default=2048,
        description="Maximum token budget for expanded context window",
    )
    min_chunk_gap: int = Field(
        default=4,
        description="Minimum gap between chunks to avoid overlap",
    )
    min_chunk_window: int = Field(
        default=4,
        description="Minimum number of chunks in expanded window",
    )


@json_schema_type
class SolrVectorIOConfig(BaseModel):
    """Configuration for Solr Vector IO provider.

    :param solr_url: Base URL of the Solr server (e.g., "http://localhost:8983/solr")
    :param collection_name: Name of the Solr collection to use
    :param vector_field: Name of the field containing DenseVectorField embeddings
    :param embedding_dimension: Dimension of the embedding vectors
    :param persistence: Config for KV store backend (SQLite only for now)
    :param request_timeout: Timeout for Solr requests in seconds
    """

    solr_url: str = Field(description="Base URL of the Solr server")
    collection_name: str = Field(description="Name of the Solr collection to use")
    vector_field: str = Field(
        description="Name of the field containing DenseVectorField embeddings"
    )
    content_field: str = Field(
        description="Name of the field containing chunk text content"
    )
    id_field: str = Field(
        default="id",
        description="Name of the field containing unique document identifier",
    )
    embedding_dimension: int = Field(description="Dimension of the embedding vectors")
    persistence: KVStoreReference | None = Field(
        description="Config for KV store backend (SQLite only for now)", default=None
    )
    request_timeout: int = Field(
        default=30, description="Timeout for Solr requests in seconds"
    )
    chunk_window_config: ChunkWindowConfig | None = Field(
        default=None,
        description="Configuration and schema mapping for chunk window expansion (optional)",
    )

    @classmethod
    def sample_run_config(
        cls,
        __distro_dir__: str,
        solr_url: str = "${env.SOLR_URL:=http://localhost:8983/solr}",
        collection_name: str = "${env.SOLR_COLLECTION:=collection}",
        vector_field: str = "${env.SOLR_VECTOR_FIELD:=embedding}",
        content_field: str = "${env.SOLR_CONTENT_FIELD:=content}",
        embedding_dimension: int = "${env.SOLR_EMBEDDING_DIM:=384}",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "solr_url": solr_url,
            "collection_name": collection_name,
            "vector_field": vector_field,
            "content_field": content_field,
            "embedding_dimension": embedding_dimension,
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="vector_io::solr",
            ).model_dump(exclude_none=True),
            # Example chunk window configuration (uncomment to enable):
            # "chunk_window_config": {
            #     "chunk_parent_id_field": "parent_id",
            #     "chunk_index_field": "chunk_index",
            #     "chunk_content_field": "chunk",
            #     "chunk_token_count_field": "num_tokens",
            #     "parent_total_chunks_field": "total_chunks",
            #     "parent_total_tokens_field": "total_tokens",
            #     "parent_content_id_field": "doc_id",
            #     "parent_content_title_field": "title",
            #     "parent_content_url_field": "reference_url",
            #     "chunk_filter_query": "is_chunk:true"
            # }
        }
