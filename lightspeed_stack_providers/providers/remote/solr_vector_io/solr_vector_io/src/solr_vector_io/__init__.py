from typing import Any

from llama_stack_api.datatypes import Api

from .config import ChunkWindowConfig, SolrVectorIOConfig
from .solr import SolrVectorIOAdapter

__all__ = [
    "ChunkWindowConfig",
    "SolrVectorIOConfig",
    "SolrVectorIOAdapter",
    "get_adapter_impl",
]


async def get_adapter_impl(config: SolrVectorIOConfig, deps: dict[Api, Any]):
    """
    Create and return a configured SolrVectorIOAdapter instance.

    Parameters:
        - config (SolrVectorIOConfig): Configuration for the adapter.
        - deps (dict[Api, Any]): Dependency mapping; expects an inference
          client at `deps[Api.inference]`.

    Returns:
        SolrVectorIOAdapter: An initialized adapter instance ready for use.
    """
    impl = SolrVectorIOAdapter(
        config,
        deps[Api.inference],
    )
    await impl.initialize()
    return impl
