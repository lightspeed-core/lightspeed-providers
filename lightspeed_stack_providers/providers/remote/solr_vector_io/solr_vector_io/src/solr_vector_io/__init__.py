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
    impl = SolrVectorIOAdapter(
        config,
        deps[Api.inference],
    )
    await impl.initialize()
    return impl
