from typing import Any

from llama_stack.providers.datatypes import Api

from .config import LightspeedAgentsImplConfig


async def get_provider_impl(
    config: LightspeedAgentsImplConfig, deps: dict[Api, Any]
):
    from .agents import LightspeedAgentsImpl

    impl = LightspeedAgentsImpl(
        config,
        deps[Api.inference],
        deps[Api.vector_io],
        deps[Api.safety],
        deps[Api.tool_runtime],
        deps[Api.tool_groups],
        [],
    )
    await impl.initialize()
    return impl
