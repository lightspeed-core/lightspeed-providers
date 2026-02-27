from typing import Any

from llama_stack.core.datatypes import AccessRule
from llama_stack_api import Api

from .config import LightspeedAgentsImplConfig


async def get_provider_impl(
    config: LightspeedAgentsImplConfig,
    deps: dict[Api, Any],
    policy: list[AccessRule],
):
    # Configure litellm to drop unsupported params for models that reject them (e.g., top_p).
    # This is safe to set globally since it only affects models that don't support these params.
    import litellm

    litellm.drop_params = True

    from .agents import LightspeedAgentsImpl

    impl = LightspeedAgentsImpl(
        config,
        deps[Api.inference],
        deps[Api.vector_io],
        deps.get(Api.safety),
        deps[Api.tool_runtime],
        deps[Api.tool_groups],
        deps[Api.conversations],
        deps[Api.prompts],
        deps[Api.files],
        policy,
    )
    await impl.initialize()
    return impl
