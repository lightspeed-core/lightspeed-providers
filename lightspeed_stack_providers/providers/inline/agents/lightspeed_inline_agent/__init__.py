from typing import Any

from llama_stack.core.datatypes import AccessRule
from llama_stack_api import Api

from .config import LightspeedAgentsImplConfig


async def get_provider_impl(
    config: LightspeedAgentsImplConfig,
    deps: dict[Api, Any],
    policy: list[AccessRule],
):
    """
    Create and initialize a LightspeedAgentsImpl.

    Configure litellm to drop unsupported params for models that reject them (e.g., top_p).
    This is safe to set globally since it only affects models that don't support these params.

    Parameters:
        config (LightspeedAgentsImplConfig): Configuration for the LightspeedAgentsImpl.
        deps (dict[Api, Any]): Mapping from Api enum keys to required service
        implementations; must contain keys for inference, vector_io,
        tool_runtime, tool_groups, conversations, prompts, files, and
        connectors. The safety service may be omitted.
        policy (list[AccessRule]): Access rules to apply to the created implementation.

    Returns:
        LightspeedAgentsImpl: An initialized LightspeedAgentsImpl instance ready for use.
    """
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
        deps[Api.connectors],
        policy,
    )
    await impl.initialize()
    return impl
