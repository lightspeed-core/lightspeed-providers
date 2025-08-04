from .lightspeed import LightspeedRemoteAgentProvider
from .config import LightspeedAgentConfig


async def get_adapter_impl(config: LightspeedAgentConfig, _deps):
    impl = LightspeedRemoteAgentProvider(config)
    await impl.initialize()
    return impl
