from typing import Any

from pydantic import BaseModel

from llama_stack.distribution.datatypes import Api

from .auth_type import AuthType
from .config import LightspeedAAPToolConfig


class LightspeedAAPToolProviderDataValidator(BaseModel):
    lightspeed_aap_api_key: str
    lightspeed_aap_auth_type: AuthType


async def get_adapter_impl(config: LightspeedAAPToolConfig, _deps: dict[Api, Any]):
    from .lightspeed_aap import LightspeedAAPToolRuntimeImp

    impl = LightspeedAAPToolRuntimeImp(config)

    await impl.initialize()
    return impl
