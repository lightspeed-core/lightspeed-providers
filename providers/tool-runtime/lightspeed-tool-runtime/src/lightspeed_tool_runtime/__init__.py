from typing import Any, Optional

from pydantic import BaseModel

from llama_stack.distribution.datatypes import Api


from .config import LightspeedToolConfig


class LightspeedToolProviderDataValidator(BaseModel):
    lightspeed_tool_groups_headers: Optional[dict[str, dict[str, str]]] = {"*": {}}


async def get_adapter_impl(config: LightspeedToolConfig, _deps: dict[Api, Any]):
    from .lightspeed import LightspeedToolRuntimeImp

    impl = LightspeedToolRuntimeImp(config)

    await impl.initialize()
    return impl
