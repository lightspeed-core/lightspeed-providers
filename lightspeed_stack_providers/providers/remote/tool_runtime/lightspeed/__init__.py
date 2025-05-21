from typing import Any, Optional
from typing_extensions import Self

from pydantic import BaseModel, model_validator

from llama_stack.distribution.datatypes import Api

from .auth_type import AuthType
from .config import LightspeedToolConfig


class LightspeedToolProviderDataValidator(BaseModel):
    lightspeed_api_key: str
    lightspeed_auth_type: AuthType
    lightspeed_auth_header: Optional[str] = None

    @model_validator(mode="after")
    def check_llama_stack_model(self) -> Self:
        if (
            self.lightspeed_auth_type == AuthType.Bearer
            and self.lightspeed_auth_header is not None
        ):
            raise ValueError(
                "header is not required when using bearer authentication type"
            )
        if (
            self.lightspeed_auth_type == AuthType.Header
            and self.lightspeed_auth_header is None
        ):
            raise ValueError("header is required when using header authentication type")

        return self


async def get_adapter_impl(config: LightspeedToolConfig, _deps: dict[Api, Any]):
    from .lightspeed import LightspeedToolRuntimeImp

    impl = LightspeedToolRuntimeImp(config)

    await impl.initialize()
    return impl
