from typing import Any

from pydantic import BaseModel


class InventoryValidatorConfig(BaseModel):
    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {}
