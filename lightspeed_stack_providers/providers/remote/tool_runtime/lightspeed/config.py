from typing import Any

from pydantic import BaseModel


class LightspeedToolConfig(BaseModel):
    """Configuration for Lightspeed Tool Runtime"""

    api_key: str | None = None

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "api_key": "${env.LIGHTSPEED_API_KEY:}",
        }
