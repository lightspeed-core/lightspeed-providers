from typing import Any

from pydantic import BaseModel


class LightspeedAAPToolConfig(BaseModel):
    """Configuration for Lightspeed AAP Tool Runtime"""

    api_key: str | None = None

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "api_key": "${env.LIGHTSPEED_APP_API_KEY:}",
        }
