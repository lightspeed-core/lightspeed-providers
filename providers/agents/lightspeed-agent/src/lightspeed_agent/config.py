from typing import Any

from pydantic import BaseModel, Field


class LightspeedAgentConfig(BaseModel):
    api_key: str | None = Field(
        default=None,
        description="The Lightspeed Agent API Key",
    )
    api_url: str | None = Field(
        default=None,
        description="The Lightspeed Agent API URL",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "api_key": "${env.LIGHTSPEED_AGENT_API_KEY:}",
            "api_url": "${env.LIGHTSPEED_AGENT_API_URL:http://localhost:8080/agent}",
        }
