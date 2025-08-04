from typing import Any, Optional
from llama_stack.providers.inline.agents.meta_reference.config import (
    MetaReferenceAgentsImplConfig,
)

from pydantic import BaseModel


class ToolsFilter(BaseModel):
    model_id: Optional[str] = None
    enabled: Optional[bool] = True


class LightspeedAgentsImplConfig(MetaReferenceAgentsImplConfig):
    """Lightspeed agent configuration"""

    tools_filter: Optional[ToolsFilter] = ToolsFilter()

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        config = super().sample_run_config(__distro_dir__)
        config["tools_filter"] = ToolsFilter(
            model_id="${env.INFERENCE_MODEL_FILTER}:}",
            enabled=True,
        )
        return config
