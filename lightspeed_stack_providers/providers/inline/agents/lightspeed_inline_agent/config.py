import os
from typing import Any, Optional
from llama_stack.log import get_logger
from llama_stack.providers.inline.agents.meta_reference.config import (
    MetaReferenceAgentsImplConfig,
)

from pydantic import BaseModel, FilePath

DEFAULT_SYSTEM_PROMPT = """
You are an intelligent assistant that helps filter a list of available tools based on a user's prompt.
The tools will be used at later stage for tool calling with the same User Prompt.
Each tool is provided with its "tool_name" and a "description".
Your task is to identify which tools are relevant to the user's prompt by considering both the tool_name and description.
Return only the 'tool_name' of the relevant tools as a JSON list of strings.
If no tools are relevant, return an empty JSON list.

Example 1:
Tools List:
    [
        {"tool_name": "create_user", "description": "create a user and return the new user information"},
        {"tool_name": "delete_user", "description": "delete the supplied user"},
        {"tool_name": "read_user_data", "description": "read a user data"}
    ]
User Prompt: "get user information"
Relevant Tools (JSON list): 
    [
        "read_user_data"
    ]

Example 2:
Tools List:
    [
        {"tool_name": "jobs_list", "description": "List Jobs"},
        {"tool_name": "workflow_jobs_list", "description": "List workflow Jobs"},
        {"tool_name": "read_user_data", "description": "read a user data"}
    ]
User Prompt: "get jobs list"
Relevant Tools (JSON list):
    [
        "jobs_list",
        "workflow_jobs_list"
    ]

Example 3:
Tools List:
    [
        {"tool_name": "job_templates_list", "description": "List Job Templates"}, 
        {"tool_name": "workflow_job_templates_list", "description": "List Workflow Job Templates"},
        {"tool_name": "read_user_data", "description": "read a user data"}
    ]
User Prompt: "get job templates list"
Relevant Tools (JSON list):
    [
        "job_templates_list",
        "workflow_job_templates_list"
    ]
"""

logger = get_logger(name=__name__, category="agents")


class ToolsFilter(BaseModel):
    model_id: Optional[str] = None
    enabled: Optional[bool] = True
    # minimum tools to enable filtering
    min_tools: Optional[int] = 10
    # the file path which content to use as system prompt
    system_prompt_path: Optional[FilePath] = None
    # if system_prompt_path is defined this gets overwritten by the content of that file
    # if no system_prompt defined the DEFAULT_SYSTEM_PROMPT is used
    system_prompt: Optional[str] = None
    # tools list to always include when filtering,
    # this may be "knowledge_search" for rag tools,
    # or any other tools that should not be filtered out
    always_include_tools: Optional[list[str]] = []

    def model_post_init(self, context: Any, /) -> None:
        if self.system_prompt_path:
            if not os.path.exists(self.system_prompt_path):
                raise ValueError(
                    f"system_prompt_path: '{self.system_prompt_path}' does not exist"
                )

            if not os.path.isfile(self.system_prompt_path):
                raise ValueError(
                    f"system_prompt_path: '{self.system_prompt_path}' is not a file"
                )

            if not os.access(self.system_prompt_path, os.R_OK):
                raise ValueError(
                    f"system_prompt_path: '{self.system_prompt_path}' is not readable"
                )

            self.system_prompt = self.system_prompt_path.read_text(encoding="utf-8")

        if not self.system_prompt:
            logger.info("use default tools filter system prompt")
            self.system_prompt = DEFAULT_SYSTEM_PROMPT

        logger.info("system_prompt: %s'", self.system_prompt)


class LightspeedAgentsImplConfig(MetaReferenceAgentsImplConfig):
    """Lightspeed agent configuration"""

    tools_filter: Optional[ToolsFilter] = ToolsFilter()

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        config = super().sample_run_config(__distro_dir__)
        config["tools_filter"] = ToolsFilter(
            model_id="${env.INFERENCE_MODEL_FILTER}:}",
            enabled=True,
            min_tools=10,
            system_prompt_path=None,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            always_include_tools=[],
        )
        return config
