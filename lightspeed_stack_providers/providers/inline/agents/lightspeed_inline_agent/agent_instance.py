import json
import uuid
from collections.abc import AsyncGenerator

from llama_stack.apis.agents import AgentConfig, AgentTurnCreateRequest, StepType
from llama_stack.distribution.datatypes import AccessRule
from llama_stack.log import get_logger
from llama_stack.providers.inline.agents.meta_reference.agent_instance import ChatAgent
from llama_stack.providers.utils.telemetry import tracing

from llama_stack.apis.inference import (
    Inference,
    UserMessage,
    SamplingParams,
    TopPSamplingStrategy,
)
from llama_stack.apis.safety import Safety
from llama_stack.apis.tools import ToolGroups, ToolRuntime
from llama_stack.apis.vector_io import VectorIO
from llama_stack.providers.utils.kvstore import KVStore

from .config import ToolsFilter

logger = get_logger(name=__name__, category="agents")


class LightspeedChatAgent(ChatAgent):
    def __init__(
        self,
        agent_id: str,
        agent_config: AgentConfig,
        inference_api: Inference,
        safety_api: Safety,
        tool_runtime_api: ToolRuntime,
        tool_groups_api: ToolGroups,
        vector_io_api: VectorIO,
        persistence_store: KVStore,
        created_at: str,
        policy: list[AccessRule],
        tools_filter_config: ToolsFilter = None,
    ):
        super().__init__(
            agent_id,
            agent_config,
            inference_api,
            safety_api,
            tool_runtime_api,
            tool_groups_api,
            vector_io_api,
            persistence_store,
            created_at,
            policy,
        )
        self.tools_filter_config = tools_filter_config

    async def create_and_execute_turn(
        self, request: AgentTurnCreateRequest
    ) -> AsyncGenerator:
        # Note: This function is the same as the base class one,
        # except we call _filter_tools_with_request AFTER _initialize_tools
        span = tracing.get_current_span()
        if span:
            span.set_attribute("session_id", request.session_id)
            span.set_attribute("agent_id", self.agent_id)
            span.set_attribute("request", request.model_dump_json())
            turn_id = str(uuid.uuid4())
            span.set_attribute("turn_id", turn_id)
            if self.agent_config.name:
                span.set_attribute("agent_name", self.agent_config.name)

        await self._initialize_tools(request.toolgroups)
        # after tools initialization filter them by prompt request
        tools_number = len(self.tool_defs) if self.tool_defs else 0
        if (
            self.tools_filter_config.enabled
            and tools_number > 0
            and tools_number > self.tools_filter_config.min_tools
        ):
            await self._filter_tools_with_request(request)
        else:
            logger.info("skip tools filtering, number of tools >>>>> %d", tools_number)

        async for chunk in self._run_turn(request, turn_id):
            yield chunk

    async def _filter_tools_with_request(self, request: AgentTurnCreateRequest) -> None:
        """
        filter self.tool_defs, self.tool_name_to_args to correspond to user prompt
        """
        # get the tools to always include from the configuration
        always_included_tools = set(self.tools_filter_config.always_include_tools)
        # define the list of already called tool names as it may happen that llm will decide to call them again
        # and the new prompt does not contain specific hints to detect/guess them from the current prompt message
        turns = await self.storage.get_session_turns(request.session_id)
        already_called_tool_names = {
            tool_call.tool_name
            for turn in turns
            for step in turn.steps
            if step.step_type == StepType.tool_execution
            for tool_call in step.tool_calls
        }
        # already called tools should be always included
        always_included_tools.update(already_called_tool_names)

        logger.info(
            "always included tool names >>>>>>> %s ",
            always_included_tools,
        )
        message = "\n".join([message.content for message in request.messages])
        tools = [
            dict(tool_name=tool.tool_name, description=tool.description)
            for tool in self.tool_defs
        ]
        tools_filter_model_id = (
            self.tools_filter_config.model_id or self.agent_config.model
        )
        logger.debug(
            "tools filter system prompt: %s", self.tools_filter_config.system_prompt
        )
        response = await self.inference_api.chat_completion(
            tools_filter_model_id,
            [
                UserMessage(content=self.tools_filter_config.system_prompt),
                UserMessage(
                    content="Filter the following tools list, the list is a list of dictionaries "
                    "that contain the tool name and it's corresponding description \n"
                    f"Tools List:\n {tools} \n"
                    f'User Prompt: "{message}" \n'
                    "return a JSON list of strings that correspond to the Relevant Tools, \n"
                    "a strict top 10 items list is needed,\n"
                    "use the tool_name and description for the correct filtering.\n"
                    "return an empty list when no relevant tools found."
                ),
            ],
            stream=False,
            sampling_params=SamplingParams(
                strategy=TopPSamplingStrategy(temperature=0.1), max_tokens="2048"
            ),
        )
        content: str = response.completion_message.content
        logger.debug("response content: >>>>>> %s ", content)
        filtered_tools_names = []
        if "[" in content and "]" in content:
            list_str = content[content.rfind("[") : content.rfind("]") + 1]
            try:
                filtered_tools_names = json.loads(list_str)
                logger.info("the filtered list is >>>>>> %s ", filtered_tools_names)
            except Exception as exp:
                filtered_tools_names = []
                logger.error(exp)

        if filtered_tools_names or always_included_tools:
            original_tools_count = len(self.tool_defs)
            self.tool_defs = list(
                filter(
                    lambda tool: tool.tool_name in filtered_tools_names
                    or tool.tool_name in always_included_tools,
                    self.tool_defs,
                )
            )
            self.tool_name_to_args = {
                key: value
                for key, value in self.tool_name_to_args.items()
                if key in filtered_tools_names or key in always_included_tools
            }
            logger.info(
                "filtered tools count (how much tools was removed):  >>>>>>> %d ",
                original_tools_count - len(self.tool_defs),
            )
            logger.info(
                "new tool names to args keys:  >>>>>>> %s ",
                list(self.tool_name_to_args.keys()),
            )
        else:
            self.tool_defs = []
            self.tool_name_to_args = {}
