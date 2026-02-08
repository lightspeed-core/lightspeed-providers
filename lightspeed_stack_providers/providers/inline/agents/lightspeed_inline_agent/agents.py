import json

from llama_stack.core.datatypes import AccessRule
from llama_stack.log import get_logger
from llama_stack.providers.inline.agents.meta_reference.agents import (
    MetaReferenceAgentsImpl,
)
from llama_stack_api import (
    Conversations,
    Files,
    Inference,
    Prompts,
    Safety,
    ToolGroups,
    ToolRuntime,
    VectorIO,
)
from llama_stack_api.agents import ResponseGuardrail
from llama_stack_api.inference import (
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAISystemMessageParam,
    OpenAIUserMessageParam,
)
from llama_stack_api.openai_responses import (
    OpenAIResponseInput,
    OpenAIResponseInputTool,
    OpenAIResponseInputToolChoice,
    OpenAIResponseObject,
    OpenAIResponsePrompt,
    OpenAIResponseText,
)

from .config import LightspeedAgentsImplConfig

logger = get_logger(name=__name__, category="agents")


class LightspeedAgentsImpl(MetaReferenceAgentsImpl):
    def __init__(
        self,
        config: LightspeedAgentsImplConfig,
        inference_api: Inference,
        vector_io_api: VectorIO,
        safety_api: Safety | None,
        tool_runtime_api: ToolRuntime,
        tool_groups_api: ToolGroups,
        conversations_api: Conversations,
        prompts_api: Prompts,
        files_api: Files,
        policy: list[AccessRule],
    ):
        super().__init__(
            config,
            inference_api,
            vector_io_api,
            safety_api,
            tool_runtime_api,
            tool_groups_api,
            conversations_api,
            prompts_api,
            files_api,
            policy,
        )
        self.config = config

    async def create_openai_response(
        self,
        input: str | list[OpenAIResponseInput],
        model: str,
        prompt: OpenAIResponsePrompt | None = None,
        instructions: str | None = None,
        parallel_tool_calls: bool | None = True,
        previous_response_id: str | None = None,
        conversation: str | None = None,
        store: bool | None = True,
        stream: bool | None = False,
        temperature: float | None = None,
        text: OpenAIResponseText | None = None,
        tool_choice: OpenAIResponseInputToolChoice | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
        include: list[str] | None = None,
        max_infer_iters: int | None = 10,
        guardrails: list[ResponseGuardrail] | None = None,
        max_tool_calls: int | None = None,
        metadata: dict[str, str] | None = None,
    ) -> OpenAIResponseObject:
        """
        Create an OpenAI response with optional tool filtering.

        This overrides the parent implementation to add tool filtering functionality
        before passing to the base agent implementation.
        """
        # Apply temperature override if configured
        if temperature is None and self.config.chatbot_temperature_override is not None:
            temperature = self.config.chatbot_temperature_override
            logger.info("Temperature override set to %s", temperature)

        # Apply tool filtering if enabled and tools are provided
        filtered_tools = tools
        if (
            tools
            and self.config.tools_filter.enabled
            and len(tools) > self.config.tools_filter.min_tools
        ):
            logger.info(
                "Tool filtering enabled - filtering %d tools (threshold: %d)",
                len(tools),
                self.config.tools_filter.min_tools,
            )
            filtered_tools = await self._filter_tools_for_response(
                input=input,
                tools=tools,
                model=model,
                conversation=conversation,
            )
            logger.info(
                "Tool filtering complete - reduced from %d to %d tools",
                len(tools),
                len(filtered_tools) if filtered_tools else 0,
            )
        else:
            logger.info(
                "Skipping tool filtering - %d tools (threshold: %d, enabled: %s)",
                len(tools) if tools else 0,
                self.config.tools_filter.min_tools,
                self.config.tools_filter.enabled,
            )

        # Call parent with filtered tools and temperature
        return await super().create_openai_response(
            input=input,
            model=model,
            prompt=prompt,
            instructions=instructions,
            parallel_tool_calls=parallel_tool_calls,
            previous_response_id=previous_response_id,
            conversation=conversation,
            store=store,
            stream=stream,
            temperature=temperature,
            text=text,
            tool_choice=tool_choice,
            tools=filtered_tools,
            include=include,
            max_infer_iters=max_infer_iters,
            guardrails=guardrails,
            max_tool_calls=max_tool_calls,
            metadata=metadata,
        )

    async def _filter_tools_for_response(
        self,
        input: str | list[OpenAIResponseInput],
        tools: list[OpenAIResponseInputTool],
        model: str,
        conversation: str | None,
    ) -> list[OpenAIResponseInputTool]:
        """
        Filter tools using LLM based on user input.

        Args:
            input: User input (string or list of messages)
            tools: List of tool configurations from Responses API
            model: Model ID for inference
            conversation: Conversation ID for retrieving history

        Returns:
            Filtered list of tools
        """
        always_included_tools = set(self.config.tools_filter.always_include_tools)

        # Previously called tools from conversation history
        if conversation:
            try:
                previously_called_tools = await self._get_previously_called_tools(
                    conversation
                )
                always_included_tools.update(previously_called_tools)
                logger.info(
                    "Always included tools (config + previously called): %s",
                    always_included_tools,
                )
            except Exception as e:
                logger.warning("Failed to retrieve conversation history: %s", e)

        tools_for_filtering = await self._extract_tool_definitions(tools)

        if not tools_for_filtering:
            logger.warning("No tool definitions found for filtering")
            return tools

        # Extract user prompt text from input
        if isinstance(input, str):
            user_prompt = input
        elif isinstance(input, list):
            user_prompt = "\n".join(
                [
                    msg.get("content", "") if isinstance(msg, dict) else str(msg)
                    for msg in input
                ]
            )
        else:
            user_prompt = str(input)

        # Call LLM to filter tools
        tools_filter_model_id = self.config.tools_filter.model_id or model
        logger.debug("Using model %s for tool filtering", tools_filter_model_id)
        logger.debug("System prompt: %s", self.config.tools_filter.system_prompt)

        filter_prompt = (
            "Filter the following tools list, the list is a list of dictionaries "
            "that contain the tool name and it's corresponding description \n"
            f"Tools List:\n {tools_for_filtering} \n"
            f'User Prompt: "{user_prompt}" \n'
            "return a JSON list of strings that correspond to the Relevant Tools, \n"
            "a strict top 10 items list is needed,\n"
            "use the tool_name and description for the correct filtering.\n"
            "return an empty list when no relevant tools found."
        )

        request = OpenAIChatCompletionRequestWithExtraBody(
            model=tools_filter_model_id,
            messages=[
                OpenAISystemMessageParam(
                    role="system", content=self.config.tools_filter.system_prompt
                ),
                OpenAIUserMessageParam(role="user", content=filter_prompt),
            ],
            stream=False,
            temperature=0.1,
            max_tokens=2048,
        )
        response = await self.inference_api.openai_chat_completion(request)

        # Parse filtered tool names from LLM response
        content: str = response.choices[0].message.content
        logger.debug("LLM filter response: %s", content)

        filtered_tool_names = []
        if "[" in content and "]" in content:
            list_str = content[content.rfind("[") : content.rfind("]") + 1]
            try:
                filtered_tool_names = json.loads(list_str)
                logger.info("Filtered tool names from LLM: %s", filtered_tool_names)
            except Exception as exp:
                logger.error("Failed to parse LLM response as JSON: %s", exp)
                filtered_tool_names = []

        # Filter the original tools list
        if filtered_tool_names or always_included_tools:
            # Create a mapping from tool names to tool configs
            tool_name_to_config = {}
            for i, tool in enumerate(tools):
                tool_dict = tool if isinstance(tool, dict) else tool.model_dump()
                tool_name = self._get_tool_name_from_config(tool_dict, i)
                tool_name_to_config[tool_name] = tool

            # Filter based on LLM response and always included tools
            filtered_tools = [
                tool_name_to_config[name]
                for name in tool_name_to_config
                if name in filtered_tool_names or name in always_included_tools
            ]

            logger.info(
                "Filtered tools count: %d removed, %d remaining",
                len(tools) - len(filtered_tools),
                len(filtered_tools),
            )
            return filtered_tools
        else:
            logger.warning("No tools matched filtering criteria, returning empty list")
            return []

    async def _get_previously_called_tools(self, conversation_id: str) -> set[str]:
        """
        Extract tool names that were called in previous conversation turns.

        Args:
            conversation_id: The conversation ID

        Returns:
            Set of tool names that were previously called
        """
        tool_names: set[str] = set()
        try:
            items_response = await self.conversations_api.list_items(conversation_id)
            items = (
                items_response.data
                if hasattr(items_response, "data")
                else items_response
            )
            for item in items:
                item_type = getattr(item, "type", None)
                if item_type == "function_call":
                    if hasattr(item, "name") and item.name:
                        tool_names.add(item.name)
                # Also check for nested tool_calls (legacy format)
                elif hasattr(item, "tool_calls") and item.tool_calls:
                    for tool_call in item.tool_calls:
                        if hasattr(tool_call, "name"):
                            tool_names.add(tool_call.name)
            logger.info("Previously called tools: %s", tool_names)
        except Exception as e:
            logger.warning("Failed to extract previously called tools: %s", e)
        return tool_names

    async def _extract_tool_definitions(
        self, tools: list[OpenAIResponseInputTool]
    ) -> list[dict[str, str]]:
        """
        Extract tool names and descriptions from tool configurations.

        For MCP tools, we call the MCP server to list available tools.
        For other tool types, we extract what we can from the config.

        Args:
            tools: List of tool configurations

        Returns:
            List of dicts with tool_name and description
        """
        tool_defs = []

        for i, tool in enumerate(tools):
            tool_dict = tool if isinstance(tool, dict) else tool.model_dump()
            tool_type = tool_dict.get("type")

            if tool_type == "mcp":
                mcp_tools = await self._get_mcp_tool_definitions(tool_dict)
                tool_defs.extend(mcp_tools)
            elif tool_type == "file_search":
                tool_defs.append(
                    {
                        "tool_name": "file_search",
                        "description": "Search through uploaded files and knowledge base",
                    }
                )
            elif tool_type == "function":
                name = tool_dict.get("name", f"function_{i}")
                description = tool_dict.get("description", "")
                tool_defs.append({"tool_name": name, "description": description})
            else:
                logger.warning("Unknown tool type: %s", tool_type)
                tool_defs.append(
                    {
                        "tool_name": f"{tool_type}_{i}",
                        "description": f"Tool of type {tool_type}",
                    }
                )

        return tool_defs

    async def _get_mcp_tool_definitions(
        self, mcp_tool_config: dict
    ) -> list[dict[str, str]]:
        """
        Get tool definitions from an MCP server.

        Args:
            mcp_tool_config: MCP tool configuration dict

        Returns:
            List of tool definitions with tool_name and description
        """
        tool_defs = []

        try:
            server_url = mcp_tool_config.get("server_url")
            server_label = mcp_tool_config.get("server_label", "unknown")

            if not server_url:
                logger.warning("MCP tool config missing server_url")
                return tool_defs

            from llama_stack_api.common.content_types import URL

            mcp_endpoint = URL(uri=server_url)
            tools_response = await self.tool_runtime_api.list_runtime_tools(
                mcp_endpoint=mcp_endpoint
            )

            for tool_def in tools_response.data:
                tool_defs.append(
                    {
                        "tool_name": tool_def.name,
                        "description": tool_def.description or "",
                    }
                )

            logger.debug(
                "Retrieved %d tools from MCP server %s",
                len(tool_defs),
                server_label,
            )
        except Exception as e:
            logger.error("Failed to get MCP tool definitions: %s", e)
            tool_defs.append(
                {
                    "tool_name": mcp_tool_config.get("server_label", "mcp_tool"),
                    "description": "MCP tool server",
                }
            )

        return tool_defs

    def _get_tool_name_from_config(self, tool_dict: dict, index: int) -> str:
        """
        Extract a consistent tool name from a tool configuration.

        Args:
            tool_dict: Tool configuration dict
            index: Index in the tools list (for fallback naming)

        Returns:
            Tool name string
        """
        tool_type = tool_dict.get("type", "unknown")

        if tool_type == "mcp":
            return tool_dict.get("server_label", f"mcp_{index}")
        elif tool_type == "file_search":
            return "file_search"
        elif tool_type == "function":
            return tool_dict.get("name", f"function_{index}")
        else:
            return f"{tool_type}_{index}"
