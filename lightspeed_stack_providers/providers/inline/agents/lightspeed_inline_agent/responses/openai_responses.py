"""
Workaround for MCP server connection cleanup issue.

This file provides a custom implementation to handle MCP session cleanup properly.
The issue is resolved in Llama Stack by https://github.com/llamastack/llama-stack/pull/4758,
but we need to use Llama Stack 0.4.3 and have backported the PR fix to this code.

TODO: Remove this workaround once we upgrade to a Llama Stack version that includes PR #4758.
"""

import time
import uuid
from typing import AsyncIterator

from llama_stack.log import get_logger
from llama_stack.providers.inline.agents.meta_reference.responses.openai_responses import (
    OpenAIResponsesImpl,
)
from llama_stack.providers.inline.agents.meta_reference.responses.streaming import (
    StreamingResponseOrchestrator,
)
from llama_stack.providers.inline.agents.meta_reference.responses.tool_executor import (
    ToolExecutor,
)
from llama_stack.providers.inline.agents.meta_reference.responses.types import (
    ChatCompletionContext,
)
from llama_stack.providers.inline.agents.meta_reference.responses.utils import (
    convert_response_text_to_chat_response_format,
)
from llama_stack.providers.utils.tools.mcp import MCPSessionManager
from llama_stack_api import (
    ConversationItem,
    OpenAIResponseInput,
    OpenAIResponseInputTool,
    OpenAIResponseInputToolChoice,
    OpenAIResponseObjectStream,
    OpenAIResponsePrompt,
    OpenAIResponseText,
    OpenAISystemMessageParam,
)
from llama_stack_api.agents import ResponseItemInclude

logger = get_logger(name=__name__, category="agents")


class MyMCPSessionManager(MCPSessionManager):
    async def __aenter__(self):
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager and cleanup all sessions."""
        await super().close_all()
        return False


class LightspeedOpenAIResponsesImpl(OpenAIResponsesImpl):
    async def _create_streaming_response(
        self,
        input: str | list[OpenAIResponseInput],
        model: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        conversation: str | None = None,
        prompt: OpenAIResponsePrompt | None = None,
        store: bool | None = True,
        temperature: float | None = None,
        text: OpenAIResponseText | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
        tool_choice: OpenAIResponseInputToolChoice | None = None,
        max_infer_iters: int | None = 10,
        guardrail_ids: list[str] | None = None,
        parallel_tool_calls: bool | None = True,
        max_tool_calls: int | None = None,
        metadata: dict[str, str] | None = None,
        include: list[ResponseItemInclude] | None = None,
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        logger.info("LightspeedOpenAIResponsesImpl._create_streaming_response")
        # These should never be None when called from create_openai_response (which sets defaults)
        # but we assert here to help mypy understand the types
        assert text is not None, "text must not be None"
        assert max_infer_iters is not None, "max_infer_iters must not be None"

        # Input preprocessing
        all_input, messages, tool_context = (
            await self._process_input_with_previous_response(
                input, tools, previous_response_id, conversation
            )
        )

        if instructions:
            messages.insert(0, OpenAISystemMessageParam(content=instructions))

        # Prepend reusable prompt (if provided)
        await self._prepend_prompt(messages, prompt)

        # Structured outputs
        response_format = await convert_response_text_to_chat_response_format(text)

        ctx = ChatCompletionContext(
            model=model,
            messages=messages,
            response_tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            response_format=response_format,
            tool_context=tool_context,
            inputs=all_input,
        )

        # Create orchestrator and delegate streaming logic
        response_id = f"resp_{uuid.uuid4()}"
        created_at = int(time.time())

        # Create a per-request MCP session manager for session reuse (fix for #4452)
        # This avoids redundant tools/list calls when making multiple MCP tool invocations
        async with MyMCPSessionManager() as mcp_session_manager:

            # Create a per-request ToolExecutor with the session manager
            request_tool_executor = ToolExecutor(
                tool_groups_api=self.tool_groups_api,
                tool_runtime_api=self.tool_runtime_api,
                vector_io_api=self.vector_io_api,
                vector_stores_config=self.tool_executor.vector_stores_config,
                mcp_session_manager=mcp_session_manager,
            )

            orchestrator = StreamingResponseOrchestrator(
                inference_api=self.inference_api,
                ctx=ctx,
                response_id=response_id,
                created_at=created_at,
                prompt=prompt,
                text=text,
                max_infer_iters=max_infer_iters,
                parallel_tool_calls=parallel_tool_calls,
                tool_executor=request_tool_executor,
                safety_api=self.safety_api,
                guardrail_ids=guardrail_ids,
                instructions=instructions,
                max_tool_calls=max_tool_calls,
                metadata=metadata,
                include=include,
            )

            # Stream the response
            final_response = None
            failed_response = None

            # Type as ConversationItem to avoid list invariance issues
            output_items: list[ConversationItem] = []
            async for stream_chunk in orchestrator.create_response():
                match stream_chunk.type:
                    case "response.completed" | "response.incomplete":
                        final_response = stream_chunk.response
                    case "response.failed":
                        failed_response = stream_chunk.response
                    case "response.output_item.done":
                        item = stream_chunk.item
                        output_items.append(item)
                    case _:
                        pass  # Other event types

                # Store and sync before yielding terminal events
                # This ensures the storage/syncing happens even if the consumer breaks after
                # receiving the event
                if (
                    stream_chunk.type in {"response.completed", "response.incomplete"}
                    and final_response
                    and failed_response is None
                ):
                    messages_to_store = list(
                        filter(
                            lambda x: not isinstance(x, OpenAISystemMessageParam),
                            orchestrator.final_messages,
                        )
                    )
                    if store:
                        # TODO: we really should work off of output_items instead of
                        #  "final_messages"
                        await self._store_response(
                            response=final_response,
                            input=all_input,
                            messages=messages_to_store,
                        )

                    if conversation:
                        await self._sync_response_to_conversation(
                            conversation, input, output_items
                        )
                        await self.responses_store.store_conversation_messages(
                            conversation, messages_to_store
                        )

                yield stream_chunk
