import uuid
import httpx
from llama_stack.apis.agents import (
    Agents,
    AgentConfig,
    AgentCreateResponse,
    Document,
    AgentToolGroup,
    Turn,
    AgentTurnResponseStreamChunk,
    AgentTurnResponseEvent,
    AgentTurnResponseTurnCompletePayload,
    AgentTurnResponseEventType,
    AgentStepResponse,
    Session,
    AgentSessionCreateResponse,
    Agent,
)

from collections.abc import AsyncIterator
from datetime import datetime

from llama_stack.apis.common.responses import PaginatedResponse
from llama_stack.apis.inference import (
    CompletionMessage,
    ToolConfig,
    ToolResponse,
    ToolResponseMessage,
    UserMessage,
)

from llama_stack.models.llama.datatypes import StopReason
from llama_stack.schema_utils import webmethod

from llama_stack.apis.agents.openai_responses import (
    OpenAIResponseInput,
    OpenAIResponseInputTool,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
)

from .config import LightspeedAgentConfig


class LightspeedRemoteAgentProvider(Agents):
    def __init__(self, config: LightspeedAgentConfig):
        self.config = config
        self.lightspeed_agent_url = config.api_url

    async def initialize(self) -> None:
        pass

    @webmethod(route="/agents", method="POST", descriptive_name="create_agent")
    async def create_agent(
        self,
        agent_config: AgentConfig,
    ) -> AgentCreateResponse:
        """Create an agent with the given configuration.

        :param agent_config: The configuration for the agent.
        :returns: An AgentCreateResponse with the agent ID.
        """
        ...
        agent_id = str(uuid.uuid4())
        return AgentCreateResponse(agent_id=agent_id)

    @webmethod(
        route="/agents/{agent_id}/session/{session_id}/turn",
        method="POST",
        descriptive_name="create_agent_turn",
    )
    async def create_agent_turn(
        self,
        agent_id: str,
        session_id: str,
        messages: list[UserMessage | ToolResponseMessage],
        stream: bool | None = False,
        documents: list[Document] | None = None,
        toolgroups: list[AgentToolGroup] | None = None,
        tool_config: ToolConfig | None = None,
    ) -> Turn | AsyncIterator[AgentTurnResponseStreamChunk]:
        """Create a new turn for an agent.

        :param agent_id: The ID of the agent to create the turn for.
        :param session_id: The ID of the session to create the turn for.
        :param messages: List of messages to start the turn with.
        :param stream: (Optional) If True, generate an SSE event stream of the response. Defaults to False.
        :param documents: (Optional) List of documents to create the turn with.
        :param toolgroups: (Optional) List of toolgroups to create the turn with, will be used in addition to the agent's config toolgroups for the request.
        :param tool_config: (Optional) The tool configuration to create the turn with, will be used to override the agent's tool_config.
        :returns: If stream=False, returns a Turn object.
                  If stream=True, returns an SSE event stream of AgentTurnResponseStreamChunk
        """
        if not stream:
            raise NotImplementedError("Non-streaming agent turns not yet implemented")
        started_at = datetime.now()

        message = dict(role=messages[0].role, content=messages[0].content)

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", self.lightspeed_agent_url, json=message
            ) as resp:
                await resp.aread()
                resp.raise_for_status()

                async def aaa():
                    async for response in resp.aiter_lines():
                        turn = Turn(
                            turn_id="ssss",
                            session_id=session_id,
                            input_messages=messages,
                            output_message=CompletionMessage(
                                role="assistant",
                                content=response,
                                stop_reason=StopReason.end_of_message,
                            ),
                            steps=[],
                            started_at=started_at,
                        )
                        chunk = AgentTurnResponseStreamChunk(
                            event=AgentTurnResponseEvent(
                                payload=AgentTurnResponseTurnCompletePayload(
                                    event_type=AgentTurnResponseEventType.turn_complete.value,
                                    turn=turn,
                                )
                            ),
                        )
                        yield chunk
                    turn = Turn(
                        turn_id="ssss",
                        session_id=session_id,
                        input_messages=messages,
                        output_message=CompletionMessage(
                            role="assistant",
                            content="",
                            stop_reason=StopReason.end_of_turn,
                        ),
                        steps=[],
                        started_at=started_at,
                    )
                    chunk = AgentTurnResponseStreamChunk(
                        event=AgentTurnResponseEvent(
                            payload=AgentTurnResponseTurnCompletePayload(
                                event_type=AgentTurnResponseEventType.turn_complete.value,
                                turn=turn,
                            )
                        ),
                    )
                    yield chunk

                return aaa()

    @webmethod(
        route="/agents/{agent_id}/session/{session_id}/turn/{turn_id}/resume",
        method="POST",
        descriptive_name="resume_agent_turn",
    )
    async def resume_agent_turn(
        self,
        agent_id: str,
        session_id: str,
        turn_id: str,
        tool_responses: list[ToolResponse],
        stream: bool | None = False,
    ) -> Turn | AsyncIterator[AgentTurnResponseStreamChunk]:
        """Resume an agent turn with executed tool call responses.

        When a Turn has the status `awaiting_input` due to pending input from client side tool calls, this endpoint can be used to submit the outputs from the tool calls once they are ready.

        :param agent_id: The ID of the agent to resume.
        :param session_id: The ID of the session to resume.
        :param turn_id: The ID of the turn to resume.
        :param tool_responses: The tool call responses to resume the turn with.
        :param stream: Whether to stream the response.
        :returns: A Turn object if stream is False, otherwise an AsyncIterator of AgentTurnResponseStreamChunk objects.
        """
        ...

    @webmethod(
        route="/agents/{agent_id}/session/{session_id}/turn/{turn_id}",
        method="GET",
    )
    async def get_agents_turn(
        self,
        agent_id: str,
        session_id: str,
        turn_id: str,
    ) -> Turn:
        """Retrieve an agent turn by its ID.

        :param agent_id: The ID of the agent to get the turn for.
        :param session_id: The ID of the session to get the turn for.
        :param turn_id: The ID of the turn to get.
        :returns: A Turn.
        """
        ...

    @webmethod(
        route="/agents/{agent_id}/session/{session_id}/turn/{turn_id}/step/{step_id}",
        method="GET",
    )
    async def get_agents_step(
        self,
        agent_id: str,
        session_id: str,
        turn_id: str,
        step_id: str,
    ) -> AgentStepResponse:
        """Retrieve an agent step by its ID.

        :param agent_id: The ID of the agent to get the step for.
        :param session_id: The ID of the session to get the step for.
        :param turn_id: The ID of the turn to get the step for.
        :param step_id: The ID of the step to get.
        :returns: An AgentStepResponse.
        """
        ...

    @webmethod(
        route="/agents/{agent_id}/session",
        method="POST",
        descriptive_name="create_agent_session",
    )
    async def create_agent_session(
        self,
        agent_id: str,
        session_name: str,
    ) -> AgentSessionCreateResponse:
        """Create a new session for an agent.

        :param agent_id: The ID of the agent to create the session for.
        :param session_name: The name of the session to create.
        :returns: An AgentSessionCreateResponse.
        """
        ...
        session_id = str(uuid.uuid4())
        return AgentSessionCreateResponse(session_id=session_id)

    @webmethod(route="/agents/{agent_id}/session/{session_id}", method="GET")
    async def get_agents_session(
        self,
        session_id: str,
        agent_id: str,
        turn_ids: list[str] | None = None,
    ) -> Session:
        """Retrieve an agent session by its ID.

        :param session_id: The ID of the session to get.
        :param agent_id: The ID of the agent to get the session for.
        :param turn_ids: (Optional) List of turn IDs to filter the session by.
        """
        ...
        pass

    @webmethod(route="/agents/{agent_id}/session/{session_id}", method="DELETE")
    async def delete_agents_session(
        self,
        session_id: str,
        agent_id: str,
    ) -> None:
        """Delete an agent session by its ID.

        :param session_id: The ID of the session to delete.
        :param agent_id: The ID of the agent to delete the session for.
        """
        ...

    @webmethod(route="/agents/{agent_id}", method="DELETE")
    async def delete_agent(
        self,
        agent_id: str,
    ) -> None:
        """Delete an agent by its ID.

        :param agent_id: The ID of the agent to delete.
        """
        ...

    @webmethod(route="/agents", method="GET")
    async def list_agents(
        self, start_index: int | None = None, limit: int | None = None
    ) -> PaginatedResponse:
        """List all agents.

        :param start_index: The index to start the pagination from.
        :param limit: The number of agents to return.
        :returns: A PaginatedResponse.
        """
        ...

    @webmethod(route="/agents/{agent_id}", method="GET")
    async def get_agent(self, agent_id: str) -> Agent:
        """Describe an agent by its ID.

        :param agent_id: ID of the agent.
        :returns: An Agent of the agent.
        """
        ...

    @webmethod(route="/agents/{agent_id}/sessions", method="GET")
    async def list_agent_sessions(
        self,
        agent_id: str,
        start_index: int | None = None,
        limit: int | None = None,
    ) -> PaginatedResponse:
        """List all session(s) of a given agent.

        :param agent_id: The ID of the agent to list sessions for.
        :returns: A PaginatedResponse.
        """
        ...

    # We situate the OpenAI Responses API in the Agents API just like we did things
    # for Inference. The Responses API, in its intent, serves the same purpose as
    # the Agents API above -- it is essentially a lightweight "agentic loop" with
    # integrated tool calling.
    #
    # Both of these APIs are inherently stateful.

    @webmethod(route="/openai/v1/responses/{id}", method="GET")
    async def get_openai_response(
        self,
        id: str,
    ) -> OpenAIResponseObject:
        """Retrieve an OpenAI response by its ID.

        :param id: The ID of the OpenAI response to retrieve.
        :returns: An OpenAIResponseObject.
        """
        ...

    @webmethod(route="/openai/v1/responses", method="POST")
    async def create_openai_response(
        self,
        input: str | list[OpenAIResponseInput],
        model: str,
        previous_response_id: str | None = None,
        store: bool | None = True,
        stream: bool | None = False,
        temperature: float | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
    ) -> OpenAIResponseObject | AsyncIterator[OpenAIResponseObjectStream]:
        """Create a new OpenAI response.

        :param input: Input message(s) to create the response.
        :param model: The underlying LLM used for completions.
        :param previous_response_id: (Optional) if specified, the new response will be a continuation of the previous response. This can be used to easily fork-off new responses from existing responses.
        """
