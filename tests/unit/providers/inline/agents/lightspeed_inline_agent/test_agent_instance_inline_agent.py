import pytest
from lightspeed_stack_providers.providers.inline.agents.lightspeed_inline_agent.agent_instance import (
    LightspeedChatAgent,
)
from lightspeed_stack_providers.providers.inline.agents.lightspeed_inline_agent.config import (
    ToolsFilter,
)
from llama_stack.apis.agents import AgentConfig, AgentTurnCreateRequest, UserMessage
from llama_stack.apis.inference import ChatCompletionResponse, CompletionMessage
from llama_stack.models.llama.datatypes import ToolDefinition

@pytest.fixture
def mock_inference_api(mocker):
    """Fixture for mocking the Inference API."""
    return mocker.AsyncMock()


@pytest.fixture
def mock_storage(mocker):
    """Fixture for mocking the storage."""
    return mocker.AsyncMock()


@pytest.fixture
def lightspeed_chat_agent(mock_inference_api, mock_storage, mocker):
    """Fixture for creating a LightspeedChatAgent instance."""
    agent_config = AgentConfig(
        model="test_model",
        instructions="test_instructions",
        tool_choice=None,
        tool_prompt_format=None,
    )
    tools_filter_config = ToolsFilter(enabled=True, min_tools=0)
    agent = LightspeedChatAgent(
        agent_id="test_agent",
        agent_config=agent_config,
        inference_api=mock_inference_api,
        safety_api=mocker.AsyncMock(),
        tool_runtime_api=mocker.AsyncMock(),
        tool_groups_api=mocker.AsyncMock(),
        vector_io_api=mocker.AsyncMock(),
        persistence_store=mocker.AsyncMock(),
        created_at="2025-10-27T00:00:00Z",
        tools_filter_config=tools_filter_config,
    )
    agent.storage = mock_storage
    agent._initialize_tools = mocker.AsyncMock()
    agent._run_turn = mocker.MagicMock()
    return agent


@pytest.mark.asyncio
async def test_create_and_execute_turn_with_filtering(lightspeed_chat_agent, mocker):
    """Test that create_and_execute_turn calls _filter_tools_with_request when enabled."""
    lightspeed_chat_agent.tool_defs = [
        ToolDefinition(tool_name="test_tool", description="A test tool")
    ]
    request = AgentTurnCreateRequest(
        agent_id="test_agent",
        session_id="test_session",
        messages=[UserMessage(content="test message")],
    )
    lightspeed_chat_agent._filter_tools_with_request = mocker.AsyncMock()

    async for _ in lightspeed_chat_agent.create_and_execute_turn(request):
        pass

    lightspeed_chat_agent._filter_tools_with_request.assert_called_once_with(request)


@pytest.mark.asyncio
async def test_create_and_execute_turn_without_filtering(lightspeed_chat_agent, mocker):
    """Test that create_and_execute_turn does not call _filter_tools_with_request when disabled."""
    lightspeed_chat_agent.tools_filter_config.enabled = False
    lightspeed_chat_agent.tool_defs = [
        ToolDefinition(tool_name="test_tool", description="A test tool")
    ]
    request = AgentTurnCreateRequest(
        agent_id="test_agent",
        session_id="test_session",
        messages=[UserMessage(content="test message")],
    )
    lightspeed_chat_agent._filter_tools_with_request = mocker.AsyncMock()

    async for _ in lightspeed_chat_agent.create_and_execute_turn(request):
        pass

    lightspeed_chat_agent._filter_tools_with_request.assert_not_called()


@pytest.mark.asyncio
async def test_filter_tools_with_request(
    lightspeed_chat_agent, mock_inference_api, mock_storage
):
    """Test the _filter_tools_with_request method."""
    lightspeed_chat_agent.tool_defs = [
        ToolDefinition(tool_name="tool1", description="Tool 1"),
        ToolDefinition(tool_name="tool2", description="Tool 2"),
    ]
    lightspeed_chat_agent.tool_name_to_args = {"tool1": {}, "tool2": {}}
    mock_storage.get_session_turns.return_value = []
    mock_inference_api.chat_completion.return_value = ChatCompletionResponse(
        completion_message=CompletionMessage(
            role="assistant", content='["tool1"]', stop_reason="end_of_turn"
        )
    )
    request = AgentTurnCreateRequest(
        agent_id="test_agent",
        session_id="test_session",
        messages=[UserMessage(content="test message")],
    )

    await lightspeed_chat_agent._filter_tools_with_request(request)

    assert len(lightspeed_chat_agent.tool_defs) == 1
    assert lightspeed_chat_agent.tool_defs[0].tool_name == "tool1"
    assert "tool1" in lightspeed_chat_agent.tool_name_to_args
    assert "tool2" not in lightspeed_chat_agent.tool_name_to_args
