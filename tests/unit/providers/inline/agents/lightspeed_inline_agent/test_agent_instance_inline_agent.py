from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_stack.core.storage.datatypes import KVStoreReference, ResponsesStoreReference
from llama_stack.providers.inline.agents.meta_reference.config import (
    AgentPersistenceConfig,
)

from lightspeed_stack_providers.providers.inline.agents.lightspeed_inline_agent.agents import (
    LightspeedAgentsImpl,
)
from lightspeed_stack_providers.providers.inline.agents.lightspeed_inline_agent.config import (
    LightspeedAgentsImplConfig,
    ToolsFilter,
)


@pytest.fixture
def mock_inference_api():
    """Fixture for mocking the Inference API."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def mock_conversations_api():
    """Fixture for mocking the Conversations API."""
    mock = AsyncMock()
    mock.list_messages.return_value = []
    return mock


@pytest.fixture
def mock_tool_runtime_api():
    """Fixture for mocking the Tool Runtime API."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def lightspeed_agents_impl(
    mock_inference_api, mock_conversations_api, mock_tool_runtime_api, mocker
):
    """Fixture for creating a LightspeedAgentsImpl instance."""
    persistence = AgentPersistenceConfig(
        agent_state=KVStoreReference(namespace="test", backend="in_memory"),
        responses=ResponsesStoreReference(
            table_name="test_responses", backend="in_memory"
        ),
    )
    config = LightspeedAgentsImplConfig(
        persistence=persistence, tools_filter=ToolsFilter(enabled=True, min_tools=0)
    )
    impl = LightspeedAgentsImpl(
        config=config,
        inference_api=mock_inference_api,
        vector_io_api=mocker.AsyncMock(),
        safety_api=mocker.AsyncMock(),
        tool_runtime_api=mock_tool_runtime_api,
        tool_groups_api=mocker.AsyncMock(),
        conversations_api=mock_conversations_api,
        prompts_api=mocker.AsyncMock(),
        files_api=mocker.AsyncMock(),
        policy=[],
    )
    return impl


def create_mock_chat_response(content: str):
    """Create a mock OpenAI chat completion response."""
    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    return mock_response


@pytest.mark.asyncio
async def test_filter_tools_for_response_filters_correctly(
    lightspeed_agents_impl, mock_inference_api
):
    """Test that _filter_tools_for_response filters tools based on LLM response."""
    # Setup mock LLM response to return filtered tool names
    mock_inference_api.openai_chat_completion.return_value = create_mock_chat_response(
        '["tool1"]'
    )

    # Setup mock tool runtime to return different tools per MCP server
    # Note: MagicMock(name=...) uses name for __repr__, so set .name after creation
    tool1_mock = MagicMock(description="Tool 1")
    tool1_mock.name = "tool1"
    tool2_mock = MagicMock(description="Tool 2")
    tool2_mock.name = "tool2"
    lightspeed_agents_impl.tool_runtime_api.list_runtime_tools.side_effect = [
        MagicMock(data=[tool1_mock]),
        MagicMock(data=[tool2_mock]),
    ]

    # Create MCP tools
    tools = [
        {
            "type": "mcp",
            "server_label": "mcp_server1",
            "server_url": "http://test1.com",
        },
        {
            "type": "mcp",
            "server_label": "mcp_server2",
            "server_url": "http://test2.com",
        },
    ]

    # Call the filtering method
    filtered_tools = await lightspeed_agents_impl._filter_tools_for_response(
        input="test message",
        tools=tools,
        model="test_model",
        conversation=None,
    )

    # Should only return tool1 as a function-type tool
    assert len(filtered_tools) == 1
    assert filtered_tools[0]["type"] == "function"
    assert filtered_tools[0]["name"] == "tool1"


@pytest.mark.asyncio
async def test_filter_tools_for_response_includes_always_included_tools(
    lightspeed_agents_impl, mock_inference_api
):
    """Test that always-included tools are preserved even when not in LLM response."""
    # Configure always-included tools (uses actual tool names, not server labels)
    lightspeed_agents_impl.config.tools_filter.always_include_tools = ["tool2"]

    # Setup mock LLM response to return only one tool name
    mock_inference_api.openai_chat_completion.return_value = create_mock_chat_response(
        '["tool1"]'
    )

    # Setup mock tool runtime to return different tools per MCP server
    # Note: MagicMock(name=...) uses name for __repr__, so set .name after creation
    tool1_mock = MagicMock(description="Tool 1")
    tool1_mock.name = "tool1"
    tool2_mock = MagicMock(description="Tool 2")
    tool2_mock.name = "tool2"
    lightspeed_agents_impl.tool_runtime_api.list_runtime_tools.side_effect = [
        MagicMock(data=[tool1_mock]),
        MagicMock(data=[tool2_mock]),
    ]

    # Create MCP tools
    tools = [
        {
            "type": "mcp",
            "server_label": "mcp_server1",
            "server_url": "http://test1.com",
        },
        {
            "type": "mcp",
            "server_label": "mcp_server2",
            "server_url": "http://test2.com",
        },
    ]

    # Call the filtering method
    filtered_tools = await lightspeed_agents_impl._filter_tools_for_response(
        input="test message",
        tools=tools,
        model="test_model",
        conversation=None,
    )

    # Should return both tools - one from LLM, one from always-included
    assert len(filtered_tools) == 2
    tool_names = [t["name"] for t in filtered_tools]
    assert "tool1" in tool_names
    assert "tool2" in tool_names


@pytest.mark.asyncio
async def test_create_openai_response_skips_filtering_when_disabled(
    lightspeed_agents_impl, mocker
):
    """Test that create_openai_response skips filtering when disabled."""
    lightspeed_agents_impl.config.tools_filter.enabled = False

    # Mock the parent's create_openai_response
    mock_parent_response = MagicMock()
    mocker.patch.object(
        LightspeedAgentsImpl.__bases__[0],
        "create_openai_response",
        return_value=mock_parent_response,
    )

    # Mock _filter_tools_for_response to track if it's called
    filter_mock = mocker.patch.object(
        lightspeed_agents_impl,
        "_filter_tools_for_response",
        return_value=[],
    )

    tools = [
        {"type": "mcp", "server_label": "test", "server_url": "http://test.com"},
    ]

    # Call create_openai_response
    await lightspeed_agents_impl.create_openai_response(
        input="test",
        model="test_model",
        tools=tools,
    )

    # Filter should not have been called
    filter_mock.assert_not_called()


@pytest.mark.asyncio
async def test_filter_tools_for_response_skips_filtering_below_threshold(
    lightspeed_agents_impl, mock_inference_api
):
    """Test that _filter_tools_for_response skips filtering when expanded tools are below threshold."""
    lightspeed_agents_impl.config.tools_filter.min_tools = 10  # High threshold

    # Setup mock tool runtime to return few tools (below threshold)
    lightspeed_agents_impl.tool_runtime_api.list_runtime_tools.side_effect = [
        MagicMock(data=[MagicMock(name="tool1", description="Tool 1")]),
        MagicMock(data=[MagicMock(name="tool2", description="Tool 2")]),
    ]

    tools = [
        {"type": "mcp", "server_label": "test1", "server_url": "http://test1.com"},
        {"type": "mcp", "server_label": "test2", "server_url": "http://test2.com"},
    ]  # Only 2 expanded tools, below threshold of 10

    # Call _filter_tools_for_response directly
    filtered_tools = await lightspeed_agents_impl._filter_tools_for_response(
        input="test",
        tools=tools,
        model="test_model",
        conversation=None,
    )

    # Should return original tools unchanged since below threshold
    assert filtered_tools == tools
    # LLM should NOT have been called
    mock_inference_api.openai_chat_completion.assert_not_called()
