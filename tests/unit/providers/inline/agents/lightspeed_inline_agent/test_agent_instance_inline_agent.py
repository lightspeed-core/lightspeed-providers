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

    # Should return MCP tools with allowed_tools set to filtered names
    assert len(filtered_tools) == 2
    for tool in filtered_tools:
        assert tool["type"] == "mcp"
        assert "tool1" in tool["allowed_tools"]


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

    # Should return MCP tools with allowed_tools containing both LLM-filtered and always-included
    assert len(filtered_tools) == 2
    for tool in filtered_tools:
        assert tool["type"] == "mcp"
        assert "tool1" in tool["allowed_tools"]
        assert "tool2" in tool["allowed_tools"]


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


@pytest.mark.asyncio
async def test_get_previously_called_tools_extracts_function_calls(
    lightspeed_agents_impl, mock_conversations_api
):
    """Test that _get_previously_called_tools extracts tool names from function_call items."""
    # Mock conversation items with function_call type
    mock_item1 = MagicMock()
    mock_item1.type = "function_call"
    mock_item1.name = "get_weather"

    mock_item2 = MagicMock()
    mock_item2.type = "function_call"
    mock_item2.name = "get_temperature"

    mock_conversations_api.list_items.return_value = [mock_item1, mock_item2]

    # Call the method
    tool_names = await lightspeed_agents_impl._get_previously_called_tools("conv_123")

    # Should extract both tool names
    assert tool_names == {"get_weather", "get_temperature"}


@pytest.mark.asyncio
async def test_get_previously_called_tools_extracts_mcp_calls(
    lightspeed_agents_impl, mock_conversations_api
):
    """Test that _get_previously_called_tools extracts tool names from mcp_call items."""
    # Mock conversation items with mcp_call type
    mock_item1 = MagicMock()
    mock_item1.type = "mcp_call"
    mock_item1.name = "mcp_tool1"

    mock_item2 = MagicMock()
    mock_item2.type = "mcp_call"
    mock_item2.name = "mcp_tool2"

    mock_conversations_api.list_items.return_value = [mock_item1, mock_item2]

    # Call the method
    tool_names = await lightspeed_agents_impl._get_previously_called_tools("conv_123")

    # Should extract both MCP tool names
    assert tool_names == {"mcp_tool1", "mcp_tool2"}


@pytest.mark.asyncio
async def test_get_previously_called_tools_extracts_mcp_approval_requests(
    lightspeed_agents_impl, mock_conversations_api
):
    """Test that _get_previously_called_tools extracts tool names from mcp_approval_request items."""
    # Mock conversation items with mcp_approval_request type
    mock_item = MagicMock()
    mock_item.type = "mcp_approval_request"
    mock_item.name = "sensitive_tool"

    mock_conversations_api.list_items.return_value = [mock_item]

    # Call the method
    tool_names = await lightspeed_agents_impl._get_previously_called_tools("conv_123")

    # Should extract the tool name
    assert tool_names == {"sensitive_tool"}


@pytest.mark.asyncio
async def test_get_previously_called_tools_handles_mixed_types(
    lightspeed_agents_impl, mock_conversations_api
):
    """Test that _get_previously_called_tools extracts tool names from mixed item types."""
    # Mock conversation items with different types
    function_item = MagicMock()
    function_item.type = "function_call"
    function_item.name = "func_tool"

    mcp_item = MagicMock()
    mcp_item.type = "mcp_call"
    mcp_item.name = "mcp_tool"

    approval_item = MagicMock()
    approval_item.type = "mcp_approval_request"
    approval_item.name = "approval_tool"

    # Non-tool item (should be ignored)
    message_item = MagicMock()
    message_item.type = "message"

    mock_conversations_api.list_items.return_value = [
        function_item,
        mcp_item,
        approval_item,
        message_item,
    ]

    # Call the method
    tool_names = await lightspeed_agents_impl._get_previously_called_tools("conv_123")

    # Should extract all tool names, ignoring non-tool items
    assert tool_names == {"func_tool", "mcp_tool", "approval_tool"}


@pytest.mark.asyncio
async def test_filter_tools_preserves_previously_called_tools(
    lightspeed_agents_impl, mock_inference_api, mock_conversations_api
):
    """Test that previously called tools are preserved even when LLM returns empty list."""
    # Setup conversation with previously called MCP tools
    mock_item = MagicMock()
    mock_item.type = "mcp_call"
    mock_item.name = "previously_used_tool"
    mock_conversations_api.list_items.return_value = [mock_item]

    # Setup mock LLM response to return empty list
    mock_inference_api.openai_chat_completion.return_value = create_mock_chat_response(
        "[]"
    )

    # Setup mock tool runtime to return tools
    tool1_mock = MagicMock(description="Previously used tool")
    tool1_mock.name = "previously_used_tool"
    tool2_mock = MagicMock(description="Other tool")
    tool2_mock.name = "other_tool"
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

    # Call the filtering method with conversation
    filtered_tools = await lightspeed_agents_impl._filter_tools_for_response(
        input="test message",
        tools=tools,
        model="test_model",
        conversation="conv_123",
    )

    # Should return MCP tools with allowed_tools containing previously called tool
    assert len(filtered_tools) == 2
    for tool in filtered_tools:
        assert tool["type"] == "mcp"
        assert "previously_used_tool" in tool["allowed_tools"]
