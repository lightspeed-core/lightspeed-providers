"""Unit tests for Lightspeed inline agent provider implementation."""

from unittest.mock import AsyncMock

import pytest
from llama_stack.core.storage.datatypes import KVStoreReference, ResponsesStoreReference
from llama_stack.providers.inline.agents.meta_reference.config import (
    AgentPersistenceConfig,
)
from pytest_mock import AsyncMockType, MockerFixture

from lightspeed_stack_providers.providers.inline.agents.lightspeed_inline_agent.agents import (
    LightspeedAgentsImpl,
)
from lightspeed_stack_providers.providers.inline.agents.lightspeed_inline_agent.config import (
    LightspeedAgentsImplConfig,
    ToolsFilter,
)


@pytest.fixture(name="mock_inference_api")
def mock_inference_api_fixture() -> AsyncMockType:
    """Fixture for mocking the Inference API."""
    return AsyncMock()


@pytest.fixture(name="mock_conversations_api")
def mock_conversations_api_fixture() -> AsyncMockType:
    """Fixture for mocking the Conversations API."""
    mock = AsyncMock()
    mock.list_messages.return_value = []
    return mock


@pytest.fixture(name="lightspeed_agents_impl")
def lightspeed_agents_impl_fixture(
    mock_inference_api: AsyncMockType,
    mock_conversations_api: AsyncMockType,
    mocker: MockerFixture,
) -> LightspeedAgentsImpl:
    """Fixture for creating a LightspeedAgentsImpl instance."""
    persistence = AgentPersistenceConfig(
        agent_state=KVStoreReference(namespace="test", backend="in_memory"),
        responses=ResponsesStoreReference(
            table_name="test_responses", backend="in_memory"
        ),
    )
    config = LightspeedAgentsImplConfig(
        persistence=persistence, tools_filter=ToolsFilter(enabled=True, min_tools=5)
    )
    return LightspeedAgentsImpl(
        config=config,
        inference_api=mock_inference_api,
        vector_io_api=mocker.AsyncMock(),
        safety_api=mocker.AsyncMock(),
        tool_runtime_api=mocker.AsyncMock(),
        tool_groups_api=mocker.AsyncMock(),
        conversations_api=mock_conversations_api,
        prompts_api=mocker.AsyncMock(),
        files_api=mocker.AsyncMock(),
        connectors_api=mocker.AsyncMock(),
        policy=[],
    )


def test_lightspeed_agents_impl_initialization(
    lightspeed_agents_impl: LightspeedAgentsImpl,
) -> None:
    """Test that LightspeedAgentsImpl initializes correctly."""
    assert lightspeed_agents_impl.config is not None
    assert lightspeed_agents_impl.config.tools_filter is not None
    assert lightspeed_agents_impl.config.tools_filter.enabled is True
    assert lightspeed_agents_impl.config.tools_filter.min_tools == 5


def test_lightspeed_agents_impl_config_defaults() -> None:
    """Test that LightspeedAgentsImplConfig has correct defaults."""
    persistence = AgentPersistenceConfig(
        agent_state=KVStoreReference(namespace="test", backend="in_memory"),
        responses=ResponsesStoreReference(
            table_name="test_responses", backend="in_memory"
        ),
    )
    config = LightspeedAgentsImplConfig(persistence=persistence)
    assert config.tools_filter is not None
    assert config.tools_filter.enabled is True
    assert config.tools_filter.min_tools == 10
    assert config.chatbot_temperature_override is None


def test_tools_filter_config() -> None:
    """Test ToolsFilter configuration."""
    filter_config = ToolsFilter(
        enabled=True,
        min_tools=5,
        always_include_tools=["tool1", "tool2"],
    )
    assert filter_config.enabled is True
    assert filter_config.min_tools == 5
    assert filter_config.always_include_tools is not None
    assert "tool1" in filter_config.always_include_tools
    assert "tool2" in filter_config.always_include_tools


@pytest.mark.asyncio
async def test_get_tool_name_from_config_mcp(
    lightspeed_agents_impl: LightspeedAgentsImpl,
) -> None:
    """Test _get_tool_name_from_config for MCP tools."""
    tool_dict = {"type": "mcp", "server_label": "my_server"}
    # pylint: disable=protected-access
    name = lightspeed_agents_impl._get_tool_name_from_config(tool_dict, 0)
    assert name == "my_server"


@pytest.mark.asyncio
async def test_get_tool_name_from_config_file_search(
    lightspeed_agents_impl: LightspeedAgentsImpl,
) -> None:
    """Test _get_tool_name_from_config for file_search tools."""
    tool_dict = {"type": "file_search", "vector_store_ids": ["vs_123"]}
    # pylint: disable=protected-access
    name = lightspeed_agents_impl._get_tool_name_from_config(tool_dict, 0)
    assert name == "file_search"


@pytest.mark.asyncio
async def test_get_tool_name_from_config_function(
    lightspeed_agents_impl: LightspeedAgentsImpl,
) -> None:
    """Test _get_tool_name_from_config for function tools."""
    tool_dict = {"type": "function", "name": "my_function"}
    # pylint: disable=protected-access
    name = lightspeed_agents_impl._get_tool_name_from_config(tool_dict, 0)
    assert name == "my_function"


@pytest.mark.asyncio
async def test_get_tool_name_from_config_unknown(
    lightspeed_agents_impl: LightspeedAgentsImpl,
) -> None:
    """Test _get_tool_name_from_config for unknown tool types."""
    tool_dict = {"type": "custom_tool"}
    # pylint: disable=protected-access
    name = lightspeed_agents_impl._get_tool_name_from_config(tool_dict, 3)
    assert name == "custom_tool_3"


@pytest.mark.asyncio
async def test_extract_tool_definitions_file_search(
    lightspeed_agents_impl: LightspeedAgentsImpl,
) -> None:
    """Test _extract_tool_definitions for file_search tools."""
    tools = [{"type": "file_search", "vector_store_ids": ["vs_123"]}]
    # pylint: disable=protected-access
    defs, _ = await lightspeed_agents_impl._extract_tool_definitions(tools)
    assert len(defs) == 1
    assert defs[0]["tool_name"] == "file_search"
    assert "knowledge base" in defs[0]["description"].lower()


@pytest.mark.asyncio
async def test_extract_tool_definitions_function(
    lightspeed_agents_impl: LightspeedAgentsImpl,
) -> None:
    """Test _extract_tool_definitions for function tools."""
    tools = [
        {"type": "function", "name": "get_weather", "description": "Get the weather"}
    ]
    # pylint: disable=protected-access
    defs, _ = await lightspeed_agents_impl._extract_tool_definitions(tools)
    assert len(defs) == 1
    assert defs[0]["tool_name"] == "get_weather"
    assert defs[0]["description"] == "Get the weather"
