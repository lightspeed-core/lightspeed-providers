import pytest

from lightspeed_stack_providers.providers.remote.tool_runtime.lightspeed.lightspeed import (
    LightspeedToolRuntimeImp,
)
from lightspeed_stack_providers.providers.remote.tool_runtime.lightspeed.config import (
    LightspeedToolConfig,
)
from llama_stack.apis.common.content_types import URL
from llama_stack.apis.tools import Tool


@pytest.fixture
def lightspeed_tool_runtime_imp(mocker):
    """Fixture for creating a LightspeedToolRuntimeImp instance."""
    config = LightspeedToolConfig()
    imp = LightspeedToolRuntimeImp(config)
    mocker.patch.object(imp, "get_request_provider_data", return_value=None)
    return imp


@pytest.mark.asyncio
async def test_list_runtime_tools(lightspeed_tool_runtime_imp, mocker):
    """Test the list_runtime_tools method."""
    mock_sse_client = mocker.patch(
        "lightspeed_stack_providers.providers.remote.tool_runtime.lightspeed.lightspeed.sse_client"
    )
    mock_session = mocker.AsyncMock()
    mock_tool = mocker.MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.inputSchema = {"properties": {"param1": {"type": "string"}}}
    mock_session.list_tools.return_value.tools = [mock_tool]
    mock_sse_client.return_value.__aenter__.return_value = (
        mocker.AsyncMock(),
        mocker.AsyncMock(),
    )
    mocker.patch(
        "lightspeed_stack_providers.providers.remote.tool_runtime.lightspeed.lightspeed.ClientSession",
    ).return_value.__aenter__.return_value = mock_session

    response = await lightspeed_tool_runtime_imp.list_runtime_tools(
        mcp_endpoint=URL(uri="http://test.com")
    )

    assert len(response.data) == 1
    assert response.data[0].name == "test_tool"


@pytest.mark.asyncio
async def test_invoke_tool(lightspeed_tool_runtime_imp, mocker):
    """Test the invoke_tool method."""
    mock_sse_client = mocker.patch(
        "lightspeed_stack_providers.providers.remote.tool_runtime.lightspeed.lightspeed.sse_client"
    )
    mock_session = mocker.AsyncMock()
    mock_session.call_tool.return_value.isError = False
    mock_result = mocker.MagicMock()
    mock_result.model_dump_json.return_value = '{"key": "value"}'
    mock_session.call_tool.return_value.content = [mock_result]
    mock_sse_client.return_value.__aenter__.return_value = (
        mocker.AsyncMock(),
        mocker.AsyncMock(),
    )
    mocker.patch(
        "lightspeed_stack_providers.providers.remote.tool_runtime.lightspeed.lightspeed.ClientSession",
    ).return_value.__aenter__.return_value = mock_session
    lightspeed_tool_runtime_imp.tool_store = mocker.AsyncMock()
    lightspeed_tool_runtime_imp.tool_store.get_tool.return_value = Tool(
        identifier="test_tool",
        provider_id="test_provider",
        name="test_tool",
        toolgroup_id="test_group",
        description="A test tool",
        parameters=[],
        metadata={"endpoint": "http://test.com"},
    )

    result = await lightspeed_tool_runtime_imp.invoke_tool(
        "test_tool", {"param1": "value1"}
    )

    assert result.error_code == 0
