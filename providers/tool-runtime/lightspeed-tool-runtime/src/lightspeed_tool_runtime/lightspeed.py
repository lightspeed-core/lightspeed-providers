from typing import Any
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.sse import sse_client

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.tools import (
    ListToolDefsResponse,
    Tool,
    ToolDef,
    ToolInvocationResult,
    ToolParameter,
    ToolRuntime,
)
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.datatypes import ToolGroupsProtocolPrivate

from .config import LightspeedToolConfig


class LightspeedToolRuntimeImp(
    ToolGroupsProtocolPrivate, ToolRuntime, NeedsRequestProviderData
):
    def __init__(self, config: LightspeedToolConfig):
        self.config = config

    async def initialize(self):
        pass

    async def register_tool(self, tool: Tool) -> None:
        pass

    async def unregister_tool(self, tool_id: str) -> None:
        pass

    def _get_auth_headers(self, tool_group_id: str | None = None) -> dict[str, str]:
        headers = {}
        # a global api_key can be passed via configuration
        api_key = self.config.api_key
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        provider_data = self.get_request_provider_data()
        if (
            provider_data is not None
            and provider_data.lightspeed_tool_groups_headers is not None
        ):
            # check for any tool_groups and the current tool_group for provided client headers
            for config_tool_group_id in ["*", tool_group_id]:
                for key, value in provider_data.lightspeed_tool_groups_headers.get(
                    config_tool_group_id, {}
                ).items():
                    if key.lower() == "authorization":
                        # this overrides the global api_key header value if specified
                        headers["Authorization"] = value
                    else:
                        headers[key] = value

        # Note: Do not raise exception when headers is empty as some mcp endpoints may be public
        # and do not support authentication

        return headers

    async def list_runtime_tools(
        self, tool_group_id: str | None = None, mcp_endpoint: URL | None = None
    ) -> ListToolDefsResponse:
        if mcp_endpoint is None:
            raise ValueError("mcp_endpoint is required")

        headers = self._get_auth_headers(tool_group_id)

        tools = []
        async with sse_client(mcp_endpoint.uri, headers=headers) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    parameters = []
                    for param_name, param_schema in tool.inputSchema.get(
                        "properties", {}
                    ).items():
                        parameters.append(
                            ToolParameter(
                                name=param_name,
                                parameter_type=param_schema.get("type", "string"),
                                description=param_schema.get("description", ""),
                            )
                        )
                    tools.append(
                        ToolDef(
                            name=tool.name,
                            description=tool.description,
                            parameters=parameters,
                            metadata={
                                "endpoint": mcp_endpoint.uri,
                            },
                        )
                    )

        return ListToolDefsResponse(data=tools)

    async def invoke_tool(
        self, tool_name: str, kwargs: dict[str, Any]
    ) -> ToolInvocationResult:
        tool = await self.tool_store.get_tool(tool_name)
        if tool.metadata is None or tool.metadata.get("endpoint") is None:
            raise ValueError(f"Tool {tool_name} does not have metadata")
        endpoint = tool.metadata.get("endpoint")
        if urlparse(endpoint).scheme not in ("http", "https"):
            raise ValueError(f"Endpoint {endpoint} is not a valid HTTP(S) URL")

        headers = self._get_auth_headers(tool.toolgroup_id)

        async with sse_client(endpoint, headers=headers) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                result = await session.call_tool(tool.identifier, kwargs)

        return ToolInvocationResult(
            content="\n".join([result.model_dump_json() for result in result.content]),
            error_code=1 if result.isError else 0,
        )
