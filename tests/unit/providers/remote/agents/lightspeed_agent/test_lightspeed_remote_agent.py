import pytest
from contextlib import asynccontextmanager

from lightspeed_stack_providers.providers.remote.agents.lightspeed_agent.lightspeed import (
    LightspeedRemoteAgentProvider,
)
from lightspeed_stack_providers.providers.remote.agents.lightspeed_agent.config import (
    LightspeedAgentConfig,
)
from llama_stack.apis.agents import AgentTurnCreateRequest, UserMessage


@pytest.fixture
def lightspeed_remote_agent_provider():
    """Fixture for creating a LightspeedRemoteAgentProvider instance."""
    config = LightspeedAgentConfig(api_url="http://test.com/agent")
    return LightspeedRemoteAgentProvider(config)


@pytest.mark.asyncio
async def test_create_agent_turn(lightspeed_remote_agent_provider, mocker):
    """Test the create_agent_turn method."""

    @asynccontextmanager
    async def mock_stream(*args, **kwargs):
        mock_response = mocker.AsyncMock()
        mock_response.aread = mocker.AsyncMock()
        mock_response.raise_for_status = mocker.MagicMock()

        async def async_generator():
            for item in ["response part 1", "response part 2"]:
                yield item

        mock_response.aiter_lines = async_generator
        yield mock_response

    mock_client = mocker.patch("httpx.AsyncClient")
    mock_client.return_value.__aenter__.return_value.stream = mock_stream

    request = AgentTurnCreateRequest(
        agent_id="test_agent",
        session_id="test_session",
        messages=[UserMessage(content="test message")],
        stream=True,
    )

    response_iterator = await lightspeed_remote_agent_provider.create_agent_turn(
        agent_id=request.agent_id,
        session_id=request.session_id,
        messages=request.messages,
        stream=request.stream,
    )

    responses = [response async for response in response_iterator]

    assert len(responses) == 3
    assert responses[0].event.payload.turn.output_message.content == "response part 1"
    assert responses[1].event.payload.turn.output_message.content == "response part 2"
    assert responses[2].event.payload.turn.output_message.content == ""
