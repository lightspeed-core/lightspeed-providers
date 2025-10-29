import pytest
from datetime import datetime
from lightspeed_stack_providers.providers.inline.agents.lightspeed_inline_agent.agents import (
    LightspeedAgentsImpl,
)
from lightspeed_stack_providers.providers.inline.agents.lightspeed_inline_agent.config import (
    LightspeedAgentsImplConfig,
)
from llama_stack.providers.inline.agents.meta_reference.persistence import AgentInfo


@pytest.fixture
def mock_persistence_store(mocker):
    """Fixture for mocking the persistence store."""
    return mocker.AsyncMock()


@pytest.fixture
def lightspeed_agents_impl(mock_persistence_store, mocker):
    """Fixture for creating a LightspeedAgentsImpl instance."""
    config = LightspeedAgentsImplConfig()
    impl = LightspeedAgentsImpl(
        config=config,
        inference_api=mocker.AsyncMock(),
        vector_io_api=mocker.AsyncMock(),
        safety_api=mocker.AsyncMock(),
        tool_runtime_api=mocker.AsyncMock(),
        tool_groups_api=mocker.AsyncMock(),
    )
    impl.persistence_store = mock_persistence_store
    return impl


@pytest.mark.asyncio
async def test_get_agent_impl(lightspeed_agents_impl, mock_persistence_store):
    """Test the _get_agent_impl method."""
    agent_id = "test_agent"
    agent_info = AgentInfo(
        agent_name="Test Agent",
        model="test_model",
        instructions="test_instructions",
        created_at=datetime.now(),
        enable_session_persistence=True,
        tool_choice=None,
        tool_prompt_format=None,
    )
    mock_persistence_store.get.return_value = agent_info.model_dump_json()

    agent = await lightspeed_agents_impl._get_agent_impl(agent_id)

    assert agent.agent_id == agent_id
    assert agent.agent_config == agent_info
    mock_persistence_store.get.assert_called_once_with(key=f"agent:{agent_id}")


@pytest.mark.asyncio
async def test_get_agent_impl_not_found(lightspeed_agents_impl, mock_persistence_store):
    """Test the _get_agent_impl method when the agent is not found."""
    agent_id = "test_agent"
    mock_persistence_store.get.return_value = None

    with pytest.raises(ValueError, match=f"Could not find agent info for {agent_id}"):
        await lightspeed_agents_impl._get_agent_impl(agent_id)


@pytest.mark.asyncio
async def test_get_agent_impl_invalid_json(
    lightspeed_agents_impl, mock_persistence_store
):
    """Test the _get_agent_impl method with invalid JSON."""
    agent_id = "test_agent"
    mock_persistence_store.get.return_value = "invalid json"

    with pytest.raises(
        ValueError, match=f"Could not validate agent info for {agent_id}"
    ):
        await lightspeed_agents_impl._get_agent_impl(agent_id)
