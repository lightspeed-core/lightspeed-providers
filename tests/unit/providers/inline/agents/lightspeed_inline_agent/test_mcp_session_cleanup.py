"""
Tests for MCP server connection cleanup workaround.

This module tests the workaround for the MCP server connection cleanup issue
that was backported from https://github.com/llamastack/llama-stack/pull/4758.
"""

from collections.abc import AsyncGenerator

import pytest
from llama_stack.core.storage.datatypes import KVStoreReference, ResponsesStoreReference
from llama_stack.providers.inline.agents.meta_reference.config import (
    AgentPersistenceConfig,
)
from pytest_mock import MockerFixture

from lightspeed_stack_providers.providers.inline.agents.lightspeed_inline_agent.agents import (
    LightspeedAgentsImpl,
)
from lightspeed_stack_providers.providers.inline.agents.lightspeed_inline_agent.config import (
    LightspeedAgentsImplConfig,
)
from lightspeed_stack_providers.providers.inline.agents.lightspeed_inline_agent.responses.openai_responses import (
    LightspeedOpenAIResponsesImpl,
    MyMCPSessionManager,
)


@pytest.fixture
def lightspeed_agents_impl(mocker: MockerFixture) -> LightspeedAgentsImpl:
    """Fixture for creating a LightspeedAgentsImpl instance."""
    persistence = AgentPersistenceConfig(
        agent_state=KVStoreReference(namespace="test", backend="in_memory"),
        responses=ResponsesStoreReference(
            table_name="test_responses", backend="in_memory"
        ),
    )
    config = LightspeedAgentsImplConfig(persistence=persistence)

    return LightspeedAgentsImpl(
        config=config,
        inference_api=mocker.AsyncMock(),
        vector_io_api=mocker.AsyncMock(),
        safety_api=mocker.AsyncMock(),
        tool_runtime_api=mocker.AsyncMock(),
        tool_groups_api=mocker.AsyncMock(),
        conversations_api=mocker.AsyncMock(),
        prompts_api=mocker.AsyncMock(),
        files_api=mocker.AsyncMock(),
        policy=[],
    )


@pytest.mark.asyncio
async def test_initialize_creates_lightspeed_openai_responses_impl(
    lightspeed_agents_impl: LightspeedAgentsImpl, mocker: MockerFixture
) -> None:
    """Test that initialize() creates a LightspeedOpenAIResponsesImpl instance."""
    # Mock the parent initialize method
    mocker.patch.object(
        lightspeed_agents_impl.__class__.__bases__[0], "initialize", return_value=None
    )

    # Mock the responses_store that should be set by parent's initialize
    lightspeed_agents_impl.responses_store = mocker.AsyncMock()

    # Remove openai_responses_impl if it exists (may be set by parent class)
    if hasattr(lightspeed_agents_impl, "openai_responses_impl"):
        delattr(lightspeed_agents_impl, "openai_responses_impl")

    # Before initialization, openai_responses_impl should not exist
    assert not hasattr(lightspeed_agents_impl, "openai_responses_impl")

    # Call initialize
    await lightspeed_agents_impl.initialize()

    # After initialization, openai_responses_impl should be created
    assert hasattr(lightspeed_agents_impl, "openai_responses_impl")
    assert isinstance(
        lightspeed_agents_impl.openai_responses_impl, LightspeedOpenAIResponsesImpl
    )


@pytest.mark.asyncio
async def test_initialize_passes_correct_dependencies(
    lightspeed_agents_impl: LightspeedAgentsImpl, mocker: MockerFixture
) -> None:
    """Test that initialize() passes correct dependencies to LightspeedOpenAIResponsesImpl."""
    # Mock the parent initialize method
    mocker.patch.object(
        lightspeed_agents_impl.__class__.__bases__[0], "initialize", return_value=None
    )

    # Mock the responses_store
    lightspeed_agents_impl.responses_store = mocker.AsyncMock()

    # Call initialize
    await lightspeed_agents_impl.initialize()

    # Verify the created instance has correct dependencies
    openai_impl = lightspeed_agents_impl.openai_responses_impl
    assert openai_impl is not None
    assert openai_impl.inference_api == lightspeed_agents_impl.inference_api
    assert openai_impl.tool_groups_api == lightspeed_agents_impl.tool_groups_api
    assert openai_impl.tool_runtime_api == lightspeed_agents_impl.tool_runtime_api
    assert openai_impl.responses_store == lightspeed_agents_impl.responses_store
    assert openai_impl.vector_io_api == lightspeed_agents_impl.vector_io_api
    assert openai_impl.safety_api == lightspeed_agents_impl.safety_api
    assert openai_impl.conversations_api == lightspeed_agents_impl.conversations_api
    assert openai_impl.prompts_api == lightspeed_agents_impl.prompts_api
    assert openai_impl.files_api == lightspeed_agents_impl.files_api


@pytest.mark.asyncio
async def test_mcp_session_manager_enter(mocker: MockerFixture) -> None:
    """Test MyMCPSessionManager __aenter__ returns self."""
    session_manager = MyMCPSessionManager()

    result = await session_manager.__aenter__()

    assert result is session_manager


@pytest.mark.asyncio
async def test_mcp_session_manager_exit_calls_close_all(mocker: MockerFixture) -> None:
    """Test MyMCPSessionManager __aexit__ calls close_all()."""
    session_manager = MyMCPSessionManager()

    # Mock the close_all method
    close_all_mock = mocker.patch.object(
        session_manager.__class__.__bases__[0], "close_all", return_value=None
    )

    # Call __aexit__
    result = await session_manager.__aexit__(None, None, None)

    # Verify close_all was called
    close_all_mock.assert_called_once()
    # Verify __aexit__ returns False (don't suppress exceptions)
    assert result is False


@pytest.mark.asyncio
async def test_mcp_session_manager_exit_returns_false_on_exception(
    mocker: MockerFixture,
) -> None:
    """Test MyMCPSessionManager __aexit__ returns False when exception occurs."""
    session_manager = MyMCPSessionManager()

    # Mock the close_all method
    mocker.patch.object(
        session_manager.__class__.__bases__[0], "close_all", return_value=None
    )

    # Simulate an exception
    exc_type = ValueError
    exc_val = ValueError("Test exception")
    exc_tb = None

    result = await session_manager.__aexit__(exc_type, exc_val, exc_tb)

    # Should return False to allow exception to propagate
    assert result is False


@pytest.mark.asyncio
async def test_mcp_session_manager_context_manager_usage(mocker: MockerFixture) -> None:
    """Test MyMCPSessionManager can be used as an async context manager."""
    session_manager = MyMCPSessionManager()

    # Mock the close_all method
    close_all_mock = mocker.patch.object(
        session_manager.__class__.__bases__[0], "close_all", return_value=None
    )

    # Use as context manager
    async with session_manager as manager:
        assert manager is session_manager

    # Verify close_all was called when exiting context
    close_all_mock.assert_called_once()


@pytest.mark.asyncio
async def test_lightspeed_openai_responses_impl_initialization(
    mocker: MockerFixture,
) -> None:
    """Test LightspeedOpenAIResponsesImpl can be instantiated."""
    # Create mock dependencies
    inference_api = mocker.AsyncMock()
    tool_groups_api = mocker.AsyncMock()
    tool_runtime_api = mocker.AsyncMock()
    responses_store = mocker.AsyncMock()
    vector_io_api = mocker.AsyncMock()
    safety_api = mocker.AsyncMock()
    conversations_api = mocker.AsyncMock()
    prompts_api = mocker.AsyncMock()
    files_api = mocker.AsyncMock()

    # Create instance
    impl = LightspeedOpenAIResponsesImpl(
        inference_api=inference_api,
        tool_groups_api=tool_groups_api,
        tool_runtime_api=tool_runtime_api,
        responses_store=responses_store,
        vector_io_api=vector_io_api,
        safety_api=safety_api,
        conversations_api=conversations_api,
        prompts_api=prompts_api,
        files_api=files_api,
        vector_stores_config=None,
    )

    # Verify instance is created correctly
    assert impl.inference_api is inference_api
    assert impl.tool_groups_api is tool_groups_api
    assert impl.tool_runtime_api is tool_runtime_api


@pytest.mark.asyncio
async def test_create_streaming_response_uses_mcp_session_manager(
    mocker: MockerFixture,
) -> None:
    """Test _create_streaming_response uses MyMCPSessionManager."""
    # Create mock dependencies
    inference_api = mocker.AsyncMock()
    tool_groups_api = mocker.AsyncMock()
    tool_runtime_api = mocker.AsyncMock()
    responses_store = mocker.AsyncMock()
    vector_io_api = mocker.AsyncMock()
    safety_api = mocker.AsyncMock()
    conversations_api = mocker.AsyncMock()
    prompts_api = mocker.AsyncMock()
    files_api = mocker.AsyncMock()

    impl = LightspeedOpenAIResponsesImpl(
        inference_api=inference_api,
        tool_groups_api=tool_groups_api,
        tool_runtime_api=tool_runtime_api,
        responses_store=responses_store,
        vector_io_api=vector_io_api,
        safety_api=safety_api,
        conversations_api=conversations_api,
        prompts_api=prompts_api,
        files_api=files_api,
        vector_stores_config=None,
    )

    # Mock required internal methods
    mocker.patch.object(
        impl, "_process_input_with_previous_response", return_value=([], [], None)
    )
    mocker.patch.object(impl, "_prepend_prompt", return_value=None)

    # Mock the StreamingResponseOrchestrator to return a simple response
    mock_orchestrator = mocker.MagicMock()

    async def mock_create_response() -> AsyncGenerator:
        """Mock generator that yields a completed response."""
        from llama_stack_api.openai_responses import OpenAIResponseObject

        mock_response = mocker.MagicMock(spec=OpenAIResponseObject)
        yield mocker.MagicMock(type="response.completed", response=mock_response)

    mock_orchestrator.create_response = mock_create_response
    mock_orchestrator.final_messages = []

    # Patch StreamingResponseOrchestrator constructor
    mocker.patch(
        "lightspeed_stack_providers.providers.inline.agents.lightspeed_inline_agent.responses.openai_responses.StreamingResponseOrchestrator",
        return_value=mock_orchestrator,
    )

    # Patch MyMCPSessionManager to track usage
    mcp_manager_instance = MyMCPSessionManager()
    mcp_manager_mock = mocker.patch(
        "lightspeed_stack_providers.providers.inline.agents.lightspeed_inline_agent.responses.openai_responses.MyMCPSessionManager",
        return_value=mcp_manager_instance,
    )

    # Mock close_all on the parent class
    close_all_mock = mocker.patch.object(
        mcp_manager_instance.__class__.__bases__[0], "close_all", return_value=None
    )

    # Call _create_streaming_response
    from llama_stack_api.openai_responses import OpenAIResponseText

    response_generator = impl._create_streaming_response(
        input="test input",
        model="test-model",
        text=OpenAIResponseText(type="text"),
        max_infer_iters=10,
    )

    # Consume the async generator
    responses = []
    async for chunk in response_generator:
        responses.append(chunk)

    # Verify MyMCPSessionManager was instantiated
    mcp_manager_mock.assert_called_once()

    # Verify close_all was called (context manager cleanup)
    close_all_mock.assert_called_once()


@pytest.mark.asyncio
async def test_create_streaming_response_cleanup_on_error(
    mocker: MockerFixture,
) -> None:
    """Test _create_streaming_response properly cleans up MCP session on error."""
    # Create mock dependencies
    inference_api = mocker.AsyncMock()
    tool_groups_api = mocker.AsyncMock()
    tool_runtime_api = mocker.AsyncMock()
    responses_store = mocker.AsyncMock()
    vector_io_api = mocker.AsyncMock()
    safety_api = mocker.AsyncMock()
    conversations_api = mocker.AsyncMock()
    prompts_api = mocker.AsyncMock()
    files_api = mocker.AsyncMock()

    impl = LightspeedOpenAIResponsesImpl(
        inference_api=inference_api,
        tool_groups_api=tool_groups_api,
        tool_runtime_api=tool_runtime_api,
        responses_store=responses_store,
        vector_io_api=vector_io_api,
        safety_api=safety_api,
        conversations_api=conversations_api,
        prompts_api=prompts_api,
        files_api=files_api,
        vector_stores_config=None,
    )

    # Mock required internal methods
    mocker.patch.object(
        impl, "_process_input_with_previous_response", return_value=([], [], None)
    )
    mocker.patch.object(impl, "_prepend_prompt", return_value=None)

    # Mock the StreamingResponseOrchestrator to raise an error
    async def mock_create_response_with_error() -> AsyncGenerator:
        """Mock generator that raises an error."""
        raise ValueError("Test error in streaming")
        yield  # This line is never reached

    mock_orchestrator = mocker.MagicMock()
    mock_orchestrator.create_response = mock_create_response_with_error

    mocker.patch(
        "lightspeed_stack_providers.providers.inline.agents.lightspeed_inline_agent.responses.openai_responses.StreamingResponseOrchestrator",
        return_value=mock_orchestrator,
    )

    # Patch MyMCPSessionManager to track cleanup
    mcp_manager_instance = MyMCPSessionManager()
    mocker.patch(
        "lightspeed_stack_providers.providers.inline.agents.lightspeed_inline_agent.responses.openai_responses.MyMCPSessionManager",
        return_value=mcp_manager_instance,
    )

    # Mock close_all on the parent class
    close_all_mock = mocker.patch.object(
        mcp_manager_instance.__class__.__bases__[0], "close_all", return_value=None
    )

    # Call _create_streaming_response and expect an error
    from llama_stack_api.openai_responses import OpenAIResponseText

    response_generator = impl._create_streaming_response(
        input="test input",
        model="test-model",
        text=OpenAIResponseText(type="text"),
        max_infer_iters=10,
    )

    # Consume the async generator and expect ValueError
    with pytest.raises(ValueError, match="Test error in streaming"):
        async for _ in response_generator:
            pass

    # Verify close_all was still called (context manager cleanup on error)
    close_all_mock.assert_called_once()
