import pytest
from string import Template

from lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.safety import (
    QuestionValidityRunner,
    SUBJECT_ALLOWED,
    SUBJECT_REJECTED,
)
from llama_stack.apis.inference import UserMessage, CompletionMessage
from llama_stack.apis.safety import SafetyViolation, ViolationLevel, RunShieldResponse
from llama_stack.apis.inference import ChatCompletionResponse


@pytest.fixture
def mock_inference_api(mocker):
    """Fixture for mocking the Inference API."""
    return mocker.AsyncMock()


@pytest.fixture
def question_validity_runner(mock_inference_api):
    """Fixture for creating a QuestionValidityRunner instance."""
    model_id = "test_model"
    model_prompt_template = Template(
        "Is this question allowed? Answer ${allowed} or ${rejected}. Question: ${message}"
    )
    invalid_question_response = "Invalid question."
    return QuestionValidityRunner(
        model_id=model_id,
        model_prompt_template=model_prompt_template,
        invalid_question_response=invalid_question_response,
        inference_api=mock_inference_api,
    )


def test_build_prompt(question_validity_runner):
    """Test that the prompt is built correctly."""
    message = UserMessage(content="How do I create a Kubernetes service?")
    prompt = question_validity_runner.build_prompt(message)
    assert "Is this question allowed?" in prompt
    assert SUBJECT_ALLOWED in prompt
    assert SUBJECT_REJECTED in prompt
    assert message.content in prompt


def test_get_shield_response_allowed(question_validity_runner):
    """Test that the shield response is correct for an allowed question."""
    response = question_validity_runner.get_shield_response(SUBJECT_ALLOWED)
    assert response.violation is None


def test_get_shield_response_rejected(question_validity_runner):
    """Test that the shield response is correct for a rejected question."""
    response = question_validity_runner.get_shield_response(SUBJECT_REJECTED)
    assert isinstance(response.violation, SafetyViolation)
    assert response.violation.violation_level == ViolationLevel.ERROR
    assert (
        response.violation.user_message
        == question_validity_runner.invalid_question_response
    )


@pytest.mark.asyncio
async def test_run_allowed(question_validity_runner, mock_inference_api):
    """Test the run method for an allowed question."""
    message = UserMessage(content="How do I create a Kubernetes service?")
    mock_inference_api.chat_completion.return_value = ChatCompletionResponse(
        completion_message=CompletionMessage(
            role="assistant", content=SUBJECT_ALLOWED, stop_reason="end_of_turn"
        )
    )

    response = await question_validity_runner.run(message)

    assert response.violation is None
    mock_inference_api.chat_completion.assert_called_once()


@pytest.mark.asyncio
async def test_run_rejected(question_validity_runner, mock_inference_api):
    """Test the run method for a rejected question."""
    message = UserMessage(content="What is the weather today?")
    mock_inference_api.chat_completion.return_value = ChatCompletionResponse(
        completion_message=CompletionMessage(
            role="assistant", content=SUBJECT_REJECTED, stop_reason="end_of_turn"
        )
    )

    response = await question_validity_runner.run(message)

    assert isinstance(response.violation, SafetyViolation)
    mock_inference_api.chat_completion.assert_called_once()


@pytest.fixture
def question_validity_shield_impl(mock_inference_api):
    """Fixture for creating a QuestionValidityShieldImpl instance."""
    from lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.safety import (
        QuestionValidityShieldImpl,
    )
    from lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.config import (
        QuestionValidityShieldConfig,
    )
    from llama_stack.apis.datatypes import Api

    config = QuestionValidityShieldConfig()
    deps = {Api.inference: mock_inference_api}
    shield = QuestionValidityShieldImpl(config, deps)
    return shield


@pytest.mark.asyncio
async def test_run_shield_allowed(question_validity_shield_impl, mocker):
    """Test the run_shield method for an allowed question."""
    mock_runner = mocker.patch(
        "lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.safety.QuestionValidityRunner"
    )
    mock_runner.return_value.run = mocker.AsyncMock(
        return_value=RunShieldResponse(violation=None)
    )
    messages = [UserMessage(content="How do I create a Kubernetes service?")]

    response = await question_validity_shield_impl.run_shield("test_shield", messages)

    assert response.violation is None
    mock_runner.return_value.run.assert_called_once()


@pytest.mark.asyncio
async def test_run_shield_rejected(question_validity_shield_impl, mocker):
    """Test the run_shield method for a rejected question."""
    mock_runner = mocker.patch(
        "lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.safety.QuestionValidityRunner"
    )
    mock_runner.return_value.run = mocker.AsyncMock(
        return_value=RunShieldResponse(
            violation=SafetyViolation(
                violation_level=ViolationLevel.ERROR,
                user_message="Invalid question.",
            )
        )
    )
    messages = [UserMessage(content="What is the weather today?")]

    response = await question_validity_shield_impl.run_shield("test_shield", messages)

    assert isinstance(response.violation, SafetyViolation)
    mock_runner.return_value.run.assert_called_once()


@pytest.mark.asyncio
async def test_run_moderation_allowed(question_validity_shield_impl, mocker):
    """Test the run_moderation method for an allowed question."""
    mock_runner = mocker.patch(
        "lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.safety.QuestionValidityRunner"
    )
    mock_runner.return_value.run = mocker.AsyncMock(
        return_value=RunShieldResponse(violation=None)
    )

    result = await question_validity_shield_impl.run_moderation(
        "How do I create a Kubernetes service?", "test_model"
    )

    assert not result.flagged


@pytest.mark.asyncio
async def test_run_moderation_rejected(question_validity_shield_impl, mocker):
    """Test the run_moderation method for a rejected question."""
    mock_runner = mocker.patch(
        "lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.safety.QuestionValidityRunner"
    )
    mock_runner.return_value.run = mocker.AsyncMock(
        return_value=RunShieldResponse(
            violation=SafetyViolation(
                violation_level=ViolationLevel.ERROR,
                user_message="Invalid question.",
            )
        )
    )

    result = await question_validity_shield_impl.run_moderation(
        "What is the weather today?", "test_model"
    )

    assert result.flagged
