"""Tests for the validity shield provider behaviour."""

from string import Template
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_stack_api import RunModerationRequest, RunShieldRequest
from llama_stack_api.inference import OpenAIUserMessageParam
from llama_stack_api.safety import RunShieldResponse, SafetyViolation, ViolationLevel
from pytest_mock import MockerFixture

from lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.safety import (
    SUBJECT_ALLOWED,
    SUBJECT_REJECTED,
    QuestionValidityRunner,
    QuestionValidityShieldImpl,
)


@pytest.fixture
def mock_inference_api() -> AsyncMock:
    """Fixture for mocking the Inference API.

    Provide a pytest fixture that supplies an AsyncMock
    representing the inference API client for tests.

    This mock is intended to stand in for the real inference API
    and can be configured by tests to return specific async
    responses or raise exceptions.

    Returns:
        AsyncMock: An asynchronous mock object that mimics the inference API client.
    """
    return AsyncMock()


@pytest.fixture
def question_validity_runner(mock_inference_api: AsyncMock) -> QuestionValidityRunner:
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


def create_mock_chat_response(content: str) -> MagicMock:
    """Create a mock OpenAI chat completion response.

    Parameters:
        content (str): The message content to place at response.choices[0].message.content.

    Returns:
        MagicMock: A mock response where `choices[0].message.content == content`.
    """
    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    return mock_response


def test_build_prompt(question_validity_runner: QuestionValidityRunner) -> None:
    """Test that the prompt is built correctly."""
    message = OpenAIUserMessageParam(
        role="user", content="How do I create a Kubernetes service?"
    )
    prompt = question_validity_runner.build_prompt(message)
    assert "Is this question allowed?" in prompt
    assert SUBJECT_ALLOWED in prompt
    assert SUBJECT_REJECTED in prompt
    assert isinstance(message.content, str)
    assert message.content in prompt


def test_get_shield_response_allowed(
    question_validity_runner: QuestionValidityRunner,
) -> None:
    """Test that the shield response is correct for an allowed question."""
    response = question_validity_runner.get_shield_response(SUBJECT_ALLOWED)
    assert response.violation is None


def test_get_shield_response_rejected(
    question_validity_runner: QuestionValidityRunner,
) -> None:
    """Test that the shield response is correct for a rejected question."""
    response = question_validity_runner.get_shield_response(SUBJECT_REJECTED)
    assert isinstance(response.violation, SafetyViolation)
    assert response.violation.violation_level == ViolationLevel.ERROR
    assert (
        response.violation.user_message
        == question_validity_runner.invalid_question_response
    )


@pytest.mark.asyncio
async def test_run_allowed(
    question_validity_runner: QuestionValidityRunner, mock_inference_api: AsyncMock
) -> None:
    """Test the run method for an allowed question."""
    message = OpenAIUserMessageParam(
        role="user", content="How do I create a Kubernetes service?"
    )
    mock_inference_api.openai_chat_completion.return_value = create_mock_chat_response(
        SUBJECT_ALLOWED
    )

    response = await question_validity_runner.run(message)

    assert response.violation is None
    mock_inference_api.openai_chat_completion.assert_called_once()


@pytest.mark.asyncio
async def test_run_rejected(
    question_validity_runner: QuestionValidityRunner, mock_inference_api: AsyncMock
) -> None:
    """Test the run method for a rejected question."""
    message = OpenAIUserMessageParam(role="user", content="What is the weather today?")
    mock_inference_api.openai_chat_completion.return_value = create_mock_chat_response(
        SUBJECT_REJECTED
    )

    response = await question_validity_runner.run(message)

    assert isinstance(response.violation, SafetyViolation)
    mock_inference_api.openai_chat_completion.assert_called_once()


@pytest.fixture
def question_validity_shield_impl(
    mock_inference_api: AsyncMock,
) -> QuestionValidityShieldImpl:
    """Fixture for creating a QuestionValidityShieldImpl instance."""
    from llama_stack_api import Api

    from lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.config import (
        QuestionValidityShieldConfig,
    )
    from lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.safety import (
        QuestionValidityShieldImpl,
    )

    config = QuestionValidityShieldConfig()
    deps = {Api.inference: mock_inference_api}
    return QuestionValidityShieldImpl(config, deps)


@pytest.mark.asyncio
async def test_run_shield_allowed(
    question_validity_shield_impl: QuestionValidityShieldImpl, mocker: MockerFixture
) -> None:
    """Test the run_shield method for an allowed question."""
    mock_runner = mocker.patch(
        "lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.safety.QuestionValidityRunner"
    )
    mock_runner.return_value.run = mocker.AsyncMock(
        return_value=RunShieldResponse(violation=None)
    )
    messages = [
        OpenAIUserMessageParam(
            role="user", content="How do I create a Kubernetes service?"
        )
    ]
    request = RunShieldRequest(shield_id="test_shield", messages=messages)

    response = await question_validity_shield_impl.run_shield(request)

    assert response.violation is None
    mock_runner.return_value.run.assert_called_once()


@pytest.mark.asyncio
async def test_run_shield_rejected(
    question_validity_shield_impl: QuestionValidityShieldImpl, mocker: MockerFixture
) -> None:
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
    messages = [
        OpenAIUserMessageParam(role="user", content="What is the weather today?")
    ]
    request = RunShieldRequest(shield_id="test_shield", messages=messages)

    response = await question_validity_shield_impl.run_shield(request)

    assert isinstance(response.violation, SafetyViolation)
    mock_runner.return_value.run.assert_called_once()


@pytest.mark.asyncio
async def test_run_moderation_allowed(
    question_validity_shield_impl: QuestionValidityShieldImpl, mocker: MockerFixture
) -> None:
    """Test the run_moderation method for an allowed question."""
    mock_runner = mocker.patch(
        "lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.safety.QuestionValidityRunner"
    )
    mock_runner.return_value.run = mocker.AsyncMock(
        return_value=RunShieldResponse(violation=None)
    )

    result = await question_validity_shield_impl.run_moderation(
        RunModerationRequest(
            input="How do I create a Kubernetes service?", model="test_model"
        )
    )

    assert not result.results[0].flagged


@pytest.mark.asyncio
async def test_run_moderation_rejected(
    question_validity_shield_impl: QuestionValidityShieldImpl, mocker: MockerFixture
) -> None:
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
        RunModerationRequest(input="What is the weather today?", model="test_model")
    )

    assert result.results[0].flagged
