"""Unit tests for Lightspeed redaction provider behaviour."""

import pytest
from llama_stack_api import RunShieldRequest
from llama_stack_api.inference import OpenAIUserMessageParam

# pylint: disable=line-too-long
from lightspeed_stack_providers.providers.inline.safety.lightspeed_redaction.lightspeed_redaction.config import (
    PatternReplacement,
    RedactionShieldConfig,
)

# pylint: disable=line-too-long
from lightspeed_stack_providers.providers.inline.safety.lightspeed_redaction.lightspeed_redaction.redaction import (
    Message,
    RedactionShieldImpl,
)


@pytest.fixture(name="redaction_shield_implementation")
def redaction_shield_impl() -> RedactionShieldImpl:
    """Fixture for creating a RedactionShieldImpl instance.

    Create a RedactionShieldImpl configured for unit tests.

    The instance is initialized with two pattern-replacement rules:
    - Replace "secret" with "[REDACTED]".
    - Replace any standalone four-digit sequence (regex `\\b\\d{4}\\b`) with "[YEAR]".

    Returns:
        A RedactionShieldImpl configured with the above pattern replacements.
    """
    config = RedactionShieldConfig(
        rules=[
            PatternReplacement(pattern="secret", replacement="[REDACTED]"),
            PatternReplacement(pattern=r"\b\d{4}\b", replacement="[YEAR]"),
        ]
    )
    return RedactionShieldImpl(config, {})


def test_compile_rules(redaction_shield_implementation: RedactionShieldImpl) -> None:
    """Test that the regex rules are compiled correctly."""
    assert len(redaction_shield_implementation.compiled_rules) == 2
    assert (
        redaction_shield_implementation.compiled_rules[0]["original_pattern"]
        == "secret"
    )
    assert (
        redaction_shield_implementation.compiled_rules[1]["original_pattern"]
        == r"\b\d{4}\b"
    )


def test_apply_redaction_rules(
    redaction_shield_implementation: RedactionShieldImpl,
) -> None:
    """Test that the redaction rules are applied correctly."""
    content = "This is a secret message from 2023."
    # pylint: disable=protected-access
    redacted_content = redaction_shield_implementation._apply_redaction_rules(content)
    assert redacted_content == "This is a [REDACTED] message from [YEAR]."


def test_apply_redaction_rules_case_insensitive(
    redaction_shield_implementation: RedactionShieldImpl,
) -> None:
    """Test that the redaction rules are applied case-insensitively."""
    content = "This is a Secret message from 2023."
    # pylint: disable=protected-access
    redacted_content = redaction_shield_implementation._apply_redaction_rules(content)
    assert redacted_content == "This is a [REDACTED] message from [YEAR]."


@pytest.mark.asyncio
async def test_run_shield(redaction_shield_implementation: RedactionShieldImpl) -> None:
    """Test the run_shield method."""
    messages: list[Message] = [
        OpenAIUserMessageParam(
            role="user", content="This is a secret message from 2023."
        )
    ]
    request = RunShieldRequest(shield_id="test_shield", messages=messages)
    response = await redaction_shield_implementation.run_shield(request)
    assert response.violation is None
    assert messages[0].content == "This is a [REDACTED] message from [YEAR]."
