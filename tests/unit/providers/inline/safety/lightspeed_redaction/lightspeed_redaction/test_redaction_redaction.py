import pytest
from lightspeed_stack_providers.providers.inline.safety.lightspeed_redaction.lightspeed_redaction.redaction import (
    RedactionShieldImpl,
)
from lightspeed_stack_providers.providers.inline.safety.lightspeed_redaction.lightspeed_redaction.config import (
    RedactionShieldConfig,
    PatternReplacement,
)
from llama_stack.apis.inference import UserMessage


@pytest.fixture
def redaction_shield_impl():
    """Fixture for creating a RedactionShieldImpl instance."""
    config = RedactionShieldConfig(
        rules=[
            PatternReplacement(pattern="secret", replacement="[REDACTED]"),
            PatternReplacement(pattern=r"\b\d{4}\b", replacement="[YEAR]"),
        ]
    )
    return RedactionShieldImpl(config, {})


def test_compile_rules(redaction_shield_impl):
    """Test that the regex rules are compiled correctly."""
    assert len(redaction_shield_impl.compiled_rules) == 2
    assert redaction_shield_impl.compiled_rules[0]["original_pattern"] == "secret"
    assert redaction_shield_impl.compiled_rules[1]["original_pattern"] == r"\b\d{4}\b"


def test_apply_redaction_rules(redaction_shield_impl):
    """Test that the redaction rules are applied correctly."""
    content = "This is a secret message from 2023."
    redacted_content = redaction_shield_impl._apply_redaction_rules(content)
    assert redacted_content == "This is a [REDACTED] message from [YEAR]."


def test_apply_redaction_rules_case_insensitive(redaction_shield_impl):
    """Test that the redaction rules are applied case-insensitively."""
    content = "This is a Secret message from 2023."
    redacted_content = redaction_shield_impl._apply_redaction_rules(content)
    assert redacted_content == "This is a [REDACTED] message from [YEAR]."


@pytest.mark.asyncio
async def test_run_shield(redaction_shield_impl):
    """Test the run_shield method."""
    messages = [UserMessage(content="This is a secret message from 2023.")]
    response = await redaction_shield_impl.run_shield("test_shield", messages)
    assert response.violation is None
    assert messages[0].content == "This is a [REDACTED] message from [YEAR]."
