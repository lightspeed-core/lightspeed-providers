import pytest
from pydantic import ValidationError

from lightspeed_stack_providers.providers.inline.safety.lightspeed_redaction.lightspeed_redaction.config import (
    PatternReplacement,
    RedactionShieldConfig,
)


def test_pattern_replacement_valid_regex():
    """Test that a valid regex pattern is accepted."""
    pattern = r"\b\d{4}\b"
    replacement = "[YEAR]"
    pr = PatternReplacement(pattern=pattern, replacement=replacement)
    assert pr.pattern == pattern
    assert pr.replacement == replacement


def test_pattern_replacement_invalid_regex():
    """Test that an invalid regex pattern raises a ValueError."""
    with pytest.raises(ValidationError) as excinfo:
        PatternReplacement(pattern="[", replacement="invalid")
    assert "Invalid regular expression pattern" in str(excinfo.value)


def test_redaction_shield_config_defaults():
    """Test that the RedactionShieldConfig model can be instantiated with default values."""
    config = RedactionShieldConfig()
    assert config.rules == []
    assert not config.case_sensitive


def test_redaction_shield_config_custom_values():
    """Test that the RedactionShieldConfig model correctly assigns custom values."""
    rules = [{"pattern": "foo", "replacement": "bar"}]
    config = RedactionShieldConfig(rules=rules, case_sensitive=True)
    assert len(config.rules) == 1
    assert config.rules[0].pattern == "foo"
    assert config.rules[0].replacement == "bar"
    assert config.case_sensitive


def test_sample_run_config():
    """Test that the sample_run_config class method returns the expected dictionary."""
    expected_config = {
        "rules": None,
        "case_sensitive": False,
    }
    assert RedactionShieldConfig.sample_run_config() == expected_config


def test_sample_run_config_with_custom_values():
    """Test that the sample_run_config class method returns the expected dictionary with custom values."""
    custom_rules = [{"pattern": "test", "replacement": "tested"}]
    expected_config = {
        "rules": custom_rules,
        "case_sensitive": True,
    }
    assert (
        RedactionShieldConfig.sample_run_config(rules=custom_rules, case_sensitive=True)
        == expected_config
    )
