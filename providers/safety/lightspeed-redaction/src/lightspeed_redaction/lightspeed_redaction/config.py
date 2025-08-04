import re
from typing import Any, Self
from pydantic import BaseModel, Field, model_validator


class PatternReplacement(BaseModel):
    """A single redaction pattern and its replacement."""

    pattern: str = Field(description="Regular expression pattern to match")
    replacement: str = Field(description="Text to replace matches with")

    @model_validator(mode="after")
    def validate_regex_pattern(self) -> Self:
        """Validate that the pattern is a valid regular expression."""
        try:
            re.compile(self.pattern)
        except re.error as e:
            raise ValueError(
                f"Invalid regular expression pattern '{self.pattern}': {e}"
            )
        return self


class RedactionShieldConfig(BaseModel):
    """Configuration for redaction shield with inline rules."""

    rules: list[PatternReplacement] = Field(
        default=[],
        description="List of redaction rules with pattern and replacement",
    )

    case_sensitive: bool = Field(
        default=False,
        description="Whether pattern matching is case sensitive",
    )

    @classmethod
    def sample_run_config(
        cls,
        rules: list[dict] | None = None,
        case_sensitive: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        return {
            "rules": rules,
            "case_sensitive": case_sensitive,
        }
