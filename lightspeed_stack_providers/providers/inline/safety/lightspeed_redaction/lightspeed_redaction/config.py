import re
from typing import Any, Optional, Self

from pydantic import BaseModel, Field, model_validator


class PatternReplacement(BaseModel):
    """A single redaction pattern and its replacement."""

    pattern: str = Field(description="Regular expression pattern to match")
    replacement: str = Field(description="Text to replace matches with")

    @model_validator(mode="after")
    def validate_regex_pattern(self) -> Self:
        """Validate that the pattern is a valid regular expression.

        Ensure the model's `pattern` field is a valid regular expression.

        Raises:
            ValueError: If `pattern` cannot be compiled as a regular
            expression; the error message includes the invalid pattern and the
            underlying regex error.

        Returns:
            Self: The validated model instance.
        """
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
        rules: Optional[list[dict]] = None,
        case_sensitive: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Build a sample run configuration dictionary for redaction rules.

        Parameters:
            - rules (Optional[list[dict]): Optional list of raw rule mappings
              (each with `pattern` and `replacement`) to include as the
              `"rules"` value.
            - case_sensitive (bool): Whether matching should be case sensitive;
              set as the `"case_sensitive"` value.
            - **kwargs: Additional ignored keyword arguments accepted for compatibility.

        Returns:
            dict[str, Any]: A dictionary with keys `"rules"` (the provided
            `rules`) and `"case_sensitive"` (the provided `case_sensitive`).
        """
        return {
            "rules": rules,
            "case_sensitive": case_sensitive,
        }
