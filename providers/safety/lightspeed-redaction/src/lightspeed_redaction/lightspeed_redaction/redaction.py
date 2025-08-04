import logging
import re
from typing import Any, Optional

from lightspeed_stack_providers.providers.inline.safety.lightspeed_redaction.config import (
    RedactionShieldConfig,
    PatternReplacement,
)

from llama_stack.apis.shields import Shield
from llama_stack.distribution.datatypes import Api
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
from llama_stack.apis.safety import (
    RunShieldResponse,
    Safety,
)
from llama_stack.apis.inference import (
    Message,
    UserMessage,
)

log = logging.getLogger(__name__)


class RedactionShieldImpl(Safety, ShieldsProtocolPrivate):
    """Redaction shield that mutates messages with inline rules."""

    def __init__(self, config: RedactionShieldConfig, deps: dict[str, Any]) -> None:
        self.config: RedactionShieldConfig = config
        self.compiled_rules: list[dict[str, Any]] = self._compile_rules()

    def _compile_rules(self) -> list[dict[str, Any]]:
        """Compile regex patterns from configuration rules."""
        compiled_rules: list[dict[str, Any]] = []

        for rule in self.config.rules:
            try:
                flags: int = 0 if self.config.case_sensitive else re.IGNORECASE
                compiled_pattern = re.compile(rule.pattern, flags)

                compiled_rules.append(
                    {
                        "pattern": compiled_pattern,
                        "replacement": rule.replacement,
                        "original_pattern": rule.pattern,
                    }
                )

                log.debug(f"Compiled redaction rule: {rule.pattern}")

            except re.error as e:
                log.error(f"Invalid regex pattern '{rule.pattern}': {e}")
            except Exception as e:
                log.error(f"Error compiling rule {rule.pattern}: {e}")

        log.info(f"Compiled {len(compiled_rules)} redaction rules")
        return compiled_rules

    async def initialize(self) -> None:
        """Initialize the shield."""
        pass

    async def shutdown(self) -> None:
        """Shutdown the shield."""
        pass

    async def register_shield(self, shield: Shield) -> None:
        """Register a shield."""
        pass

    async def run_shield(
        self,
        shield_id: str,
        messages: list[Message],
        params: Optional[dict[str, Any]] = None,
    ) -> RunShieldResponse:
        """Run the redaction shield - mutates messages directly."""

        for message in messages:
            if isinstance(message, UserMessage) and isinstance(message.content, str):
                original_content: str = message.content
                redacted_content: str = self._apply_redaction_rules(original_content)

                if redacted_content != original_content:
                    message.content = redacted_content  # Mutating in-place

        return RunShieldResponse(violation=None)

    def _apply_redaction_rules(self, content: str) -> str:
        """Apply all redaction rules to content."""
        if not content or not self.compiled_rules:
            return content

        redacted_content: str = content
        applied_rules: list[str] = []

        for rule in self.compiled_rules:
            try:
                if rule["pattern"].search(redacted_content):
                    redacted_content = rule["pattern"].sub(
                        rule["replacement"], redacted_content
                    )
                    applied_rules.append(rule["original_pattern"])

            except Exception as e:
                log.debug(f"Error applying rule {rule['original_pattern']}: {e}")

        if applied_rules:
            log.debug(f"Applied {len(applied_rules)} redaction rules to content")

        return redacted_content
