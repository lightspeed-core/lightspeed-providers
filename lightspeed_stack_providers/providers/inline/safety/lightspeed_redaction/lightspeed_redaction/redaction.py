import logging
import re
import uuid
from typing import Any

from llama_stack_api import (
    ModerationObject,
    ModerationObjectResults,
    RunModerationRequest,
    RunShieldRequest,
    RunShieldResponse,
    Safety,
    Shield,
    ShieldsProtocolPrivate,
)
from llama_stack_api.inference import (
    OpenAIAssistantMessageParam,
    OpenAIDeveloperMessageParam,
    OpenAISystemMessageParam,
    OpenAIToolMessageParam,
    OpenAIUserMessageParam,
)

from .config import (
    RedactionShieldConfig,
)

# Message type alias for run_shield parameter
Message = (
    OpenAIUserMessageParam
    | OpenAISystemMessageParam
    | OpenAIAssistantMessageParam
    | OpenAIToolMessageParam
    | OpenAIDeveloperMessageParam
)

log = logging.getLogger(__name__)


class RedactionShieldImpl(Safety, ShieldsProtocolPrivate):
    """Redaction shield that mutates messages with inline rules."""

    def __init__(self, config: RedactionShieldConfig, deps: dict[str, Any]) -> None:
        """
        Initialize the redaction shield and precompile regex replacement rules from configuration.

        Parameters:
            - config (RedactionShieldConfig): Configuration containing
              redaction rules and options (for example, rule patterns,
              replacements, and case sensitivity).
            - deps (dict[str, Any]): Optional dependency map used by the
              implementation (services or helpers).
        """
        self.config: RedactionShieldConfig = config
        self.compiled_rules: list[dict[str, Any]] = self._compile_rules()

    def _compile_rules(self) -> list[dict[str, Any]]:
        """Compile regex patterns from configuration rules.

        Compile configured regex redaction rules into executable patterns.

        Each rule from self.config.rules is compiled (respecting
        self.config.case_sensitive) into a dictionary with keys:
        - "pattern": the compiled `re.Pattern` object
        - "replacement": the rule's replacement string
        - "original_pattern": the original pattern string

        Invalid or failing patterns are logged and skipped.

        Returns:
            list[dict[str, Any]]: Compiled rule dictionaries as described above.
        """
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

    async def shutdown(self) -> None:
        """Shutdown the shield."""

    async def register_shield(self, shield: Shield) -> None:
        """Register a shield.

        Register a shield instance with this redaction implementation.

        Parameters:
            shield (Shield): Shield instance to register. Current
            implementation is a no-op and does not persist or modify the
            shield.
        """

    async def run_shield(
        self,
        request: RunShieldRequest,
    ) -> RunShieldResponse:
        """Run the redaction shield - mutates messages directly.

        Apply configured redaction rules to each message in-place.

        Each message in `messages` that has a string `content` will be
        processed; if redaction modifies the text, the message's `content` is
        replaced with the redacted string.

        Parameters:
            - request (RunShieldRequest): Request containing shield_id and
              messages; any message with a string `content` may be mutated
              in place.

        Returns:
            RunShieldResponse: A response object; `violation` is `None` for this implementation.
        """
        messages = request.messages
        for message in messages:
            if hasattr(message, "content") and isinstance(message.content, str):
                original_content: str = message.content
                redacted_content: str = self._apply_redaction_rules(original_content)

                if redacted_content != original_content:
                    message.content = redacted_content  # Mutating in-place

        return RunShieldResponse(violation=None)

    async def run_moderation(self, request: RunModerationRequest) -> ModerationObject:
        """Run moderation on input text, checking for sensitive data.

        When sensitive data is detected, the content is flagged and blocked.
        This serves as a safety net for flows that don't use run_shield.
        For flows using lightspeed_inline_agent, redaction happens in the agent.

        Returns:
            ModerationObject: Contains a generated `id`, `model` (uses
            `request.model` or "lightspeed-redaction" if unspecified), and a
            `results` list where each ModerationObjectResults includes
            `flagged` status and metadata with `{"contains_sensitive_data":
                                                 True}` for altered inputs and
            `{"contains_sensitive_data": False}` for unchanged inputs.
        """
        inputs = request.input if isinstance(request.input, list) else [request.input]
        results = []

        for text_input in inputs:
            redacted = self._apply_redaction_rules(text_input)
            contains_sensitive_data = redacted != text_input

            if contains_sensitive_data:
                results.append(
                    ModerationObjectResults(
                        flagged=True,
                        categories={"sensitive_data": True},
                        category_scores={"sensitive_data": 1.0},
                        category_applied_input_types={"sensitive_data": ["text"]},
                        user_message=(
                            "Your message appears to contain sensitive information "
                            "(such as passwords, API keys, or other secrets). "
                            "Please remove sensitive data and try again."
                        ),
                        metadata={"contains_sensitive_data": True},
                    )
                )
            else:
                results.append(
                    ModerationObjectResults(
                        flagged=False,
                        categories={},
                        category_scores={},
                        metadata={"contains_sensitive_data": False},
                    )
                )

        return ModerationObject(
            id=f"modr-{uuid.uuid4()}",
            model=request.model or "lightspeed-redaction",
            results=results,
        )

    def _apply_redaction_rules(self, content: str) -> str:
        """Apply all redaction rules to content.

        Apply configured regex redaction rules to the provided text and return
        the resulting string.

        If `content` is empty or there are no compiled rules, the original
        `content` is returned unchanged. When one or more rules match, their
        substitutions are applied sequentially and the transformed string is
        returned.

        Returns:
            str: The text after applying redaction substitutions; the original
            text if no changes were made.
        """
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
