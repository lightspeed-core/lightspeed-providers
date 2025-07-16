import logging
import re
import tempfile
from typing import Any

from llama_stack.apis.inference import (
    CompletionMessage,
    Message,
)
from llama_stack.apis.shields import Shield
from llama_stack.apis.safety import (
    SafetyViolation,
    ViolationLevel,
    RunShieldResponse,
    Safety,
)

from lightspeed_stack_providers.providers.inline.safety.aap_inventory_validator.aap_inventory_tool import InventoryValidator
from lightspeed_stack_providers.providers.inline.safety.aap_inventory_validator.config import InventoryValidatorConfig

log = logging.getLogger(__name__)


class InventoryValidatorImpl(Safety):
    INI_FILE_PATTERN = re.compile(r'```ini\s*\n(.+)\n```', re.DOTALL)

    def __init__(self, config: InventoryValidatorConfig) -> None:
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        pass

    async def run_shield(
        self,
        shield_id: str,
        messages: list[Message],
        params: dict[str, Any] = None,
    ) -> RunShieldResponse:
        violation = None

        inventory_string = self.extract_inventory_string(messages)
        if inventory_string:
            with tempfile.NamedTemporaryFile(delete_on_close=False) as fp:
                fp.write(inventory_string.encode(encoding="utf-8"))
                fp.close()

                validator = InventoryValidator('containerized', 'growth')  # TODO
                result = validator.validate_inventory(fp.name)

                log.debug(result)
                if len(validator.errors) > 0:
                    violation = SafetyViolation(
                        violation_level=ViolationLevel.ERROR,
                        user_message="\n".join(validator.errors)
                    )

        return RunShieldResponse(violation=violation)

    def extract_inventory_string(self, messages: list[Message]) -> str | None:
        if not messages:
            log.debug("No messages are found to validate.")
            return None

        last_message: CompletionMessage = messages[- 1]
        if not isinstance(last_message, CompletionMessage):
            log.debug("The last message received is not a CompletionMessage.")
            return None

        m = InventoryValidatorImpl.INI_FILE_PATTERN.findall(last_message.content)
        if not m:
            log.debug("No inventory file was found in the last message.")
            return None
        elif len(m) > 1:
            log.warning("More than one ({len(m}) inventory files were found in the last message.")

        return m[0]


