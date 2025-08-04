import logging
from typing import Any
from string import Template

from lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.config import (
    QuestionValidityShieldConfig,
)

from llama_stack.apis.shields import Shield
from llama_stack.distribution.datatypes import Api
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
from llama_stack.apis.safety import (
    SafetyViolation,
    ViolationLevel,
    RunShieldResponse,
    Safety,
)
from llama_stack.apis.inference import (
    Inference,
    Message,
    UserMessage,
)

log = logging.getLogger(__name__)


SUBJECT_REJECTED = "REJECTED"
SUBJECT_ALLOWED = "ALLOWED"


class QuestionValidityShieldImpl(Safety, ShieldsProtocolPrivate):

    def __init__(self, config: QuestionValidityShieldConfig, deps) -> None:
        self.config = config
        self.model_prompt_template = Template(f"{self.config.model_prompt}")
        self.inference_api = deps[Api.inference]

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
        shield = await self.shield_store.get_shield(shield_id)
        if not shield:
            raise ValueError(f"Unknown shield {shield_id}")

        messages = messages.copy()
        # [TODO] manstis: Ensure this is the latest User message
        message: UserMessage = messages[len(messages) - 1 :][0]
        log.debug(f"Shield UserMessage: {message.content}")

        impl = QuestionValidityRunner(
            model_id=self.config.model_id,
            model_prompt_template=self.model_prompt_template,
            invalid_question_response=self.config.invalid_question_response,
            inference_api=self.inference_api,
        )

        return await impl.run(message)


class QuestionValidityRunner:
    def __init__(
        self,
        model_id: str,
        model_prompt_template: Template,
        invalid_question_response: str,
        inference_api: Inference,
    ):
        self.model_id = model_id
        self.model_prompt_template = model_prompt_template
        self.invalid_question_response = invalid_question_response
        self.inference_api = inference_api

    def build_text_shield_input(self, message: UserMessage) -> UserMessage:
        return UserMessage(content=self.build_prompt(message))

    def build_prompt(self, message: UserMessage) -> str:
        prompt = self.model_prompt_template.substitute(
            allowed=SUBJECT_ALLOWED,
            rejected=SUBJECT_REJECTED,
            message=message.content,
        )
        log.debug(f"Shield prompt: {prompt}")
        return prompt

    def get_shield_response(self, response: str) -> RunShieldResponse:
        response = response.strip()
        log.debug(f"Shield response: {response}")

        if response == SUBJECT_ALLOWED:
            return RunShieldResponse(violation=None)

        return RunShieldResponse(
            violation=SafetyViolation(
                violation_level=ViolationLevel.ERROR,
                user_message=self.invalid_question_response,
            ),
        )

    async def run(self, message: UserMessage) -> RunShieldResponse:
        shield_input_message = self.build_text_shield_input(message)
        log.debug(f"Shield input message: {shield_input_message}")

        response = await self.inference_api.chat_completion(
            model_id=self.model_id,
            messages=[shield_input_message],
            stream=False,
        )
        content = response.completion_message.content
        content = content.strip()
        return self.get_shield_response(content)
