import logging
import uuid
from typing import Any
from string import Template

from lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.config import (
    QuestionValidityShieldConfig,
)

from llama_stack.apis.shields import Shield
from llama_stack.apis.datatypes import Api
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
from llama_stack.apis.safety import (
    SafetyViolation,
    ViolationLevel,
    RunShieldResponse,
    Safety,
)
from llama_stack.apis.safety.safety import (
    ModerationObject,
    ModerationObjectResults,
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
        self.shield_store = {}

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        self.shield_store[shield.identifier] = shield

    async def run_moderation(self, input: str | list[str], model: str) -> ModerationObject:
        """Run moderation on input text to check if it's a valid question."""
        inputs = input if isinstance(input, list) else [input]
        results = []

        for text_input in inputs:
            log.info(f"Running QuestionValidityShield moderation on input: {text_input[:100]}...")
            try:
                user_msg = UserMessage(content=text_input)
                
                impl = QuestionValidityRunner(
                    model_id=self.config.model_id,
                    model_prompt_template=self.model_prompt_template,
                    invalid_question_response=self.config.invalid_question_response,
                    inference_api=self.inference_api,
                )

                run_shield_response = await impl.run(user_msg)
                moderation_result = self._get_moderation_object_results(run_shield_response)
                
            except Exception as e:
                log.error(f"QuestionValidityShield moderation failed: {e}")
                # Create safe fallback response on failure to avoid blocking legitimate requests
                moderation_result = ModerationObjectResults(
                    flagged=False,
                    categories={},
                    category_scores={},
                    category_applied_input_types={},
                    user_message=None,
                    metadata={"moderation_error": str(e)},
                )
            results.append(moderation_result)

        return ModerationObject(id=str(uuid.uuid4()), model=model, results=results)

    def _get_moderation_object_results(self, run_shield_response: RunShieldResponse) -> ModerationObjectResults:
        """Convert RunShieldResponse to ModerationObjectResults."""
        if run_shield_response.violation is None:
            # Safe result - question is valid
            return ModerationObjectResults(
                flagged=False,
                categories={},
                category_scores={"question_validity": 0.0},
                category_applied_input_types={},
                user_message=None,
                metadata={},
            )
        else:
            # Unsafe result - question is invalid
            return ModerationObjectResults(
                flagged=True,
                categories={"question_validity": run_shield_response.violation.violation_level.value},
                category_scores={"question_validity": 1.0},
                category_applied_input_types={"question_validity": "text"},
                user_message=run_shield_response.violation.user_message,
                metadata={"violation_level": run_shield_response.violation.violation_level.value},
            )

    async def run_shield(
        self,
        shield_id: str,
        messages: list[Message],
        params: dict[str, Any] = None,
    ) -> RunShieldResponse:
        shield = self.shield_store.get(shield_id)
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
