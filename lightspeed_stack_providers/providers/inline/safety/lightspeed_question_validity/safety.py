import logging
import uuid
from string import Template
from typing import Any

from llama_stack_api import (
    Api,
    Inference,
    ModerationObject,
    ModerationObjectResults,
    RunShieldResponse,
    Safety,
    SafetyViolation,
    ShieldsProtocolPrivate,
    UserMessage,
    ViolationLevel,
)
from llama_stack_api.inference import (
    OpenAIAssistantMessageParam,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIDeveloperMessageParam,
    OpenAISystemMessageParam,
    OpenAIToolMessageParam,
    OpenAIUserMessageParam,
)

# Message type alias for run_shield parameter
Message = (
    OpenAIUserMessageParam
    | OpenAISystemMessageParam
    | OpenAIAssistantMessageParam
    | OpenAIToolMessageParam
    | OpenAIDeveloperMessageParam
)

from lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.config import (
    QuestionValidityShieldConfig,
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

    async def run_moderation(
        self, input: str | list[str], model: str | None = None
    ) -> ModerationObject:
        """Run moderation on input text to check if it's a valid question."""
        inputs = input if isinstance(input, list) else [input]
        results = []

        impl = QuestionValidityRunner(
            model_id=self.config.model_id,
            model_prompt_template=self.model_prompt_template,
            invalid_question_response=self.config.invalid_question_response,
            inference_api=self.inference_api,
        )

        for text in inputs:
            run_response = await impl.run(UserMessage(content=text))
            results.append(self._get_moderation_object_results(run_response))

        return ModerationObject(
            id=f"modr-{uuid.uuid4()}",
            model=model or self.config.model_id,
            results=results,
        )

    def _get_moderation_object_results(
        self, run_shield_response: RunShieldResponse
    ) -> ModerationObjectResults:
        """Convert RunShieldResponse to ModerationObjectResults."""
        if run_shield_response.violation is None:
            return ModerationObjectResults(
                flagged=False,
                categories={},
                category_scores={"question_validity": 0.0},
                category_applied_input_types={},
                user_message=None,
                metadata={},
            )
        else:
            return ModerationObjectResults(
                flagged=True,
                categories={"question_validity": True},
                category_scores={"question_validity": 1.0},
                category_applied_input_types={"question_validity": ["text"]},
                user_message=run_shield_response.violation.user_message,
                metadata={
                    "violation_level": run_shield_response.violation.violation_level.value
                },
            )

    async def run_shield(
        self,
        shield_id: str,
        messages: list[Message],
        params: dict[str, Any] = None,
    ) -> RunShieldResponse:
        # Take last user message (can be UserMessage or OpenAIUserMessageParam)
        user_messages = [
            m
            for m in messages
            if isinstance(m, (UserMessage, OpenAIUserMessageParam))
            or (hasattr(m, "role") and m.role == "user")
        ]
        if not user_messages:
            return RunShieldResponse(violation=None)
        last_msg = user_messages[-1]
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        message = UserMessage(content=content)
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

        request = OpenAIChatCompletionRequestWithExtraBody(
            model=self.model_id,
            messages=[
                OpenAIUserMessageParam(
                    role="user", content=shield_input_message.content
                ),
            ],
            stream=False,
        )
        response = await self.inference_api.openai_chat_completion(request)
        content = response.choices[0].message.content
        content = content.strip()
        return self.get_shield_response(content)
