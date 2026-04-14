import logging
import uuid
from string import Template

from llama_stack_api import (
    Api,
    Inference,
    ModerationObject,
    ModerationObjectResults,
    RunModerationRequest,
    RunShieldRequest,
    RunShieldResponse,
    Safety,
    SafetyViolation,
    ShieldsProtocolPrivate,
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

from lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.config import (
    QuestionValidityShieldConfig,
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


SUBJECT_REJECTED = "REJECTED"
SUBJECT_ALLOWED = "ALLOWED"


class QuestionValidityShieldImpl(Safety, ShieldsProtocolPrivate):
    def __init__(self, config: QuestionValidityShieldConfig, deps) -> None:
        """
        Initialize the shield implementation with its configuration and runtime dependencies.

        Parameters:
            - config (QuestionValidityShieldConfig): Configuration containing
              `model_prompt`, `model_id`, and `invalid_question_response`;
              `model_prompt` is compiled into a Template for prompt generation.
            - deps (Mapping): Dependency container; the inference client is
              retrieved from deps[Api.inference] and stored as
              `self.inference_api`.
        """
        self.config = config
        self.model_prompt_template = Template(f"{self.config.model_prompt}")
        self.inference_api = deps[Api.inference]

    async def initialize(self) -> None:
        """
        Perform provider initialization tasks.

        This implementation performs no actions.
        """
        pass

    async def shutdown(self) -> None:
        """
        Perform provider shutdown and release any held resources.

        This implementation performs no actions (no-op).
        """
        pass

    async def run_moderation(self, request: RunModerationRequest) -> ModerationObject:
        """Run moderation on input text to check if it's a valid question.

        Check whether one or more input texts are valid questions and produce a moderation result.

        Parameters:
            - request (RunModerationRequest): Contains `input` (a string or
              list of strings to check) and an optional `model` override.

        Returns:
            ModerationObject: Moderation result with a generated `id`, the
            selected `model`, and `results` entries for each input indicating
            question validity and related metadata.
        """
        inputs = request.input if isinstance(request.input, list) else [request.input]
        results = []

        impl = QuestionValidityRunner(
            model_id=self.config.model_id,
            model_prompt_template=self.model_prompt_template,
            invalid_question_response=self.config.invalid_question_response,
            inference_api=self.inference_api,
        )

        for text in inputs:
            run_response = await impl.run(
                OpenAIUserMessageParam(role="user", content=text)
            )
            results.append(self._get_moderation_object_results(run_response))

        return ModerationObject(
            id=f"modr-{uuid.uuid4()}",
            model=request.model or self.config.model_id,
            results=results,
        )

    def _get_moderation_object_results(
        self, run_shield_response: RunShieldResponse
    ) -> ModerationObjectResults:
        """Convert RunShieldResponse to ModerationObjectResults.

        Map a RunShieldResponse into a ModerationObjectResults that represents
        the question validity decision.

        Parameters:
            - run_shield_response (RunShieldResponse): Shield decision produced
              by QuestionValidityRunner.

        Returns:
            ModerationObjectResults: If `run_shield_response.violation` is
            None, returns an object with `flagged=False`, empty categories, and
            `"question_validity"` score 0.0. If a violation is present, returns
            `flagged=True`, `categories={"question_validity": True}`,
            `category_scores={"question_validity": 1.0}`,
            `category_applied_input_types={"question_validity": ["text"]}`,
            `user_message` set from the violation, and `metadata` containing
            the violation level value under `"violation_level"`.
        """
        if run_shield_response.violation is None:
            return ModerationObjectResults(
                flagged=False,
                categories={},
                category_scores={"question_validity": 0.0},
                category_applied_input_types={},
                user_message=None,
                metadata={},
            )
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
        request: RunShieldRequest,
    ) -> RunShieldResponse:
        # Take last user message (OpenAIUserMessageParam)
        """
        Run the question-validity shield against the most recent user message in the message list.

        This extracts the last message that represents a user (supports
        multiple message param types), and returns the shield's decision for
        that message. If no user message is present, returns a response
        indicating no violation.

        Parameters:
            - request (RunShieldRequest): Request containing shield_id and
              messages; the last message with role "user" or a user message
              type will be evaluated.

        Returns:
            RunShieldResponse: `violation=None` if no user message was found;
            otherwise the shield's decision for the evaluated user message.
        """
        messages = request.messages
        user_messages = [
            m
            for m in messages
            if isinstance(m, OpenAIUserMessageParam)
            or (hasattr(m, "role") and m.role == "user")
        ]
        if not user_messages:
            return RunShieldResponse(violation=None)
        last_msg = user_messages[-1]
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        message = OpenAIUserMessageParam(role="user", content=content)
        log.debug(f"Shield message: {message.content}")

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
        """
        Initialize a QuestionValidityRunner with model.

        Parameters:
            - model_id (str): Identifier of the model to use for shield completions.
            - model_prompt_template (Template): Template used to render the
              model prompt; expects substitutions for `allowed`, `rejected`,
              and `message`.
            - invalid_question_response (str): Message returned to the user
              when the model indicates a question is invalid.
            - inference_api (Inference): Inference client used to call the
              model for producing shield decisions.
        """
        self.model_id = model_id
        self.model_prompt_template = model_prompt_template
        self.invalid_question_response = invalid_question_response
        self.inference_api = inference_api

    def build_text_shield_input(
        self, message: OpenAIUserMessageParam
    ) -> OpenAIUserMessageParam:
        """
        Create a message whose content is the shield prompt generated from the provided message.

        Parameters:
            message (OpenAIUserMessageParam): The original user message to base the shield prompt on.

        Returns:
            OpenAIUserMessageParam: A new message containing the generated shield prompt as its content.
        """
        return OpenAIUserMessageParam(role="user", content=self.build_prompt(message))

    def build_prompt(self, message: OpenAIUserMessageParam) -> str:
        """
        Build the shield prompt.

        Build the shield prompt by substituting the template variables with
        the message content and subject tokens.

        Parameters:
            - message (OpenAIUserMessageParam): The user message whose content
              will be inserted into the template as `message`.

        Returns:
            str: The prompt string with `allowed`, `rejected`, and `message` values substituted.
        """
        prompt = self.model_prompt_template.substitute(
            allowed=SUBJECT_ALLOWED,
            rejected=SUBJECT_REJECTED,
            message=message.content,
        )
        log.debug(f"Shield prompt: {prompt}")
        return prompt

    def get_shield_response(self, response: str) -> RunShieldResponse:
        """
        Retrieve the shield response.

        Interpret the model's raw response and map it to a RunShieldResponse
        indicating acceptance or a violation.

        Parameters:
            - response (str): Raw text returned by the model to interpret.

        Returns:
            RunShieldResponse: `violation=None` if `response` equals
            `SUBJECT_ALLOWED`; otherwise a `RunShieldResponse` containing a
            `SafetyViolation` with `violation_level=ViolationLevel.ERROR` and
            `user_message` set to the configured invalid-question response.
        """
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

    async def run(self, message: OpenAIUserMessageParam) -> RunShieldResponse:
        """
        Run the question-validity shield.

        Run the question-validity shield for a single user message by sending
        the built prompt to the configured inference API and interpreting the
        model's response.

        The model's stripped output is interpreted literally: if it equals
        "ALLOWED", a `RunShieldResponse` with `violation=None` is returned; for
        any other value a `RunShieldResponse` containing a `SafetyViolation`
        (with `ViolationLevel.ERROR` and the configured invalid-question user
        message) is returned.

        Returns:
            RunShieldResponse: Shield decision derived from the model output.
        """
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
