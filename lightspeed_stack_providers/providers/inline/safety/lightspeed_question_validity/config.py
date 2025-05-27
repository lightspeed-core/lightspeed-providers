from typing import Any

from pydantic import BaseModel, Field

DEFAULT_MODEL_PROMPT = """
Instructions:
- You are a question classifying tool
- You are an expert in kubernetes and openshift
- Your job is to determine where or a user's question is related to kubernetes and/or openshift technologies and to provide a one-word response.
- If a question appears to be related to kubernetes or openshift technologies, answer with the word ${allowed}, otherwise answer with the word ${rejected}.
- Do not explain your answer, just provide the one-word response. Do not give any other response.


Example Question:
Why is the sky blue?
Example Response:
${rejected}

Example Question:
Why is the grass green?
Example Response:
${rejected}

Example Question:
Why is sand yellow?
Example Response:
${rejected}

Example Question:
Can you help configure my cluster to automatically scale?
Example Response:
${allowed}

Question:
${message}
Response:
"""

DEFAULT_INVALID_QUESTION_RESPONSE = """
Hi, I'm the OpenShift Lightspeed assistant, I can help you with questions about OpenShift, 
please ask me a question related to OpenShift.
"""


class QuestionValidityShieldConfig(BaseModel):
    model_id: str | None = Field(
        default=None,
        description="The model_id to use for the guard",
    )
    model_prompt: str | None = Field(
        default=DEFAULT_MODEL_PROMPT,
        description="The default prompt sent to the LLM used to validate the Users' question.",
    )
    invalid_question_response: str | None = Field(
        default=DEFAULT_INVALID_QUESTION_RESPONSE,
        description="The default response when the Users' question is determined to be invalid.",
    )

    @classmethod
    def sample_run_config(
        cls,
        model_id: str = "${env.INFERENCE_MODEL}",
        model_prompt: str = DEFAULT_MODEL_PROMPT,
        invalid_question_response: str = DEFAULT_INVALID_QUESTION_RESPONSE,
        **kwargs,
    ) -> dict[str, Any]:
        return {
            "model_id": model_id,
            "model_prompt": model_prompt,
            "invalid_question_response": invalid_question_response,
        }
