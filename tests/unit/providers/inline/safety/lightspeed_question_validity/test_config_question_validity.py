from lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.config import (
    QuestionValidityShieldConfig,
    DEFAULT_MODEL_PROMPT,
    DEFAULT_INVALID_QUESTION_RESPONSE,
)


def test_question_validity_shield_config_defaults():
    """
    Test that the QuestionValidityShieldConfig model can be instantiated with default values.
    """
    config = QuestionValidityShieldConfig()
    assert config.model_id is None
    assert config.model_prompt == DEFAULT_MODEL_PROMPT
    assert config.invalid_question_response == DEFAULT_INVALID_QUESTION_RESPONSE


def test_question_validity_shield_config_custom_values():
    """
    Test that the QuestionValidityShieldConfig model correctly assigns custom values.
    """
    custom_model_id = "test-model"
    custom_model_prompt = "test-prompt"
    custom_invalid_question_response = "test-response"
    config = QuestionValidityShieldConfig(
        model_id=custom_model_id,
        model_prompt=custom_model_prompt,
        invalid_question_response=custom_invalid_question_response,
    )
    assert config.model_id == custom_model_id
    assert config.model_prompt == custom_model_prompt
    assert config.invalid_question_response == custom_invalid_question_response


def test_sample_run_config():
    """
    Test that the sample_run_config class method returns the expected dictionary.
    """
    expected_config = {
        "model_id": "${env.INFERENCE_MODEL}",
        "model_prompt": DEFAULT_MODEL_PROMPT,
        "invalid_question_response": DEFAULT_INVALID_QUESTION_RESPONSE,
    }
    assert QuestionValidityShieldConfig.sample_run_config() == expected_config


def test_sample_run_config_with_custom_values():
    """
    Test that the sample_run_config class method returns the expected dictionary with custom values.
    """
    custom_model_id = "custom-model"
    custom_model_prompt = "custom-prompt"
    custom_invalid_question_response = "custom-response"
    expected_config = {
        "model_id": custom_model_id,
        "model_prompt": custom_model_prompt,
        "invalid_question_response": custom_invalid_question_response,
    }
    assert (
        QuestionValidityShieldConfig.sample_run_config(
            model_id=custom_model_id,
            model_prompt=custom_model_prompt,
            invalid_question_response=custom_invalid_question_response,
        )
        == expected_config
    )
