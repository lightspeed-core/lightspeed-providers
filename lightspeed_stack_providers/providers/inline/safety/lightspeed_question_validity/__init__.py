from typing import Any

from lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.config import (
    QuestionValidityShieldConfig,
)


async def get_provider_impl(
    config: QuestionValidityShieldConfig,
    deps: dict[str, Any],
):
    """
    Load and initialize a QuestionValidityShield implementation for the provided configuration.

    Parameters:
        config (QuestionValidityShieldConfig): Configuration for the question validity shield.
        deps (dict[str, Any]): Dependency mapping passed to the implementation constructor.

    Returns:
        QuestionValidityShieldImpl: An initialized QuestionValidityShield implementation instance.

    Raises:
        AssertionError: If `config` is not an instance of `QuestionValidityShieldConfig`.
    """
    from lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.safety import (
        QuestionValidityShieldImpl,
    )

    assert isinstance(
        config, QuestionValidityShieldConfig
    ), f"Unexpected config type: {type(config)}"

    impl = QuestionValidityShieldImpl(config, deps)
    await impl.initialize()
    return impl
