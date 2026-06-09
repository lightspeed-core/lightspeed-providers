"""Question validity safety provider implementation."""

from typing import TYPE_CHECKING, Any

from lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.config import (
    QuestionValidityShieldConfig,
)

if TYPE_CHECKING:
    from lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.safety import (  # pylint: disable=line-too-long
        QuestionValidityShieldImpl,
    )


async def get_provider_impl(
    config: QuestionValidityShieldConfig,
    deps: dict[str, Any],
) -> "QuestionValidityShieldImpl":
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
    from lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.safety import (  # pylint: disable=line-too-long
        QuestionValidityShieldImpl,
    )

    assert isinstance(
        config, QuestionValidityShieldConfig
    ), f"Unexpected config type: {type(config)}"

    impl = QuestionValidityShieldImpl(config, deps)
    await impl.initialize()
    return impl
