from typing import Any

from lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.config import (
    QuestionValidityShieldConfig,
)


async def get_provider_impl(
    config: QuestionValidityShieldConfig,
    deps: dict[str, Any],
):
    from lightspeed_stack_providers.providers.inline.safety.lightspeed_question_validity.safety import (
        QuestionValidityShieldImpl,
    )

    assert isinstance(
        config, QuestionValidityShieldConfig
    ), f"Unexpected config type: {type(config)}"

    impl = QuestionValidityShieldImpl(config, deps)
    await impl.initialize()
    return impl
