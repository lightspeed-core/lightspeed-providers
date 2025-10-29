from typing import Any
from .config import RedactionShieldConfig


async def get_provider_impl(
    config: RedactionShieldConfig,
    deps: dict[str, Any],
):
    from .redaction import (
        RedactionShieldImpl,
    )

    assert isinstance(
        config, RedactionShieldConfig
    ), f"Unexpected config type: {type(config)}"
    impl = RedactionShieldImpl(config, deps)
    await impl.initialize()
    return impl
