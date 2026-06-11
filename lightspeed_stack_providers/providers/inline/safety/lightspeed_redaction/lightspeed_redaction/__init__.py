"""Answer redactor provider implementation."""

from typing import TYPE_CHECKING, Any

from .config import RedactionShieldConfig

if TYPE_CHECKING:
    from .redaction import RedactionShieldImpl


async def get_provider_impl(
    config: RedactionShieldConfig,
    deps: dict[str, Any],
) -> "RedactionShieldImpl":
    """
    Create and initialize a RedactionShield provider implementation from the given config and deps.

    Parameters:
        config (RedactionShieldConfig): Configuration for the RedactionShield provider.
        deps (dict[str, Any]): Runtime dependencies required by the implementation.

    Returns:
        RedactionShieldImpl: An initialized provider implementation.

    Raises:
        AssertionError: If `config` is not a `RedactionShieldConfig`.
    """
    # pylint: disable=import-outside-toplevel
    from .redaction import (
        RedactionShieldImpl,
    )

    assert isinstance(
        config, RedactionShieldConfig
    ), f"Unexpected config type: {type(config)}"
    impl = RedactionShieldImpl(config, deps)  # pyright: ignore[reportAbstractUsage]
    await impl.initialize()
    return impl
