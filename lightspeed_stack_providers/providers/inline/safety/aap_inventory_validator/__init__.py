from typing import Any

from lightspeed_stack_providers.providers.inline.safety.aap_inventory_validator.config import (
    InventoryValidatorConfig,
)


async def get_provider_impl(
    config: InventoryValidatorConfig,
    deps: dict[str, Any],
):
    from lightspeed_stack_providers.providers.inline.safety.aap_inventory_validator.safety import (
        InventoryValidatorImpl,
    )

    assert isinstance(
        config, InventoryValidatorConfig
    ), f"Unexpected config type: {type(config)}"

    impl = InventoryValidatorImpl(config)
    await impl.initialize()
    return impl
