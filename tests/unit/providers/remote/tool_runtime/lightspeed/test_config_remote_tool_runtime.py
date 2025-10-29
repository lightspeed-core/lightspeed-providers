from lightspeed_stack_providers.providers.remote.tool_runtime.lightspeed.config import (
    LightspeedToolConfig,
)


def test_lightspeed_tool_config_defaults():
    """Test that the LightspeedToolConfig model can be instantiated with a default value."""
    config = LightspeedToolConfig()
    assert config.api_key is None


def test_lightspeed_tool_config_custom_values():
    """Test that the LightspeedToolConfig model correctly assigns a custom value."""
    api_key = "test_key"
    config = LightspeedToolConfig(api_key=api_key)
    assert config.api_key == api_key


def test_sample_run_config():
    """Test that the sample_run_config class method returns the expected dictionary."""
    expected_config = {
        "api_key": "${env.LIGHTSPEED_API_KEY:}",
    }
    assert LightspeedToolConfig.sample_run_config("/fake/dir") == expected_config
