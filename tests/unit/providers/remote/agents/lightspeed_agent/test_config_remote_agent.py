from lightspeed_stack_providers.providers.remote.agents.lightspeed_agent.config import (
    LightspeedAgentConfig,
)


def test_lightspeed_agent_config_defaults():
    """Test that the LightspeedAgentConfig model can be instantiated with default values."""
    config = LightspeedAgentConfig()
    assert config.api_key is None
    assert config.api_url is None


def test_lightspeed_agent_config_custom_values():
    """Test that the LightspeedAgentConfig model correctly assigns custom values."""
    api_key = "test_key"
    api_url = "http://test.com"
    config = LightspeedAgentConfig(api_key=api_key, api_url=api_url)
    assert config.api_key == api_key
    assert config.api_url == api_url


def test_sample_run_config():
    """Test that the sample_run_config class method returns the expected dictionary."""
    expected_config = {
        "api_key": "${env.LIGHTSPEED_AGENT_API_KEY:}",
        "api_url": "${env.LIGHTSPEED_AGENT_API_URL:http://localhost:8080/agent}",
    }
    assert LightspeedAgentConfig.sample_run_config("/fake/dir") == expected_config
