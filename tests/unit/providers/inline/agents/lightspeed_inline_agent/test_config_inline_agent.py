import pytest
from pydantic import ValidationError
import os

from lightspeed_stack_providers.providers.inline.agents.lightspeed_inline_agent.config import (
    ToolsFilter,
    LightspeedAgentsImplConfig,
    DEFAULT_SYSTEM_PROMPT,
)


def test_tools_filter_defaults():
    """Test that the ToolsFilter model can be instantiated with default values."""
    tools_filter = ToolsFilter()
    assert tools_filter.model_id is None
    assert tools_filter.enabled is True
    assert tools_filter.min_tools == 10
    assert tools_filter.system_prompt_path is None
    assert tools_filter.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert tools_filter.always_include_tools == []


def test_tools_filter_with_valid_system_prompt_path(tmp_path):
    """Test that the ToolsFilter model correctly loads the system prompt from a file."""
    prompt_content = "This is a test prompt."
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text(prompt_content)

    tools_filter = ToolsFilter(system_prompt_path=prompt_file)
    assert tools_filter.system_prompt == prompt_content


def test_tools_filter_with_nonexistent_system_prompt_path():
    """Test that a nonexistent system_prompt_path raises a ValueError."""
    with pytest.raises(ValidationError):
        ToolsFilter(system_prompt_path="/path/to/nonexistent/file")


def test_tools_filter_with_directory_as_system_prompt_path(tmp_path):
    """Test that a directory as system_prompt_path raises a ValueError."""
    with pytest.raises(ValidationError):
        ToolsFilter(system_prompt_path=tmp_path)


def test_tools_filter_with_unreadable_system_prompt_path(tmp_path):
    """Test that an unreadable system_prompt_path raises a ValueError."""
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("test")
    os.chmod(prompt_file, 0o000)  # Remove all permissions

    with pytest.raises(ValidationError):
        ToolsFilter(system_prompt_path=prompt_file)


def test_lightspeed_agents_impl_config_defaults():
    """Test that the LightspeedAgentsImplConfig model can be instantiated with default values."""
    config = LightspeedAgentsImplConfig()
    assert isinstance(config.tools_filter, ToolsFilter)


def test_lightspeed_agents_impl_config_sample_run_config():
    """Test the sample_run_config class method."""
    config = LightspeedAgentsImplConfig.sample_run_config("/fake/dir")
    assert "tools_filter" in config
    assert isinstance(config["tools_filter"], ToolsFilter)
