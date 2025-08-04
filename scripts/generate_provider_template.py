#!/usr/bin/env python3
"""
Template generator for llama-stack external providers.

This script generates a complete template for a new external provider
that follows the llama-stack provider format and can be published to PyPI.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Provider type templates
PROVIDER_TEMPLATES = {
    "inline-agent": {
        "description": "Inline agent provider for llama-stack",
        "yaml_template": """api: Api.agents
provider_type: inline::{provider_name}
config_class: {package_name}.config.{ConfigClass}
module: {package_name}
adapter:
  adapter_type: {adapter_type}
  pip_packages: []
  config_class: {package_name}.config.{ConfigClass}
  module: {package_name}
api_dependencies: [ inference, safety, vector_io, vector_dbs, tool_runtime, tool_groups ]
optional_api_dependencies: []
""",
    "inline-inference": {
        "description": "Inline inference provider for llama-stack",
        "yaml_template": """api: Api.inference
provider_type: inline::{provider_name}
config_class: {package_name}.config.{ConfigClass}
module: {package_name}
adapter:
  adapter_type: {adapter_type}
  pip_packages: []
  config_class: {package_name}.config.{ConfigClass}
  module: {package_name}
api_dependencies: []
optional_api_dependencies: []
""",
        "init_template": """from typing import Any

from llama_stack.distribution.datatypes import AccessRule, Api

from .config import {ConfigClass}


async def get_provider_impl(
    config: {ConfigClass}, deps: dict[Api, Any], policy: list[AccessRule]
):
    from .inference import {ImplClass}

    impl = {ImplClass}(config, deps, policy)
    await impl.initialize()
    return impl
""",
        "config_template": """from typing import Any, Optional
from pydantic import BaseModel


class {ConfigClass}(BaseModel):
    \"\"\"Configuration for {provider_name} inference provider\"\"\"
    
    # Add your configuration fields here
    model_id: str = "gpt-3.5-turbo"
    
    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {{
            "model_id": "${{env.MODEL_ID:gpt-3.5-turbo}}",
        }}
""",
        "inference_template": """from typing import Any
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="inference")


class {ImplClass}:
    def __init__(self, config, deps, policy):
        self.config = config
        self.deps = deps
        self.policy = policy
        
    async def initialize(self):
        \"\"\"Initialize the inference provider\"\"\"
        # Add your initialization logic here
        pass
        
    async def chat_completion(self, model_id: str, messages: list, **kwargs: Any):
        \"\"\"Handle chat completion requests\"\"\"
        # Add your chat completion logic here
        pass
""",
    },
    "inline-vector-io": {
        "description": "Inline vector I/O provider for llama-stack",
        "yaml_template": """api: Api.vector_io
provider_type: inline::{provider_name}
config_class: {package_name}.config.{ConfigClass}
module: {package_name}
adapter:
  adapter_type: {adapter_type}
  pip_packages: []
  config_class: {package_name}.config.{ConfigClass}
  module: {package_name}
api_dependencies: []
optional_api_dependencies: []
""",
        "init_template": """from typing import Any

from llama_stack.distribution.datatypes import AccessRule, Api

from .config import {ConfigClass}


async def get_provider_impl(
    config: {ConfigClass}, deps: dict[Api, Any], policy: list[AccessRule]
):
    from .vector_io import {ImplClass}

    impl = {ImplClass}(config, deps, policy)
    await impl.initialize()
    return impl
""",
        "config_template": """from typing import Any, Optional
from pydantic import BaseModel


class {ConfigClass}(BaseModel):
    \"\"\"Configuration for {provider_name} vector I/O provider\"\"\"
    
    # Add your configuration fields here
    enabled: bool = True
    
    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {{
            "enabled": True,
        }}
""",
        "vector_io_template": """from typing import Any
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="vector_io")


class {ImplClass}:
    def __init__(self, config, deps, policy):
        self.config = config
        self.deps = deps
        self.policy = policy
        
    async def initialize(self):
        \"\"\"Initialize the vector I/O provider\"\"\"
        # Add your initialization logic here
        pass
        
    async def read_vectors(self, collection: str, **kwargs: Any):
        \"\"\"Read vectors from storage\"\"\"
        # Add your vector reading logic here
        pass
        
    async def write_vectors(self, collection: str, vectors: list, **kwargs: Any):
        \"\"\"Write vectors to storage\"\"\"
        # Add your vector writing logic here
        pass
""",
    },
    "inline-vector-dbs": {
        "description": "Inline vector database provider for llama-stack",
        "yaml_template": """api: Api.vector_dbs
provider_type: inline::{provider_name}
config_class: {package_name}.config.{ConfigClass}
module: {package_name}
adapter:
  adapter_type: {adapter_type}
  pip_packages: []
  config_class: {package_name}.config.{ConfigClass}
  module: {package_name}
api_dependencies: []
optional_api_dependencies: []
""",
        "init_template": """from typing import Any

from llama_stack.distribution.datatypes import AccessRule, Api

from .config import {ConfigClass}


async def get_provider_impl(
    config: {ConfigClass}, deps: dict[Api, Any], policy: list[AccessRule]
):
    from .vector_dbs import {ImplClass}

    impl = {ImplClass}(config, deps, policy)
    await impl.initialize()
    return impl
""",
        "config_template": """from typing import Any, Optional
from pydantic import BaseModel


class {ConfigClass}(BaseModel):
    \"\"\"Configuration for {provider_name} vector database provider\"\"\"
    
    # Add your configuration fields here
    enabled: bool = True
    
    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {{
            "enabled": True,
        }}
""",
        "vector_dbs_template": """from typing import Any
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="vector_dbs")


class {ImplClass}:
    def __init__(self, config, deps, policy):
        self.config = config
        self.deps = deps
        self.policy = policy
        
    async def initialize(self):
        \"\"\"Initialize the vector database provider\"\"\"
        # Add your initialization logic here
        pass
        
    async def create_collection(self, name: str, **kwargs: Any):
        \"\"\"Create a new collection\"\"\"
        # Add your collection creation logic here
        pass
        
    async def delete_collection(self, name: str):
        \"\"\"Delete a collection\"\"\"
        # Add your collection deletion logic here
        pass
""",
    },
    "inline-tool-groups": {
        "description": "Inline tool groups provider for llama-stack",
        "yaml_template": """api: Api.tool_groups
provider_type: inline::{provider_name}
config_class: {package_name}.config.{ConfigClass}
module: {package_name}
adapter:
  adapter_type: {adapter_type}
  pip_packages: []
  config_class: {package_name}.config.{ConfigClass}
  module: {package_name}
api_dependencies: []
optional_api_dependencies: []
""",
        "init_template": """from typing import Any

from llama_stack.distribution.datatypes import AccessRule, Api

from .config import {ConfigClass}


async def get_provider_impl(
    config: {ConfigClass}, deps: dict[Api, Any], policy: list[AccessRule]
):
    from .tool_groups import {ImplClass}

    impl = {ImplClass}(config, deps, policy)
    await impl.initialize()
    return impl
""",
        "config_template": """from typing import Any, Optional
from pydantic import BaseModel


class {ConfigClass}(BaseModel):
    \"\"\"Configuration for {provider_name} tool groups provider\"\"\"
    
    # Add your configuration fields here
    enabled: bool = True
    
    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {{
            "enabled": True,
        }}
""",
        "tool_groups_template": """from typing import Any
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="tool_groups")


class {ImplClass}:
    def __init__(self, config, deps, policy):
        self.config = config
        self.deps = deps
        self.policy = policy
        
    async def initialize(self):
        \"\"\"Initialize the tool groups provider\"\"\"
        # Add your initialization logic here
        pass
        
    async def get_tool_groups(self) -> list[Any]:
        \"\"\"Get available tool groups\"\"\"
        # Add your tool groups logic here
        return []
""",
    },
        "init_template": """from typing import Any

from llama_stack.distribution.datatypes import AccessRule, Api

from .config import {ConfigClass}


async def get_provider_impl(
    config: {ConfigClass}, deps: dict[Api, Any], policy: list[AccessRule]
):
    from .agents import {ImplClass}

    impl = {ImplClass}(
        config,
        deps[Api.inference],
        deps[Api.vector_io],
        deps[Api.safety],
        deps[Api.tool_runtime],
        deps[Api.tool_groups],
        policy,
    )
    await impl.initialize()
    return impl
""",
        "config_template": """from typing import Any, Optional
from pydantic import BaseModel


class {ConfigClass}(BaseModel):
    \"\"\"Configuration for {provider_name} provider\"\"\"
    
    # Add your configuration fields here
    example_field: Optional[str] = None
    
    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {{
            "example_field": "${{env.EXAMPLE_FIELD:}}",
        }}
""",
        "agents_template": """import json
import uuid
from collections.abc import AsyncGenerator

from llama_stack.apis.agents import AgentConfig, AgentTurnCreateRequest, StepType
from llama_stack.distribution.datatypes import AccessRule
from llama_stack.log import get_logger
from llama_stack.providers.inline.agents.meta_reference.agent_instance import ChatAgent
from llama_stack.providers.utils.telemetry import tracing

from llama_stack.apis.inference import (
    Inference,
    UserMessage,
    SamplingParams,
    TopPSamplingStrategy,
)
from llama_stack.apis.safety import Safety
from llama_stack.apis.tools import ToolGroups, ToolRuntime
from llama_stack.apis.vector_io import VectorIO
from llama_stack.providers.utils.kvstore import KVStore

logger = get_logger(name=__name__, category="agents")


class {ImplClass}(ChatAgent):
    def __init__(
        self,
        agent_config: AgentConfig,
        inference_api: Inference,
        vector_io_api: VectorIO,
        safety_api: Safety,
        tool_runtime_api: ToolRuntime,
        tool_groups_api: ToolGroups,
        policy: list[AccessRule],
    ):
        super().__init__(
            agent_config,
            inference_api,
            vector_io_api,
            safety_api,
            tool_runtime_api,
            tool_groups_api,
            policy,
        )
        
    async def initialize(self):
        \"\"\"Initialize the agent\"\"\"
        # Add your initialization logic here
        pass
        
    async def chat(
        self, request: AgentTurnCreateRequest
    ) -> AsyncGenerator[StepType, None]:
        \"\"\"Handle chat requests\"\"\"
        # Add your chat implementation here
        yield StepType.thinking
        # Your implementation here
        yield StepType.complete
""",
    },
    
    "remote-agent": {
        "description": "Remote agent provider for llama-stack",
        "yaml_template": """api: Api.agents
adapter:
  adapter_type: {adapter_type}
  pip_packages: []
  config_class: {package_name}.config.{ConfigClass}
  module: {package_name}
api_dependencies: []
optional_api_dependencies: []
""",
        "init_template": """from .{module_name} import {ImplClass}
from .config import {ConfigClass}


async def get_adapter_impl(config: {ConfigClass}, _deps):
    impl = {ImplClass}(config)
    await impl.initialize()
    return impl
""",
        "config_template": """from typing import Any
from pydantic import BaseModel, Field


class {ConfigClass}(BaseModel):
    \"\"\"Configuration for {provider_name} provider\"\"\"
    
    # Add your configuration fields here
    api_key: str | None = Field(
        default=None,
        description="The API Key",
    )
    api_url: str | None = Field(
        default=None,
        description="The API URL",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {{
            "api_key": "${{env.API_KEY:}}",
            "api_url": "${{env.API_URL:http://localhost:8080}}",
        }}
""",
        "module_template": """from typing import Any
from llama_stack.apis.agents import AgentConfig, AgentTurnCreateRequest, StepType
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="agents")


class {ImplClass}:
    def __init__(self, config):
        self.config = config
        
    async def initialize(self):
        \"\"\"Initialize the remote agent\"\"\"
        # Add your initialization logic here
        pass
        
    async def chat(
        self, request: AgentTurnCreateRequest
    ) -> AsyncGenerator[StepType, None]:
        \"\"\"Handle chat requests\"\"\"
        # Add your chat implementation here
        yield StepType.thinking
        # Your implementation here
        yield StepType.complete
""",
    },
    
    "remote-tool-runtime": {
        "description": "Remote tool runtime provider for llama-stack",
        "yaml_template": """api: Api.tool_runtime
adapter:
  adapter_type: {adapter_type}
  pip_packages: []
  config_class: {package_name}.config.{ConfigClass}
  module: {package_name}
  provider_data_validator: {package_name}.{ImplClass}ProviderDataValidator
api_dependencies: []
optional_api_dependencies: []
""",
    "remote-inference": {
        "description": "Remote inference provider for llama-stack",
        "yaml_template": """api: Api.inference
adapter:
  adapter_type: {adapter_type}
  pip_packages: []
  config_class: {package_name}.config.{ConfigClass}
  module: {package_name}
api_dependencies: []
optional_api_dependencies: []
""",
        "init_template": """from .{module_name} import {ImplClass}
from .config import {ConfigClass}


async def get_adapter_impl(config: {ConfigClass}, _deps):
    impl = {ImplClass}(config)
    await impl.initialize()
    return impl
""",
        "config_template": """from typing import Any
from pydantic import BaseModel, Field


class {ConfigClass}(BaseModel):
    \"\"\"Configuration for {provider_name} provider\"\"\"
    
    # Add your configuration fields here
    api_key: str | None = Field(
        default=None,
        description="The API Key",
    )
    api_url: str | None = Field(
        default=None,
        description="The API URL",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {{
            "api_key": "${{env.API_KEY:}}",
            "api_url": "${{env.API_URL:http://localhost:8080}}",
        }}
""",
        "module_template": """from typing import Any
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="inference")


class {ImplClass}:
    def __init__(self, config):
        self.config = config
        
    async def initialize(self):
        \"\"\"Initialize the inference provider\"\"\"
        # Add your initialization logic here
        pass
        
    async def chat_completion(self, model_id: str, messages: list, **kwargs: Any):
        \"\"\"Handle chat completion requests\"\"\"
        # Add your chat completion logic here
        pass
""",
    },
    "remote-vector-io": {
        "description": "Remote vector I/O provider for llama-stack",
        "yaml_template": """api: Api.vector_io
adapter:
  adapter_type: {adapter_type}
  pip_packages: []
  config_class: {package_name}.config.{ConfigClass}
  module: {package_name}
api_dependencies: []
optional_api_dependencies: []
""",
        "init_template": """from .{module_name} import {ImplClass}
from .config import {ConfigClass}


async def get_adapter_impl(config: {ConfigClass}, _deps):
    impl = {ImplClass}(config)
    await impl.initialize()
    return impl
""",
        "config_template": """from typing import Any
from pydantic import BaseModel, Field


class {ConfigClass}(BaseModel):
    \"\"\"Configuration for {provider_name} provider\"\"\"
    
    # Add your configuration fields here
    api_key: str | None = Field(
        default=None,
        description="The API Key",
    )
    api_url: str | None = Field(
        default=None,
        description="The API URL",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {{
            "api_key": "${{env.API_KEY:}}",
            "api_url": "${{env.API_URL:http://localhost:8080}}",
        }}
""",
        "module_template": """from typing import Any
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="vector_io")


class {ImplClass}:
    def __init__(self, config):
        self.config = config
        
    async def initialize(self):
        \"\"\"Initialize the vector I/O provider\"\"\"
        # Add your initialization logic here
        pass
        
    async def read_vectors(self, collection: str, **kwargs: Any):
        \"\"\"Read vectors from storage\"\"\"
        # Add your vector reading logic here
        pass
        
    async def write_vectors(self, collection: str, vectors: list, **kwargs: Any):
        \"\"\"Write vectors to storage\"\"\"
        # Add your vector writing logic here
        pass
""",
    },
    "remote-vector-dbs": {
        "description": "Remote vector database provider for llama-stack",
        "yaml_template": """api: Api.vector_dbs
adapter:
  adapter_type: {adapter_type}
  pip_packages: []
  config_class: {package_name}.config.{ConfigClass}
  module: {package_name}
api_dependencies: []
optional_api_dependencies: []
""",
        "init_template": """from .{module_name} import {ImplClass}
from .config import {ConfigClass}


async def get_adapter_impl(config: {ConfigClass}, _deps):
    impl = {ImplClass}(config)
    await impl.initialize()
    return impl
""",
        "config_template": """from typing import Any
from pydantic import BaseModel, Field


class {ConfigClass}(BaseModel):
    \"\"\"Configuration for {provider_name} provider\"\"\"
    
    # Add your configuration fields here
    api_key: str | None = Field(
        default=None,
        description="The API Key",
    )
    api_url: str | None = Field(
        default=None,
        description="The API URL",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {{
            "api_key": "${{env.API_KEY:}}",
            "api_url": "${{env.API_URL:http://localhost:8080}}",
        }}
""",
        "module_template": """from typing import Any
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="vector_dbs")


class {ImplClass}:
    def __init__(self, config):
        self.config = config
        
    async def initialize(self):
        \"\"\"Initialize the vector database provider\"\"\"
        # Add your initialization logic here
        pass
        
    async def create_collection(self, name: str, **kwargs: Any):
        \"\"\"Create a new collection\"\"\"
        # Add your collection creation logic here
        pass
        
    async def delete_collection(self, name: str):
        \"\"\"Delete a collection\"\"\"
        # Add your collection deletion logic here
        pass
""",
    },
    "remote-tool-groups": {
        "description": "Remote tool groups provider for llama-stack",
        "yaml_template": """api: Api.tool_groups
adapter:
  adapter_type: {adapter_type}
  pip_packages: []
  config_class: {package_name}.config.{ConfigClass}
  module: {package_name}
api_dependencies: []
optional_api_dependencies: []
""",
        "init_template": """from .{module_name} import {ImplClass}
from .config import {ConfigClass}


async def get_adapter_impl(config: {ConfigClass}, _deps):
    impl = {ImplClass}(config)
    await impl.initialize()
    return impl
""",
        "config_template": """from typing import Any
from pydantic import BaseModel, Field


class {ConfigClass}(BaseModel):
    \"\"\"Configuration for {provider_name} provider\"\"\"
    
    # Add your configuration fields here
    api_key: str | None = Field(
        default=None,
        description="The API Key",
    )
    api_url: str | None = Field(
        default=None,
        description="The API URL",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {{
            "api_key": "${{env.API_KEY:}}",
            "api_url": "${{env.API_URL:http://localhost:8080}}",
        }}
""",
        "module_template": """from typing import Any
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="tool_groups")


class {ImplClass}:
    def __init__(self, config):
        self.config = config
        
    async def initialize(self):
        \"\"\"Initialize the tool groups provider\"\"\"
        # Add your initialization logic here
        pass
        
    async def get_tool_groups(self) -> list[Any]:
        \"\"\"Get available tool groups\"\"\"
        # Add your tool groups logic here
        return []
""",
    },
        "init_template": """from typing import Any, Optional

from pydantic import BaseModel

from llama_stack.distribution.datatypes import Api

from .config import {ConfigClass}


class {ImplClass}ProviderDataValidator(BaseModel):
    \"\"\"Data validator for {provider_name} provider\"\"\"
    # Add your validation fields here
    headers: Optional[dict[str, dict[str, str]]] = {{"*": {{}}}}


async def get_adapter_impl(config: {ConfigClass}, _deps: dict[Api, Any]):
    from .{module_name} import {ImplClass}

    impl = {ImplClass}(config)

    await impl.initialize()
    return impl
""",
        "config_template": """from typing import Any

from pydantic import BaseModel


class {ConfigClass}(BaseModel):
    \"\"\"Configuration for {provider_name} tool runtime\"\"\"

    # Add your configuration fields here
    api_key: str | None = None

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {{
            "api_key": "${{env.API_KEY:}}",
        }}
""",
        "module_template": """from typing import Any
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="tool_runtime")


class {ImplClass}:
    def __init__(self, config):
        self.config = config
        
    async def initialize(self):
        \"\"\"Initialize the tool runtime\"\"\"
        # Add your initialization logic here
        pass
        
    async def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        \"\"\"Execute a tool\"\"\"
        # Add your tool execution logic here
        pass
""",
    },
    
    "inline-safety": {
        "description": "Inline safety provider for llama-stack",
        "yaml_template": """module: {package_name}
config_class: {package_name}.config.{ConfigClass}
pip_packages: []
api_dependencies:
  - inference
optional_api_dependencies: []
""",
        "init_template": """from typing import Any

from .config import {ConfigClass}


async def get_provider_impl(
    config: {ConfigClass},
    deps: dict[str, Any],
):
    from .safety import {ImplClass}

    assert isinstance(
        config, {ConfigClass}
    ), f"Unexpected config type: {{type(config)}}"

    impl = {ImplClass}(config, deps)
    await impl.initialize()
    return impl
""",
        "config_template": """from typing import Any, Optional
from pydantic import BaseModel


class {ConfigClass}(BaseModel):
    \"\"\"Configuration for {provider_name} safety provider\"\"\"
    
    # Add your configuration fields here
    enabled: bool = True
    
    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {{
            "enabled": True,
        }}
""",
        "safety_template": """from typing import Any
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="safety")


class {ImplClass}:
    def __init__(self, config, deps):
        self.config = config
        self.deps = deps
        
    async def initialize(self):
        \"\"\"Initialize the safety provider\"\"\"
        # Add your initialization logic here
        pass
        
    async def validate(self, content: str) -> bool:
        \"\"\"Validate content for safety\"\"\"
        # Add your validation logic here
        return True
""",
    },
}


def generate_provider_template(
    provider_name: str,
    provider_type: str,
    output_dir: str = ".",
    description: Optional[str] = None,
) -> None:
    """Generate a complete provider template"""
    
    if provider_type not in PROVIDER_TEMPLATES:
        print(f"Error: Unknown provider type '{provider_type}'")
        print(f"Available types: {list(PROVIDER_TEMPLATES.keys())}")
        sys.exit(1)
    
    # Convert provider name to package name
    package_name = provider_name.replace("-", "_")
    
    # Generate class names
    ConfigClass = "".join(word.capitalize() for word in provider_name.split("-")) + "Config"
    ImplClass = "".join(word.capitalize() for word in provider_name.split("-")) + "Impl"
    adapter_type = provider_name.replace("-", "_")
    module_name = provider_name.replace("-", "_")
    
    # Get template
    template = PROVIDER_TEMPLATES[provider_type]
    
    # Create output directory
    output_path = Path(output_dir) / provider_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate files
    files_to_create = [
        ("pyproject.toml", generate_pyproject_toml(provider_name, template["description"])),
        ("README.md", generate_readme(provider_name, template["description"])),
        ("src/__init__.py", ""),
        (f"src/{package_name}/__init__.py", template["init_template"].format(
            package_name=package_name,
            ConfigClass=ConfigClass,
            ImplClass=ImplClass,
            adapter_type=adapter_type,
            module_name=module_name,
        )),
        (f"src/{package_name}/config.py", template["config_template"].format(
            ConfigClass=ConfigClass,
            provider_name=provider_name,
        )),
    ]
    
    # Add specific module files based on provider type
    if provider_type == "inline-agent":
        files_to_create.append((f"src/{package_name}/agents.py", template["agents_template"].format(
            ImplClass=ImplClass,
        )))
    elif provider_type == "remote-agent":
        files_to_create.append((f"src/{package_name}/{module_name}.py", template["module_template"].format(
            ImplClass=ImplClass,
        )))
    elif provider_type == "remote-tool-runtime":
        files_to_create.append((f"src/{package_name}/{module_name}.py", template["module_template"].format(
            ImplClass=ImplClass,
        )))
    elif provider_type == "inline-safety":
        files_to_create.append((f"src/{package_name}/safety.py", template["safety_template"].format(
            ImplClass=ImplClass,
        )))
    elif provider_type == "inline-inference":
        files_to_create.append((f"src/{package_name}/inference.py", template["inference_template"].format(
            ImplClass=ImplClass,
        )))
    elif provider_type == "inline-vector-io":
        files_to_create.append((f"src/{package_name}/vector_io.py", template["vector_io_template"].format(
            ImplClass=ImplClass,
        )))
    elif provider_type == "inline-vector-dbs":
        files_to_create.append((f"src/{package_name}/vector_dbs.py", template["vector_dbs_template"].format(
            ImplClass=ImplClass,
        )))
    elif provider_type == "inline-tool-groups":
        files_to_create.append((f"src/{package_name}/tool_groups.py", template["tool_groups_template"].format(
            ImplClass=ImplClass,
        )))
    elif provider_type in ["remote-inference", "remote-vector-io", "remote-vector-dbs", "remote-tool-groups"]:
        files_to_create.append((f"src/{package_name}/{module_name}.py", template["module_template"].format(
            ImplClass=ImplClass,
        )))
    
    # Create files
    for file_path, content in files_to_create:
        full_path = output_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, "w") as f:
            f.write(content)
    
    # Generate YAML config
    yaml_content = template["yaml_template"].format(
        package_name=package_name,
        ConfigClass=ConfigClass,
        adapter_type=adapter_type,
        provider_name=provider_name,
        ImplClass=ImplClass,
    )
    
    config_path = output_path / "config" / f"{provider_name}.yaml"
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, "w") as f:
        f.write(yaml_content)
    
    print(f"✅ Generated provider template: {output_path}")
    print(f"📁 Package structure created")
    print(f"📄 YAML config: {config_path}")
    print(f"📦 Ready for PyPI publishing")


def generate_pyproject_toml(provider_name: str, description: str) -> str:
    """Generate pyproject.toml content"""
    return f"""[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{provider_name}"
version = "0.1.0"
description = "{description}"
readme = "README.md"
license = {{text = "MIT"}}
requires-python = ">=3.12"
authors = [
    {{name = "Your Name", email = "your.email@example.com"}}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "llama-stack==0.2.16",
    "httpx",
    "pydantic>=2.10.6",
]

[project.urls]
Homepage = "https://github.com/your-org/{provider_name}"
Repository = "https://github.com/your-org/{provider_name}"
Issues = "https://github.com/your-org/{provider_name}/issues"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
"{provider_name.replace('-', '_')}" = ["*.yaml", "*.yml"]

[tool.setuptools.data-files]
config = ["config/*.yaml"]
"""


def generate_readme(provider_name: str, description: str) -> str:
    """Generate README.md content"""
    return f"""# {provider_name}

{description}

## Installation

```bash
pip install {provider_name}
```

## Usage

This provider is designed to work with llama-stack.

### Configuration

Add the provider configuration to your llama-stack configuration:

```yaml
# Example configuration
providers:
  {provider_name}:
    enabled: true
    # Add your configuration here
```

### Integration

The provider will be automatically loaded by llama-stack when properly configured.

## Development

### Local Development

```bash
# Clone the repository
git clone <repository-url>
cd {provider_name}

# Install in development mode
pip install -e .

# Run tests
python -m pytest
```

### Building

```bash
# Build the package
python -m build

# Install from wheel
pip install dist/*.whl
```

## License

MIT License - see LICENSE file for details.
"""


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate llama-stack provider template")
    parser.add_argument("provider_name", help="Name of the provider (kebab-case)")
    parser.add_argument("provider_type", help="Type of provider", 
                       choices=list(PROVIDER_TEMPLATES.keys()))
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--description", help="Custom description")
    
    args = parser.parse_args()
    
    # Validate provider name
    if not args.provider_name.replace("-", "").isalnum():
        print("Error: Provider name must contain only letters, numbers, and hyphens")
        sys.exit(1)
    
    # Generate template
    generate_provider_template(
        args.provider_name,
        args.provider_type,
        args.output_dir,
        args.description,
    )


if __name__ == "__main__":
    main() 