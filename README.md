# Lightspeed Llama-Stack Providers

A collection of llama-stack external providers organized by type and published as individual PyPI packages.

## 📁 Repository Structure

```
providers/
├── agents/
│   ├── lightspeed-inline-agent/     # Inline agent with tool filtering
│   └── lightspeed-agent/            # Remote agent service
├── safety/
│   ├── lightspeed-question-validity/ # Content validation
│   └── lightspeed-redaction/        # Content redaction
├── tool-runtime/
│   └── lightspeed-tool-runtime/     # MCP-based tool execution
├── inference/                       # Ready for future providers
├── vector-io/                       # Ready for future providers
├── vector-dbs/                      # Ready for future providers
└── tool-groups/                     # Ready for future providers
```

## 🚀 Quick Start

### Install Individual Packages

```bash
# Agent providers
pip install lightspeed-inline-agent
pip install lightspeed-agent

# Safety providers
pip install lightspeed-question-validity
pip install lightspeed-redaction

# Tool runtime provider
pip install lightspeed-tool-runtime
```

### Build All Packages

```bash
# Build all packages
make build-all

# Build specific package
make build-package PACKAGE=lightspeed-inline-agent
```

## 🔧 Development

### Local Development

```bash
# Clone the repository
git clone <repository-url>
cd lightspeed-providers

# Install development dependencies
make install-deps

# Build all packages
make build-all

# Test built packages
make test-build
```

### Adding New Providers

```bash
# Generate template
make generate-template PROVIDER_NAME=my-provider PROVIDER_TYPE=inline-agent

# Build and test
make build-package PACKAGE=my-provider
make test-build
```

## 📦 Build and Release Automation

### Automated Release Process

1. **Update Version**: Edit version in `pyproject.toml`
2. **Create Tag**: `git tag v0.1.13 && git push origin v0.1.13`
3. **GitHub Actions**: Automatically builds, tests, and publishes to PyPI

### Manual Release Process

```bash
# Build all packages
make build-all

# Test packages
make test-build

# Publish to PyPI
make publish-all

# Create git tag
git tag v0.1.13
git push origin v0.1.13
```

## 🔄 Version Management

### Lockstep Versioning

All providers use **lockstep versioning** - all packages share the same version from the main `pyproject.toml`:

```toml
# pyproject.toml
version = "0.1.12"  # Single source of truth
```

### Version Increment Process

1. **Update Version**: Edit version in `pyproject.toml`
2. **Build**: `make build-all` (automatically uses new version)
3. **Release**: `git tag v0.1.13 && git push origin v0.1.13`

## 🏗️ Build System

### Makefile Targets

```bash
# Build targets
make build-all                    # Build all packages
make build-package PACKAGE=name   # Build specific package

# Test targets
make test-build                   # Test all built packages
make test-package PACKAGE=name    # Test specific package

# Publish targets
make publish-all                  # Publish all packages to PyPI
make publish-package PACKAGE=name # Publish specific package

# Development targets
make install-deps                 # Install development dependencies
make clean                        # Clean build artifacts
make generate-template            # Generate new provider template
```

## 🔍 Provider Types

### Inline vs Remote Providers

**Inline Providers** (run in-process):
- `lightspeed-inline-agent` - Complex orchestration with multiple APIs
- `lightspeed-question-validity` - Real-time content validation
- `lightspeed-redaction` - Real-time content redaction

**Remote Providers** (external services):
- `lightspeed-agent` - External agent service integration
- `lightspeed-tool-runtime` - MCP-based tool execution

## 📋 Available Packages

| Package | Type | Description |
|---------|------|-------------|
| `lightspeed-inline-agent` | Inline Agent | Agent orchestration with tool filtering |
| `lightspeed-agent` | Remote Agent | Remote agent service integration |
| `lightspeed-question-validity` | Inline Safety | Content validation and filtering |
| `lightspeed-redaction` | Inline Safety | Content redaction and safety |
| `lightspeed-tool-runtime` | Remote Tool Runtime | MCP-based tool execution |

## 🤝 Contributing

### Provider Implementation Guidelines

**Inline Providers** use `get_provider_impl()` and run in-process:
```yaml
api: Api.agents
provider_type: inline::provider_name
config_class: package_name.config.ConfigClass
module: package_name
```

**Remote Providers** use `get_adapter_impl()` and run as external services:
```yaml
api: Api.agents
adapter:
  adapter_type: provider_name
  config_class: package_name.config.ConfigClass
  module: package_name
```

## 📄 License

MIT License - see LICENSE file for details.
