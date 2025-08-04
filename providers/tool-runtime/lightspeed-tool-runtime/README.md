# lightspeed-tool-runtime

Lightspeed tool runtime provider for llama-stack.

## Installation

```bash
pip install lightspeed-tool-runtime
```

## Usage

This provider is designed to work with llama-stack.

### Configuration

Add the provider configuration to your llama-stack configuration:

```yaml
# Example configuration
providers:
  lightspeed-tool-runtime:
    enabled: true
    api_key: "${env.LIGHTSPEED_API_KEY}"
```

### Integration

The provider will be automatically loaded by llama-stack when properly configured.

## Development

### Local Development

```bash
# Clone the repository
git clone <repository-url>
cd lightspeed-providers/providers/tool-runtime/lightspeed-tool-runtime

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