# lightspeed-redaction

Lightspeed redaction safety provider for llama-stack.

## Installation

```bash
pip install lightspeed-redaction
```

## Usage

This provider is designed to work with llama-stack.

### Configuration

Add the provider configuration to your llama-stack configuration:

```yaml
# Example configuration
providers:
  lightspeed-redaction:
    enabled: true
    # Add your redaction configuration here
```

### Integration

The provider will be automatically loaded by llama-stack when properly configured.

## Development

### Local Development

```bash
# Clone the repository
git clone <repository-url>
cd lightspeed-providers/providers/safety/lightspeed-redaction

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