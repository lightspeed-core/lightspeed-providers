# lightspeed-question-validity

Lightspeed question validity safety provider for llama-stack.

## Installation

```bash
pip install lightspeed-question-validity
```

## Usage

This provider is designed to work with llama-stack.

### Configuration

Add the provider configuration to your llama-stack configuration:

```yaml
# Example configuration
providers:
  lightspeed-question-validity:
    enabled: true
    model_id: "gpt-3.5-turbo"
    model_prompt: "Validate if this question is appropriate..."
    invalid_question_response: "This question is not appropriate."
```

### Integration

The provider will be automatically loaded by llama-stack when properly configured.

## Development

### Local Development

```bash
# Clone the repository
git clone <repository-url>
cd lightspeed-providers/providers/safety/lightspeed-question-validity

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