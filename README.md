# Lightspeed Providers

Custom provider implementations for Llama Stack that extend the capabilities of AI applications with specialized safety and content filtering features.

## Overview

This repository contains custom providers for Llama Stack, including:

- **Question Validity Shield**: Ensures queries are related to OpenShift/Ansible topics (It can be Configured for other Platforms).
- **Redaction Shield**: Automatically detects and redacts sensitive information from user messages
- **Additional safety and content filtering providers**

## Building and publishing

Manual procedure, assuming an existing PyPI API token available:

    ## Generate distribution archives to be uploaded into Python registry
    pdm run python -m build
    ## Upload distribution archives into Python registry
    pdm run python -m twine upload --repository ${PYTHON_REGISTRY} dist/*

## Features

###  Redaction Shield
- **Pattern-based redaction**: YAML-configurable regex patterns for flexible content filtering
- **Automatic detection**: Detects credit card numbers, API keys, tokens, passwords, and custom patterns

###  Question Validity Shield
- **Topic validation**: Ensures queries are related to specified topics (OpenShift/Ansible) (It can be configured for other Platforms)
- **LLM-powered classification**: Uses AI to determine query relevance
- **Customizable responses**: Configure custom messages for invalid queries

## Quick Start

### Prerequisites

- Python >= 3.12
- Llama Stack >= 0.2.16
- pydantic >= 2.10.6

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/lightspeed-core/lightspeed-providers.git
   cd lightspeed-providers
   ```

2. **Install dependencies**
   ```bash
   pip install -e .
   ```

3. **Install Llama Stack** (if not already installed)
   ```bash
   pip install llama-stack
   ```

### Local Setup

#### Method 1: Using with Existing Llama Stack

1. **Install the Python package**
   ```bash
   pip install lightspeed_stack_providers
   ```
2. **Configure your run.yaml** (see Configuration section below)

## Configuration 

### 1. Register External Providers

Add to your `run.yaml` file:

```yaml
# External providers configuration
external_providers_dir: ${env.EXTERNAL_PROVIDERS_DIR:/providers.d}

# Changes in the providers

providers:
  safety:
  - provider_id: llama-guard
    provider_type: inline::llama-guard
    config:
      excluded_categories: []

# For Redaction shields

  - provider_id: lightspeed_redaction
    provider_type: inline::lightspeed_redaction
    config:
        case_sensitive: false
        rules:
          - pattern: "(?i)(password|passwd)[\\s:=]+[^\\s]+"
            replacement: "[REDACTED_PASSWORD]"
          
          - pattern: "(?i)(registry|image):\\s*([\\w\\d\\.-]+)(:[\\w\\d\\.-]+)?"
            replacement: "\\1: [REDACTED_IMAGE]"
          
          - pattern: "(?i)(url|endpoint):\\s*https?://[\\w\\.-]+(:\\d+)?(/[\\w\\d\\.-]*)*"
            replacement: "\\1: [REDACTED_URL]"

        
          - pattern: "\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b"
            replacement: "[REDACTED_IP]"
        
          - pattern: "(?i)(api_key|secret)[\\s:=]+[a-zA-Z0-9\\-_]{16,}"
            replacement: "[REDACTED_SECRET]"
          
          - pattern: "(?i)(ssh-rsa|ssh-ed25519)\\s+[A-Za-z0-9+/=]+"
            replacement: "[REDACTED_SSH_KEY]"

# for question validity 

  - provider_id: lightspeed_question_validity
    provider_type: inline::lightspeed_question_validity
    config:
      model_id: ${env.INFERENCE_MODEL}
      model_prompt: |-
        Instructions:
        - You are a question classifying tool
        - You are an expert in ansible
        - Your job is to determine where or a user's question is related to ansible technologies and to provide a one-word response
        - If a question appears to be related to ansible technologies, answer with the word ${allowed}, otherwise answer with the word ${rejected}
        - Do not explain your answer, just provide the one-word response


        Example Question:
        Why is the sky blue?
        Example Response:
        ${rejected}

        Example Question:
        Can you help generate an ansible playbook to install an ansible collection?
        Example Response:
        ${allowed}

        Example Question:
        Can you help write an ansible role to install an ansible collection?
        Example Response:
        ${allowed}

        Question:
        ${message}
        Response:
        invalid_question_response: |-
        Hi, I'm the Ansible Lightspeed Intelligent Assistant, I can help you with questions about Ansible,
        please ask me a question related to Ansible.

# changes in the agents : 

shields:
  - shield_id: lightspeed_question_validity-shield
    provider_id: lightspeed_question_validity
  - shield_id: redaction-shield
    provider_id: lightspeed_redaction
    provider_shield_id: lightspeed-redaction-shield
```
## Usage Examples

### Testing Redaction

```bash
# Test the redaction shield
curl -X POST "http://localhost:8321/v1/safety/run_shield" \
  -H "Content-Type: application/json" \
  -d '{
    "shield_id": "redaction-shield",
    "messages": [
      {
        "role": "user", 
        "content": "My API key is abc123xyz and password is secret456"
      }
    ]
  }'
```
### Adding New Providers

1. **Create provider directory**
   ```bash
   mkdir -p ./providers.d/inline/safety/
   mkdir -p ./providers.d/remote/tool_runtime/
   curl -o ./providers.d/inline/safety/lightspeed_question_validity.yaml https://raw.githubusercontent.com/lightspeed-core/lightspeed-            providers/refs/heads/main/resources/external_providers/inline/safety/lightspeed_question_validity.yaml
   curl -o ./providers.d/inline/safety/lightspeed_question_validity.yaml https://raw.githubusercontent.com/lightspeed-core/lightspeed-            providers/refs/heads/main/resources/external_providers/inline/safety/lightspeed_redaction.yaml
   curl -o ./providers.d/remote/tool_runtime/lightspeed.yaml https://raw.githubusercontent.com/lightspeed-core/lightspeed-providers/refs/heads/main/resources/external_providers/remote/tool_runtime/lightspeed.yaml
   
   ```
3. **Add external provider definition**
   ```yaml
   # resources/external_providers/your_provider.yaml
   module: lightspeed_stack_providers.providers.inline.safety.your_provider
   config_class: lightspeed_stack_providers.providers.inline.safety.your_provider.config.YourProviderConfig
   pip_packages: ["lightspeed_stack_providers"]
   api_dependencies:
     - safety
   ```
## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Documentation

- [Llama Stack Documentation](https://llama-stack.readthedocs.io/)
- [External Providers Guide](https://llama-stack.readthedocs.io/en/latest/providers/external/external-providers-guide.html)
- [Safety Providers](https://llama-stack.readthedocs.io/en/latest/providers/safety/index.html)
- [Building Applications](https://llama-stack.readthedocs.io/en/latest/building_applications/safety.html)
