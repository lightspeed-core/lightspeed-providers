# Lightspeed inline agent

This agent can be used as a simple replacement of the default llama-stack agent.

lightspeed inline agent allow to filter tools using inference call, where a top 10 tools is preferable, 
instead of for example 300 (without filtering an error may happen), 
the inference should analyse relying on the user prompt.

This agent does not change the behavior of the default agent but inherit from it and act only tools 
filtering before calling _run_turn function. 

example llama-stack run config:

```yaml
...
agents:
  - provider_id: lightspeed_inline_agent
    provider_type: inline::lightspeed_inline_agent
    config:
      persistence:
        agent_state:
          namespace: lightspeed_agents
          backend: kv_default
        responses:
          table_name: lightspeed_responses
          backend: sql_default
      tools_filter:
        # Optional: Whether to enable tools filtering, default value is true
        enabled: true
        # Optional: The model to use for filtering, the default value is the inference model used
        model_id: ${env.INFERENCE_MODEL_FILTER:=}
        # Optional: From how much tools we start filtering, default value is 10
        min_tools: 10
        # Optional: the file path of the system prompt, default value is None
        system_prompt_path: ${env.FILTER_SYSTEM_PROMPT_PATH:=}
        # Optional: the system prompt if not in a file,
        # when system_prompt_path is defined system_prompt will be the content of the indicated file
        # when system_prompt is empty, the default filtering system prompt is used
        system_prompt: ${env.FILTER_SYSTEM_PROMPT:=}
        # Optional: tools to always include, for example rag tools, to not be filtered out,
        # the default is an empty list
        always_include_tools:
          - knowledge_search
      # Optional: Override temperature for this agent (default: None)
      chatbot_temperature_override: 1.0
...
external_providers_dir: ~/.llama/distributions/ollama/external_providers/
```

The tools that has been called also will be included in subsequent calls, as llm may fire events to call them again.

The /external_providers/ directory correspond to resources/external_providers/ directory of this project.
