"""
Behave environment configuration for Question Validity Shield tests.
"""

import os


def before_all(context):
    """Setup before all tests."""
    # Set default configuration
    context.config.setup_logging()

    # Set base URL from environment or default
    context.base_url = os.getenv("LLAMA_STACK_BASE_URL", "http://localhost:8321")


def after_scenario(context, scenario):
    """Log scenario results."""
    if scenario.status.name == "failed":
        print(f"❌ {scenario.name}")
        if (
            hasattr(context, "client")
            and hasattr(context.client, "last_response")
            and context.client.last_response
        ):
            print(f"   Status: {context.client.last_response.status_code}")
            print(f"   Data: {context.client.last_response_data}")
    else:
        print(f"✅ {scenario.name}")
