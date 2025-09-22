"""
Common step definitions for llama-stack question validity tests.
"""

import time
import requests
from behave import given


class LlamaStackClient:
    """Client for interacting with the llama-stack API."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )
        self.last_response = None
        self.last_response_data = None

    def wait_for_service(self, timeout: int = 60) -> bool:
        """Wait for the llama-stack service to be ready."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{self.base_url}/v1/health")
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        return False

    def run_shield(self, shield_id: str, message: str):
        """Run a safety shield on a message."""
        payload = {
            "shield_id": shield_id,
            "messages": [{"role": "user", "content": message}],
        }

        self.last_response = self.session.post(
            f"{self.base_url}/v1/safety/run-shield", json=payload
        )
        if self.last_response.status_code == 200:
            self.last_response_data = self.last_response.json()
        return self.last_response


# Background steps
@given('the llama-stack is running on "{base_url}"')
def step_given_llama_stack_running(context, base_url):
    """Ensure the llama-stack is running and accessible."""
    context.client = LlamaStackClient(base_url)

    # Wait for service to be ready
    if not context.client.wait_for_service(timeout=30):
        raise ConnectionError(f"llama-stack is not accessible at {base_url}")

    context.base_url = base_url
