"""
Pytest configuration file for Solr vector IO tests.
"""

import pytest
import requests


@pytest.fixture(scope="session", autouse=True)
def setup_kvstore_backend(tmp_path_factory):
    """
    Register a KV store backend for persistence tests.
    This runs before all tests in the session.
    """
    from llama_stack.providers.utils.kvstore.kvstore import register_kvstore_backends
    from llama_stack.core.storage.datatypes import SqliteKVStoreConfig
    import os

    # Create a temporary database file that persists across adapter instances
    tmp_dir = tmp_path_factory.mktemp("kvstore")
    db_path = str(tmp_dir / "test_kvstore.db")

    # Register a test SQLite backend
    backends = {
        "test_sqlite": SqliteKVStoreConfig(
            namespace="test",
            db_path=db_path,  # Use persistent file instead of :memory:
        ),
    }
    register_kvstore_backends(backends)

    yield

    # Cleanup: clear registered backends and remove database file
    register_kvstore_backends({})
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture(scope="session", autouse=True)
def check_solr_running():
    """
    Pre-test check to ensure Solr is running before executing any tests.
    Aborts all tests if Solr is not accessible.
    """
    # Import SOLR_URL from test file to avoid duplication
    from tests import SOLR_URL, COLLECTION_NAME

    solr_test_url = SOLR_URL + "/" + COLLECTION_NAME + "/select"
    print(solr_test_url)

    try:
        response = requests.get(solr_test_url, timeout=5)
        if response.status_code == 200:
            print(f"✓ Solr is running at {solr_test_url}")
        else:
            print(f"\n✗ FAILED: Solr returned status code {response.status_code}")
            print(f"  Expected: 200, Got: {response.status_code}")
            pytest.exit(
                f"Solr is not responding correctly at {solr_test_url} "
                f"(status code: {response.status_code}). "
                f"Please start Solr before running tests.",
                returncode=1,
            )
    except requests.exceptions.ConnectionError:
        print(f"\n✗ FAILED: Could not connect to Solr at {SOLR_URL}")
        print("  Error: Connection refused")
        print("\nPossible solutions:")
        print("  1. Start Solr if it's not running")
        print("  2. Verify Solr is running on the correct host/port")
        print("  3. Check firewall settings")
        pytest.exit(
            f"Cannot connect to Solr at {SOLR_URL}. "
            f"Please start Solr before running tests.",
            returncode=1,
        )
    except requests.exceptions.Timeout:
        print(f"\n✗ FAILED: Timeout connecting to Solr at {SOLR_URL}")
        print("  Request timed out after 5 seconds")
        pytest.exit(
            f"Timeout connecting to Solr at {SOLR_URL}. "
            f"Solr may be starting up or experiencing issues.",
            returncode=1,
        )
    except Exception as e:
        print(f"\n✗ FAILED: Unexpected error checking Solr at {SOLR_URL}")
        print(f"  Error: {type(e).__name__}: {e}")
        pytest.exit(f"Unexpected error checking Solr: {e}", returncode=1)

    yield
