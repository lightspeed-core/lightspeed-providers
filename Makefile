.PHONY: help install test test-unit test-e2e lint format clean build setup-dev

# Install dependencies
install:	## Install all dependencies using uv
	uv pip install -e .[dev,test]

test: test-unit	## Run the unit tests

test-unit:	## Run the unit tests
	@echo "Running unit tests..."
	uv run pytest tests/unit/ --cov=lightspeed_stack_providers --cov-report=term-missing

# Run question validity tests (requires running stack)
test-e2e:	## Run end to end tests for the service
	@echo "Running BDD tests with Behave..."
	PYTHONDONTWRITEBYTECODE=1 uv run behave --tags=@question_validity -D dump_errors=true tests/features

# Run solr_vector_io tests (requires running RHOKP as RAG index)
test-solr:	## Run Solr vector_io tests
	@echo "Running solr_vector_io tests..."
	uv run lightspeed_stack_providers/providers/remote/solr_vector_io/solr_vector_io/tests.py

check-types:	## Checks type hints in sources
	uv run mypy --explicit-package-bases --disallow-untyped-calls --disallow-untyped-defs --disallow-incomplete-defs --ignore-missing-imports --disable-error-code attr-defined lightspeed_stack_providers/ tests/

security-check: ## Check the project for security issues
	uv run bandit -c pyproject.toml -r lightspeed_stack_providers/ tests

lint:	## Lint source code, including tests
	uv run ruff check .
	uv run black --check .

format: ## Format the code into unified format
	uv run ruff format .
	uv run black .

clean:	## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete


build: clean	## Build package
	uv run python -m build

help: ## Show this help screen
	@echo 'Usage: make <OPTIONS> ... <TARGETS>'
	@echo ''
	@echo 'Available targets are:'
	@echo ''
	@grep -E '^[ a-zA-Z0-9_./-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-33s\033[0m %s\n", $$1, $$2}'
	@echo ''
