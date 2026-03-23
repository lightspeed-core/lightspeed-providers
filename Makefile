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

shellcheck: ## Run shellcheck
	wget -qO- "https://github.com/koalaman/shellcheck/releases/download/stable/shellcheck-stable.linux.x86_64.tar.xz" | tar -xJv \
	shellcheck --version
	shellcheck -- */*.sh

black:	## Check source code using Black code formatter
	uv run black --check .

pylint:	## Check source code using Pylint static code analyser
	uv run pylint lightspeed_stack_providers tests/

pyright:	## Check source code using Pyright static type checker
	uv run pyright lightspeed_stack_providers tests/

docstyle:	## Check the docstring style using Docstyle checker
	uv run pydocstyle -v lightspeed_stack_providers/ tests/

ruff:	## Check source code using Ruff linter
	uv run ruff check . --per-file-ignores=tests/*:S101 --per-file-ignores=scripts/*:S101

verify:	## Run all linters
	$(MAKE) black
	$(MAKE) pylint
	$(MAKE) pyright
	$(MAKE) ruff
	$(MAKE) docstyle
	$(MAKE) check-types

doc:	## Generate documentation for developers
	scripts/gen_doc.py

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
