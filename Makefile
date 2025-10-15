.PHONY: help install test test-e2e lint format clean build setup-dev

# Default target
help:
	@echo "Available targets:"
	@echo "  install       - Install dependencies using uv"
	@echo "  test-e2e      - Run question validity tests (requires running stack)"
	@echo "  lint          - Run linting checks"
	@echo "  format        - Format code"
	@echo "  clean         - Clean build artifacts"
	@echo "  build         - Build the package"

# Install dependencies
install:
	uv sync

# Run question validity tests (requires running stack)
test-e2e:
	@echo "Running BDD tests with Behave..."
	PYTHONDONTWRITEBYTECODE=1 uv run behave --tags=@question_validity -D dump_errors=true tests/features

# Lint code
lint:
	uv run ruff check .
	uv run black --check .

# Format code
format:
	uv run ruff format .
	uv run black .

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build package
build: clean
	uv run python -m build

