# Makefile for building and publishing Lightspeed provider packages

.PHONY: help build-all build-package publish-all publish-package clean test install-deps

# Default target
help:
	@echo "Available targets:"
	@echo "  build-all        - Build all provider packages"
	@echo "  build-package    - Build a specific package (use PACKAGE=name)"
	@echo "  publish-all      - Build and publish all packages to PyPI"
	@echo "  publish-package  - Build and publish specific package (use PACKAGE=name)"
	@echo "  clean            - Clean build artifacts"
	@echo "  test             - Run tests"
	@echo "  install-deps     - Install development dependencies"
	@echo "  generate-template - Generate new provider template (use PROVIDER_NAME=name PROVIDER_TYPE=type)"
	@echo ""
	@echo "Available packages:"
	@echo "  lightspeed-inline-agent"
	@echo "  lightspeed-agent"
	@echo "  lightspeed-tool-runtime"
	@echo "  lightspeed-question-validity"
	@echo "  lightspeed-redaction"
	@echo ""
	@echo "Available provider types:"
	@echo "  inline-agent     - Inline agent provider"
	@echo "  remote-agent     - Remote agent provider"
	@echo "  remote-tool-runtime - Remote tool runtime provider"
	@echo "  inline-safety    - Inline safety provider"
	@echo "  inline-inference - Inline inference provider"
	@echo "  remote-inference - Remote inference provider"
	@echo "  inline-vector-io - Inline vector I/O provider"
	@echo "  remote-vector-io - Remote vector I/O provider"
	@echo "  inline-vector-dbs - Inline vector database provider"
	@echo "  remote-vector-dbs - Remote vector database provider"
	@echo "  inline-tool-groups - Inline tool groups provider"
	@echo "  remote-tool-groups - Remote tool groups provider"

# Install development dependencies
install-deps:
	pip install build twine
	pip install pytest

# Build all packages
build-all:
	@echo "Building all provider packages..."
	python scripts/build_reorganized.py

# Build specific package
build-package:
	@if [ -z "$(PACKAGE)" ]; then \
		echo "Error: PACKAGE variable not set. Usage: make build-package PACKAGE=package-name"; \
		exit 1; \
	fi
	@echo "Building package: $(PACKAGE)"
	python scripts/build_reorganized.py --provider $(PACKAGE)

# Publish all packages to PyPI
publish-all: build-all
	@echo "Publishing all packages to PyPI..."
	@for package in providers/*/*; do \
		if [ -d "$$package" ] && [ -d "$$package/dist" ]; then \
			echo "Publishing $$(basename $$package)..."; \
			cd "$$package" && python -m twine upload dist/* && cd -; \
		fi; \
	done

# Publish specific package to PyPI
publish-package: build-package
	@if [ -z "$(PACKAGE)" ]; then \
		echo "Error: PACKAGE variable not set. Usage: make publish-package PACKAGE=package-name"; \
		exit 1; \
	fi
	@echo "Publishing package: $(PACKAGE)"
	@package_dir=$$(find providers -name "$(PACKAGE)" -type d | head -1); \
	if [ -n "$$package_dir" ] && [ -d "$$package_dir/dist" ]; then \
		cd "$$package_dir" && python -m twine upload dist/* && cd -; \
	else \
		echo "Package not built yet. Run 'make build-package PACKAGE=$(PACKAGE)' first."; \
		exit 1; \
	fi

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find providers -name "dist" -type d -exec rm -rf {} + 2>/dev/null || true
	find providers -name "build" -type d -exec rm -rf {} + 2>/dev/null || true
	find providers -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true

# Run tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v

# Version management
version:
	@echo "Current version: $$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")")

# Update version (usage: make bump-version VERSION=1.2.3)
bump-version:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION variable not set. Usage: make bump-version VERSION=1.2.3"; \
		exit 1; \
	fi
	@echo "Updating version to $(VERSION)..."
	sed -i.bak 's/version = ".*"/version = "$(VERSION)"/' pyproject.toml
	rm pyproject.toml.bak
	@echo "Version updated to $(VERSION)"

# Build and test all packages
test-build: build-all
	@echo "Testing built packages..."
	@for package in providers/*/*; do \
		if [ -d "$$package" ] && [ -d "$$package/dist" ]; then \
			echo "Testing $$(basename $$package)..."; \
			cd "$$package" && python -m pip install dist/*.whl --force-reinstall && cd -; \
		fi; \
	done

# Show package info
package-info:
	@if [ -z "$(PACKAGE)" ]; then \
		echo "Error: PACKAGE variable not set. Usage: make package-info PACKAGE=package-name"; \
		exit 1; \
	fi
	@echo "Package info for $(PACKAGE):"
	@package_dir=$$(find providers -name "$(PACKAGE)" -type d | head -1); \
	if [ -n "$$package_dir" ] && [ -f "$$package_dir/pyproject.toml" ]; then \
		echo "pyproject.toml:"; \
		cat "$$package_dir/pyproject.toml"; \
	else \
		echo "Package not found or not built yet. Run 'make build-package PACKAGE=$(PACKAGE)' first."; \
	fi

# Generate new provider template
generate-template:
	@if [ -z "$(PROVIDER_NAME)" ]; then \
		echo "Error: PROVIDER_NAME variable not set. Usage: make generate-template PROVIDER_NAME=my-provider PROVIDER_TYPE=inline-agent"; \
		exit 1; \
	fi
	@if [ -z "$(PROVIDER_TYPE)" ]; then \
		echo "Error: PROVIDER_TYPE variable not set. Usage: make generate-template PROVIDER_NAME=my-provider PROVIDER_TYPE=inline-agent"; \
		exit 1; \
	fi
	@echo "Generating template for $(PROVIDER_NAME) ($(PROVIDER_TYPE))..."
	python scripts/generate_provider_template.py $(PROVIDER_NAME) $(PROVIDER_TYPE) 