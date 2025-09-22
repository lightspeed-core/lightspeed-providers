# vim: set filetype=dockerfile
FROM registry.access.redhat.com/ubi9/ubi-minimal

ARG APP_ROOT=/app-root

ENV PATH="$PATH:/root/.local/bin"

WORKDIR ${APP_ROOT}

# Copy project files
COPY run.yaml ./
COPY pyproject.toml ./
COPY uv.lock ./
COPY LICENSE ./
COPY README.md ./
COPY lightspeed_stack_providers/ ./lightspeed_stack_providers/
COPY resources/ ./resources/

# Install system dependencies
RUN microdnf install -y --nodocs --setopt=keepcache=0 --setopt=tsflags=nodocs \
    python3.12 python3.12-devel python3.12-pip git tar gcc gcc-c++ make

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv -h

RUN uv sync --locked --group dev --group test

# Install the lightspeed providers package
RUN uv pip install -e .

# Verify llama command is available
RUN uv run python -c "import llama_stack; print('llama-stack imported successfully')"
RUN uv run llama --help

# Expose port
EXPOSE 8321

# Default command - use llama command directly
CMD ["uv", "run", "llama", "stack", "run", "/app/run.yaml"]