# syntax=docker/dockerfile:1.6
FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so we can leverage Docker's layer
# cache.  Copy only the manifest, install, then copy the source.
COPY pyproject.toml README.md /app/
RUN pip install --upgrade pip && pip install .

COPY ai_assistant_server /app/ai_assistant_server
COPY tools /app/tools

# Default: serve over SSE so the container is reachable from
# other containers (stdio is great for local subprocess use,
# but doesn't survive a container boundary).
ENV AI_ASSISTANT_SERVER_TRANSPORT=sse \
    AI_ASSISTANT_SERVER_HOST=0.0.0.0 \
    AI_ASSISTANT_SERVER_PORT=8765 \
    AI_ASSISTANT_SERVER_TOOLS_DIR=/app/tools

EXPOSE 8765

# Mount your own ./tools at /app/tools to override the samples:
#   docker run -v $(pwd)/tools:/app/tools ai-assistant-server
ENTRYPOINT ["ai-assistant-server"]
