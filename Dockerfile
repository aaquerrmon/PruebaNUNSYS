# Imagen única para la POC: API FastAPI, UI Chainlit, API de producción,
# bootstrap (seed + RAG) y descarga de modelos.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PIP_NO_CACHE_DIR=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Instalación de dependencias en capa separada (mejor caché de Docker)
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-install-project

COPY src/ src/
COPY scripts/ scripts/
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src:/app

EXPOSE 8080 8585 8090

CMD ["uvicorn", "agentic_poc.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
