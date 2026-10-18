"""Proveedores de modelo locales (vLLM) para LangChain.

Única puerta de entrada a la inferencia: cualquier componente que necesite un
LLM o embeddings pasa por aquí, y todas las factorías validan antes que los
endpoints son locales (`Settings.assert_local_inference`). No existe ningún
camino de fallback a proveedores remotos.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_openai import ChatOpenAI

from agentic_poc.config import Settings, get_settings
from agentic_poc.services.embeddings import LocalE5Embeddings

logger = logging.getLogger(__name__)


def create_chat_model(
    settings: Settings | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
) -> ChatOpenAI:
    """ChatOpenAI apuntando al servidor vLLM local.

    El "pensamiento" de Qwen3 se controla con `chat_template_kwargs`; por
    defecto se desactiva para reducir latencia y evitar trazas internas
    irrelevantes en la respuesta final.
    """
    cfg = settings or get_settings()
    cfg.assert_local_inference()
    extra_body: dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": cfg.llm_enable_thinking}
    }
    # `max_completion_tokens` es el nombre de campo que aceptan sin aviso tanto
    # langchain-openai como el servidor vLLM (el alias canónico de OpenAI);
    # pasar `max_tokens` por model_kwargs provoca un UserWarning en cada llamada.
    return ChatOpenAI(
        model=cfg.llm_model,
        base_url=cfg.llm_base_url,
        api_key=cfg.llm_api_key,  # type: ignore[arg-type]
        temperature=cfg.llm_temperature if temperature is None else temperature,
        max_completion_tokens=cfg.llm_max_tokens if max_tokens is None else max_tokens,
        timeout=cfg.llm_timeout_seconds,
        max_retries=cfg.llm_max_retries,
        streaming=streaming,
        extra_body=extra_body,
    )


def create_embeddings(settings: Settings | None = None) -> LocalE5Embeddings:
    """Adaptador de embeddings e5 apuntando al servidor vLLM local."""
    cfg = settings or get_settings()
    cfg.assert_local_inference()
    return LocalE5Embeddings(
        cfg.embeddings_base_url,
        model=cfg.embeddings_model,
        api_key=cfg.embeddings_api_key,
        dimensions=cfg.embeddings_dimensions,
        timeout_seconds=cfg.embeddings_timeout_seconds,
        batch_size=cfg.embeddings_batch_size,
        passage_prefix=cfg.embeddings_passage_prefix,
        query_prefix=cfg.embeddings_query_prefix,
    )


def probe_endpoint(base_url: str, *, timeout_seconds: float = 5.0) -> dict[str, Any]:
    """Comprueba que un servidor OpenAI-compatible responde (para healthchecks)."""
    url = base_url.rstrip("/") + "/models"
    response = httpx.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Respuesta inesperada de {url}")
    return payload
