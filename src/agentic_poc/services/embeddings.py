"""Embeddings locales (e5) servidos por vLLM, compatibles con LangChain.

vLLM expone el endpoint OpenAI `/embeddings`, pero no conoce la convencion de
prefijos de los modelos e5 (`passage:` para documentos, `query:` para
consultas), asi que esa responsabilidad es nuestra: el adaptador aplica el
prefijo correcto antes de enviar y valida la dimensionalidad devuelta.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import httpx
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class EmbeddingsServiceError(RuntimeError):
    """Fallo al obtener embeddings (red, HTTP o dimensiones inesperadas)."""


class LocalE5Embeddings(Embeddings):
    """Adaptador de embeddings e5 sobre un servidor vLLM local.

    No hay rutas de fallback: si el servidor local no responde, se propaga
    `EmbeddingsServiceError` (la ausencia de resultados nunca es silenciosa).
    """

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        api_key: str = "local-vllm-key",
        dimensions: int = 384,
        timeout_seconds: float = 30.0,
        batch_size: int = 8,
        passage_prefix: str = "passage:",
        query_prefix: str = "query:",
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/embeddings"
        self._model = model
        self._dimensions = dimensions
        self._batch_size = max(1, batch_size)
        self._passage_prefix = passage_prefix
        self._query_prefix = query_prefix
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        )
        self._async_client = async_client
        self._owns_client = client is None

    # -- API de LangChain ---------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prepared = [f"{self._passage_prefix}{text}" for text in texts]
        return self._embed_sync(prepared)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_sync([f"{self._query_prefix}{text}"])[0]

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prepared = [f"{self._passage_prefix}{text}" for text in texts]
        return await self._embed_async(prepared)

    async def aembed_query(self, text: str) -> list[float]:
        vectors = await self._embed_async([f"{self._query_prefix}{text}"])
        return vectors[0]

    # -- transporte ---------------------------------------------------------

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                response = self._client.post(
                    self._url,
                    json={"model": self._model, "input": batch, "encoding_format": "float"},
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise EmbeddingsServiceError(
                    f"El servidor de embeddings local no respondio correctamente: {exc}"
                ) from exc
            vectors.extend(self._parse(response.json(), expected=len(batch)))
        return vectors

    async def _embed_async(self, texts: list[str]) -> list[list[float]]:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=self._client.timeout,
                headers=dict(self._client.headers),
            )
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                response = await self._async_client.post(
                    self._url,
                    json={"model": self._model, "input": batch, "encoding_format": "float"},
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise EmbeddingsServiceError(
                    f"El servidor de embeddings local no respondio correctamente: {exc}"
                ) from exc
            vectors.extend(self._parse(response.json(), expected=len(batch)))
        return vectors

    def _parse(self, payload: dict, *, expected: int) -> list[list[float]]:
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise EmbeddingsServiceError(
                f"Respuesta de embeddings invalida: se esperaban {expected} vectores"
            )
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors: list[list[float]] = []
        for item in ordered:
            vector = item.get("embedding")
            if not isinstance(vector, list):
                raise EmbeddingsServiceError("Vector de embedding ausente en la respuesta")
            if self._dimensions and len(vector) != self._dimensions:
                raise EmbeddingsServiceError(
                    f"Dimension inesperada: {len(vector)} (se esperaban {self._dimensions}). "
                    "Revisa EMBEDDINGS_DIMENSIONS."
                )
            vectors.append([float(value) for value in vector])
        return vectors

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
