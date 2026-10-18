"""Cliente HTTP (async) del sistema de producción simulado.

- Timeouts cortos y reintentos limitados sobre errores de transporte/5xx.
- Cualquier fallo se comunica con `ProductionServiceError`; las tools lo
  convierten en `warnings` y lo registran en la traza (nunca en ``sources``).
- Los registros se validan con el modelo de dominio `ProductionRecord`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import httpx

from agentic_poc.models.domain import ProductionRecord

logger = logging.getLogger(__name__)

DEFAULT_PATH = "/api/v1/production-orders"


class ProductionServiceError(RuntimeError):
    """Fallo de acceso a la API de producción (red, HTTP o esquema)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ProductionClient:
    """Cliente ligero reutilizable (un cliente httpx compartido por proceso)."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 10.0,
        max_retries: int = 1,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._max_retries = max(0, max_retries)
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_seconds),
            headers={"Accept": "application/json"},
        )
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def health(self) -> bool:
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def fetch_records(self, erp_order_ids: Sequence[int] = ()) -> dict[int, ProductionRecord]:
        """Devuelve {erp_order_id: registro}. Lista vacía => todos los registros."""
        params: dict[str, Any] = {"limit": 500}
        if erp_order_ids:
            params["erp_order_ids"] = ",".join(str(oid) for oid in sorted(set(erp_order_ids)))
        payload = await self._request_json(DEFAULT_PATH, params)
        items = payload.get("items")
        if not isinstance(items, list):
            raise ProductionServiceError(
                "Respuesta inesperada de la API de producción (sin 'items')"
            )
        records: dict[int, ProductionRecord] = {}
        for item in items:
            try:
                record = ProductionRecord.model_validate(item)
            except Exception as exc:  # pydantic.ValidationError
                raise ProductionServiceError(
                    f"Registro de producción con esquema invalido: {exc}"
                ) from exc
            records[record.erp_order_id] = record
        return records

    async def _request_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        attempts = self._max_retries + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = await self._client.get(path, params=params)
                if response.status_code >= 500:
                    raise ProductionServiceError(
                        f"La API de produccion respondio {response.status_code}",
                        status_code=response.status_code,
                    )
                if response.status_code != 200:
                    raise ProductionServiceError(
                        f"La API de produccion respondio {response.status_code}: "
                        f"{response.text[:200]}",
                        status_code=response.status_code,
                    )
                return response.json()
            except (httpx.HTTPError, ProductionServiceError) as exc:
                last_error = exc
                if attempt < attempts:
                    logger.warning(
                        "production_api intento %s/%s fallido: %s", attempt, attempts, exc
                    )
        if isinstance(last_error, ProductionServiceError):
            raise last_error
        raise ProductionServiceError(
            f"No se pudo contactar con la API de produccion en {self._base_url}: {last_error}"
        ) from last_error
