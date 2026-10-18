"""API REST simulada del sistema de producción.

Endpoints (JSON, todas las respuestas son copias de `ProductionRecord`):

- ``GET /health``                     -> estado del servicio
- ``GET /api/v1/production-orders``   -> lista con filtros ``erp_order_ids``,
  ``status`` (CSV), ``delay_only`` y ``limit``
- ``GET /api/v1/production-orders/{id}`` -> un registro
- ``GET /api/v1/summary``             -> recuento por estado (``erp_order_ids`` opcional)

Los datos viven en memoria y se regeneran al arrancar desde los fixtures.
"""

from __future__ import annotations

from collections import Counter

from fastapi import FastAPI, HTTPException, Query

from agentic_poc.models.domain import ProductionRecord, ProductionStatus
from agentic_poc.production_api.fixtures import build_fixtures

MAX_LIMIT = 500


def _parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def _parse_ids(raw: str | None) -> list[int]:
    ids: list[int] = []
    for chunk in _parse_csv(raw):
        try:
            ids.append(int(chunk))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"ID no numerico: {chunk!r}") from exc
    return ids


class ProductionStore:
    """Almacenamiento en memoria del simulador."""

    def __init__(self, records: list[ProductionRecord] | None = None) -> None:
        self._records: dict[int, ProductionRecord] = {}
        self.reload(records)

    def reload(self, records: list[ProductionRecord] | None = None) -> int:
        records = records if records is not None else build_fixtures()
        self._records = {record.id: record for record in records}
        return len(self._records)

    def get(self, record_id: int) -> ProductionRecord | None:
        return self._records.get(record_id)

    @property
    def count(self) -> int:
        return len(self._records)

    def query(
        self,
        *,
        erp_order_ids: list[int] | None = None,
        statuses: list[ProductionStatus] | None = None,
        delay_only: bool = False,
        limit: int = 100,
    ) -> list[ProductionRecord]:
        wanted = set(erp_order_ids) if erp_order_ids else None
        status_set = set(statuses) if statuses else None
        results: list[ProductionRecord] = []
        for record in sorted(self._records.values(), key=lambda r: r.id):
            if wanted is not None and record.erp_order_id not in wanted:
                continue
            if status_set is not None and record.status not in status_set:
                continue
            if delay_only and not (record.delay_flag or record.status == ProductionStatus.DELAYED):
                continue
            results.append(record)
            if len(results) >= limit:
                break
        return results

    def summary(self, erp_order_ids: list[int] | None = None) -> dict[str, object]:
        records = self.query(erp_order_ids=erp_order_ids, limit=MAX_LIMIT)
        counts: Counter[str] = Counter(record.status.value for record in records)
        return {
            "total": len(records),
            "by_status": {status.value: counts.get(status.value, 0) for status in ProductionStatus},
        }


def create_app(store: ProductionStore | None = None) -> FastAPI:
    app = FastAPI(
        title="API de produccion (simulada)",
        version="1.0.0",
        description="Sistema de produccion simulado para la POC agentica.",
    )
    store = store or ProductionStore()
    app.state.store = store

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "records": store.count}

    @app.get("/api/v1/production-orders")
    def list_orders(
        erp_order_ids: str | None = Query(
            None, description="IDs de pedido ERP, separados por coma"
        ),
        status: str | None = Query(None, description="Estados, separados por coma"),
        delay_only: bool = Query(False, description="Solo registros con retraso"),
        limit: int = Query(100, ge=1, le=MAX_LIMIT),
    ) -> dict[str, object]:
        statuses: list[ProductionStatus] = []
        for value in _parse_csv(status):
            try:
                statuses.append(ProductionStatus(value))
            except ValueError as exc:
                raise HTTPException(
                    status_code=422, detail=f"Estado desconocido: {value!r}"
                ) from exc
        records = store.query(
            erp_order_ids=_parse_ids(erp_order_ids) or None,
            statuses=statuses or None,
            delay_only=delay_only,
            limit=limit,
        )
        return {
            "items": [record.model_dump(mode="json") for record in records],
            "count": len(records),
        }

    @app.get("/api/v1/production-orders/{record_id}")
    def get_order(record_id: int) -> dict[str, object]:
        record = store.get(record_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Registro de produccion no encontrado")
        return record.model_dump(mode="json")

    @app.get("/api/v1/summary")
    def summary(erp_order_ids: str | None = Query(None)) -> dict[str, object]:
        return store.summary(_parse_ids(erp_order_ids) or None)

    return app


app = create_app()


def main() -> None:  # pragma: no cover - punto de entrada uvicorn
    import uvicorn

    from agentic_poc.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "agentic_poc.production_api.app:app",
        host=settings.production_bind_host,
        port=settings.production_port,
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
