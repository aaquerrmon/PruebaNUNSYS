"""Tests de integración con el ERP: consultas del repositorio y solo lectura real."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError

from agentic_poc.config import Settings
from agentic_poc.infrastructure.database import make_readonly_engine
from agentic_poc.repositories.erp_repository import ErpRepository
from agentic_poc.services.business import BusinessService
from agentic_poc.services.production_client import ProductionClient


def test_seed_idempotente(erp_engine) -> None:  # type: ignore[no-untyped-def]
    from agentic_poc.infrastructure.erp_seed import seed_if_configured

    stats = seed_if_configured(erp_engine)
    assert stats["orders"] >= 20
    # en re-ejecucion las cuentas de clientes no se recalculan (-1 = ya existian)
    assert stats["customers"] == -1 or stats["customers"] >= 10


def test_busqueda_y_pedido_por_cliente(erp_engine) -> None:  # type: ignore[no-untyped-def]
    repo = ErpRepository(erp_engine, max_rows=500)
    hits = repo.search_customers("ALFKI")
    assert [c.customer_id for c in hits] == ["ALFKI"]

    orders = repo.get_orders(customer_id="ALFKI")
    assert orders and all(o.customer_id == "ALFKI" for o in orders)
    assert all(
        o.shipped_date is None for o in repo.get_orders(customer_id="ALFKI", only_pending=True)
    )


def test_totales_decimales_del_repositorio(erp_engine) -> None:  # type: ignore[no-untyped-def]
    from decimal import Decimal

    repo = ErpRepository(erp_engine, max_rows=500)
    totals = repo.get_totals([10643])
    assert 10643 in totals
    assert totals[10643].total == Decimal("266.00")
    assert totals[10643].total == totals[10643].gross - totals[10643].discount_amount


def test_erp_solo_lectura_real(erp_engine, settings: Settings) -> None:  # type: ignore[no-untyped-def]
    """El motor de la aplicación no permite escrituras (mode=ro + query_only)."""
    path = Path(settings.erp_database_url.replace("sqlite:///", ""))
    engine = make_readonly_engine(f"sqlite:///{path.resolve()}")
    with engine.connect() as conn:
        from sqlalchemy import text

        with pytest.raises(OperationalError):
            conn.execute(text("DELETE FROM Orders"))
    engine.dispose()


def test_pending_order_ids_cruzables(erp_engine) -> None:  # type: ignore[no-untyped-def]
    repo = ErpRepository(erp_engine, max_rows=500)
    pending = repo.pending_order_ids()
    assert pending and all(isinstance(i, int) for i in pending)


async def test_business_service_bundle(
    erp_engine: Engine,
    production_service_client: ProductionClient,
) -> None:
    repo = ErpRepository(erp_engine, max_rows=500)
    service = BusinessService(repo, production_service_client, today=date(2026, 10, 15))
    bundle = await service.order_amounts([10643])
    assert [o.order_id for o in bundle.orders] == [10643]
    assert 10643 in bundle.totals
    assert bundle.totals[10643].total > 0
    insight = bundle.insights[0]
    assert insight.classification.value in {"completado", "pendiente", "bloqueado", "retrasado"}
    # La API de producción está inyectada (ASGI) y responde: sin error
    assert bundle.production_error is None
    assert bundle.production, "debe cruzarse con el sistema de produccion"


async def test_produccion_no_disponible_no_rompe(erp_engine: Engine) -> None:
    """Si la API de producción falla, el análisis sigue (con production_error)."""
    broken = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_broken_app()),
        base_url="http://production.test",
    )
    repo = ErpRepository(erp_engine, max_rows=500)
    service = BusinessService(
        repo, ProductionClient("http://production.test", client=broken), today=date(2026, 10, 15)
    )
    bundle = await service.order_amounts([10643])
    assert bundle.production_error is not None
    assert bundle.insights  # la clasificación sigue disponible


def _broken_app() -> FastAPI:
    from fastapi import FastAPI, HTTPException

    api = FastAPI()

    @api.get("/{path:path}")
    async def boom(path: str) -> dict[str, str]:
        raise HTTPException(status_code=500, detail="caido")

    return api
