"""Tests de la API de producción simulada (vía transporte ASGI, sin puerto)."""

from __future__ import annotations

import httpx
import pytest


async def test_health(production_client: httpx.AsyncClient) -> None:
    response = await production_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_listado_y_filtro_por_estado(production_client: httpx.AsyncClient) -> None:
    response = await production_client.get("/api/v1/production-orders", params={"limit": 500})
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) >= 20

    blocked = await production_client.get(
        "/api/v1/production-orders", params={"status": "BLOCKED", "limit": 500}
    )
    assert blocked.status_code == 200
    assert all(i["status"] == "BLOCKED" for i in blocked.json()["items"])
    assert blocked.json()["items"], "debe haber pedidos BLOCKED en los fixtures"


async def test_filtro_por_ids(production_client: httpx.AsyncClient) -> None:
    response = await production_client.get(
        "/api/v1/production-orders", params={"erp_order_ids": "10643,10692", "limit": 500}
    )
    ids = {i["erp_order_id"] for i in response.json()["items"]}
    assert ids == {10643, 10692}


async def test_detalle_y_404(production_client: httpx.AsyncClient) -> None:
    listing = await production_client.get("/api/v1/production-orders", params={"limit": 1})
    record_id = listing.json()["items"][0]["id"]
    detail = await production_client.get(f"/api/v1/production-orders/{record_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == record_id

    missing = await production_client.get("/api/v1/production-orders/999999")
    assert missing.status_code == 404


async def test_summary(production_client: httpx.AsyncClient) -> None:
    response = await production_client.get("/api/v1/summary")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] >= 20
    assert set(payload["by_status"]) <= {
        "PENDING",
        "IN_PROGRESS",
        "BLOCKED",
        "DELAYED",
        "COMPLETED",
    }


async def test_status_invalido_es_422(production_client: httpx.AsyncClient) -> None:
    response = await production_client.get("/api/v1/production-orders", params={"status": "NOPE"})
    assert response.status_code == 422


@pytest.mark.parametrize("status", ["BLOCKED", "DELAYED"])
async def test_estados_relevantes_presentes(
    production_client: httpx.AsyncClient, status: str
) -> None:
    response = await production_client.get(
        "/api/v1/production-orders", params={"status": status, "limit": 500}
    )
    assert response.status_code == 200
    assert response.json()["items"]
