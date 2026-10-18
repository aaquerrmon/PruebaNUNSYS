"""Fixtures compartidas: ERP sembrado en temporal, API de producción in-memory,
ajustes aislados y utilidades de salto para pruebas que necesitan servicios."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import Engine

from agentic_poc.config import Settings, get_settings
from agentic_poc.infrastructure.database import make_writable_engine
from agentic_poc.infrastructure.erp_seed import seed
from agentic_poc.services.production_client import ProductionClient

_TMP = Path(__file__).parent / "_tmp"


@pytest.fixture()
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Settings aislados por test, sin tocar la base ni los vectores reales."""
    get_settings.cache_clear()
    data_dir = tmp_path / "data"
    env = {
        "APP_ENV": "test",
        "ERP_DATABASE_URL": f"sqlite:///{data_dir / 'erp' / 'northwind.db'}",
        "APP_DB_PATH": str(data_dir / "app.db"),
        "VECTOR_STORE_PATH": str(data_dir / "vector_store"),
        "DOCUMENTS_DIR": str(data_dir / "documents"),
        "PRODUCTION_API_URL": "http://127.0.0.1:9",
        "LLM_BASE_URL": "http://127.0.0.1:9/v1",
        "EMBEDDINGS_BASE_URL": "http://127.0.0.1:9/v1",
        "ALLOW_REMOTE_INFERENCE": "false",
        "LOG_FORMAT": "text",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    _clear_engine_caches()
    yield get_settings()
    get_settings.cache_clear()
    _clear_engine_caches()


def _clear_engine_caches() -> None:
    from agentic_poc.api import dependencies as deps

    deps._get_engine_app.cache_clear()
    deps._get_engine_erp.cache_clear()


@pytest.fixture()
def erp_engine(settings: Settings) -> Iterator[Engine]:
    """Motor SQLite con los fixtures de Northwind sembrados (escritura solo aquí)."""
    db_path = Path(settings.erp_database_url.replace("sqlite:///", ""))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = make_writable_engine(settings.erp_database_url)
    seed(engine, today=__import__("datetime").date(2026, 10, 15))
    yield engine
    engine.dispose()


def pytest_configure(config: pytest.Config) -> None:
    _TMP.mkdir(exist_ok=True)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Salta automáticamente las pruebas local_llm/gpu si el servicio no está sano."""
    from agentic_poc.services.llm import probe_endpoint

    try:
        cfg = get_settings()
        probe_endpoint(cfg.llm_base_url)
        probe_endpoint(cfg.embeddings_base_url)
        llm_ok = True
    except Exception:
        llm_ok = False

    for item in items:
        if "local_llm" in item.keywords and not llm_ok:
            item.add_marker(pytest.mark.skip(reason="servidores vLLM locales no disponibles"))
        if "gpu" in item.keywords:
            import shutil

            if shutil.which("nvidia-smi") is None:
                item.add_marker(pytest.mark.skip(reason="nvidia-smi no disponible"))


@pytest.fixture()
def production_app() -> FastAPI:
    """App FastAPI de producción montada con transporte ASGI (sin puerto real)."""
    from agentic_poc.production_api.app import create_app

    return create_app()


@pytest.fixture()
def production_client(production_app: FastAPI) -> Iterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=production_app)
    client = httpx.AsyncClient(transport=transport, base_url="http://production.test")
    yield client


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evita que un .env local del desarrollador contamine los tests."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    os.environ.pop("OPENAI_API_KEY", None)


@pytest.fixture()
def production_service_client(production_app: FastAPI) -> ProductionClient:
    """ProductionClient apuntando a la app de producción vía transporte ASGI."""
    from agentic_poc.services.production_client import ProductionClient

    transport = httpx.ASGITransport(app=production_app)
    client = httpx.AsyncClient(transport=transport, base_url="http://production.test")
    return ProductionClient("http://production.test", client=client)
