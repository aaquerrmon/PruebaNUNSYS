"""Tests de la API FastAPI: endpoints, memoria conversacional y validación.

Usan un orquestador de prueba inyectado por `dependency_overrides`: no
requieren ningún servicio vLLM (la inferencia real se prueba con el marcador
`local_llm`).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from agentic_poc.api.app import app
from agentic_poc.api.dependencies import get_orchestrator
from agentic_poc.config import Settings
from agentic_poc.models.responses import (
    Citation,
    ExecutionStep,
    QueryResponse,
)


class _StubOrchestrator:
    """Orquestador mínimo que emula la forma real de la respuesta."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def answer(self, question: str, history: list | None = None) -> QueryResponse:
        history = history or []
        self.calls.append((question, len(history)))
        return QueryResponse(
            answer=f"Respuesta para: {question}",
            sources=["ERP", "Producción"],
            reasoning=["get_orders: 4 registros (fuente ERP)"],
            data={"orders": [{"order_id": 10643, "classification": "pendiente"}]},
            citations=[
                Citation(
                    doc_id="contrato_marco_suministro",
                    source="contrato_marco_suministro.pdf",
                    page=2,
                    snippet="Los plazos...",
                    score=0.82,
                )
            ],
            warnings=[],
            execution_trace=[
                ExecutionStep(tool="get_orders", source="ERP", record_count=4, duration_ms=5.0)
            ],
            trace_id="trace-1",
            duration_ms=12.0,
            model="local-llm",
            used_memory=bool(history),
        )


@pytest.fixture()
def client(settings, erp_engine) -> Iterator[TestClient]:  # type: ignore[no-untyped-def]
    stub = _StubOrchestrator()
    app.dependency_overrides[get_orchestrator] = lambda: stub
    with TestClient(app) as test_client:
        test_client.stub = stub  # type: ignore[attr-defined]
        yield test_client
    app.dependency_overrides.clear()


def test_query_devuelve_contrato_completo(client: TestClient) -> None:
    response = client.post("/api/query", json={"question": "¿Pedidos de ALFKI?"})
    assert response.status_code == 200
    body = response.json()
    for key in ("answer", "sources", "reasoning", "execution_trace", "trace_id", "citations"):
        assert key in body
    assert body["sources"] == ["ERP", "Producción"]
    assert body["execution_trace"][0]["tool"] == "get_orders"


def test_query_valida_pregunta_corta(client: TestClient) -> None:
    assert client.post("/api/query", json={"question": "x"}).status_code == 422


def test_memoria_conversacional(client: TestClient) -> None:
    first = client.post("/api/query", json={"question": "primer pregunta", "conversation_id": "c1"})
    assert first.json()["used_memory"] is False
    second = client.post(
        "/api/query", json={"question": "segunda pregunta", "conversation_id": "c1"}
    )
    assert second.json()["used_memory"] is True

    history = client.get("/api/conversations/c1/history")
    assert history.status_code == 200
    roles = [m["role"] for m in history.json()["messages"]]
    assert roles == ["human", "assistant", "human", "assistant"]


def test_historial_vacio(client: TestClient) -> None:
    response = client.get("/api/conversations/desconocida/history")
    assert response.status_code == 200
    assert response.json()["messages"] == []


def test_query_stream_eventos(client: TestClient) -> None:
    events: list[str] = []
    with client.stream(
        "POST", "/api/query/stream", json={"question": "¿Pedidos de ALFKI?"}
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(line[6:])
    joined = "\n".join(events)
    assert '"event":"start"' in joined
    assert '"event":"answer"' in joined
    assert '"event":"done"' in joined
    assert "Respuesta para:" in joined


def test_health_reporta_componentes(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert set(body) >= {"status", "llm", "embeddings", "production_api", "erp", "documents"}
    assert body["erp"] == "ok"
    assert isinstance(body["documents"], int)


def test_upload_rechaza_no_pdf(client: TestClient) -> None:
    response = client.post(
        "/api/documents",
        files={"file": ("nota.txt", b"hola", "text/plain")},
    )
    assert response.status_code == 415


def test_lista_documentos_refleja_el_indice(client: TestClient, settings) -> None:  # type: ignore[no-untyped-def]
    """Regresión: GET /api/documents devolvía 500 (DocumentInfo con slots, sin __dict__)."""
    from pathlib import Path

    from agentic_poc.api.dependencies import get_rag_service
    from test_rag import _make_pdf

    documents_dir = Path(settings.documents_dir)
    documents_dir.mkdir(parents=True, exist_ok=True)
    pdf = _make_pdf(str(documents_dir / "doc_prueba.pdf"), ["Contenido de prueba del documento."])
    rag = _fake_rag(settings)
    rag.add_pdf(pdf)
    app.dependency_overrides[get_rag_service] = lambda: rag
    try:
        response = client.get("/api/documents")
    finally:
        app.dependency_overrides.pop(get_rag_service, None)

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert set(body[0]) == {"doc_id", "source", "chunks", "pages"}
    assert body[0]["chunks"] >= 1
    assert body[0]["pages"] == 1
    assert body[0]["source"].endswith(".pdf")


def _fake_rag(settings):  # type: ignore[no-untyped-def]
    """RAG aislado con embeddings falsos (sin vLLM) e índice en tmp."""
    from pathlib import Path

    from agentic_poc.services.rag import RagService
    from test_rag import _FakeEmbeddings

    return RagService(
        _FakeEmbeddings(),  # type: ignore[arg-type]
        persist_directory=Path(settings.vector_store_path),
        chunk_size=200,
        chunk_overlap=40,
        score_threshold=0.0,
    )


def test_upload_pdf_valido_lo_ingesta(client: TestClient, settings) -> None:  # type: ignore[no-untyped-def]
    from pathlib import Path

    from agentic_poc.api.dependencies import get_rag_service
    from test_rag import _make_pdf

    documents_dir = Path(settings.documents_dir)
    documents_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = _make_pdf(
        str(documents_dir / "fuente_subida.pdf"), ["Cláusula de prueba del contrato."]
    )
    rag = _fake_rag(settings)
    app.dependency_overrides[get_rag_service] = lambda: rag
    try:
        with pdf_path.open("rb") as handle:
            response = client.post(
                "/api/documents",
                files={"file": ("contrato_prueba.pdf", handle, "application/pdf")},
            )
        listed = client.get("/api/documents")
    finally:
        app.dependency_overrides.pop(get_rag_service, None)

    assert response.status_code == 201
    body = response.json()
    assert body["pages"] == 1
    assert body["chunks"] >= 1
    assert listed.status_code == 200
    assert any(d["source"] == "contrato_prueba.pdf" for d in listed.json())


def test_upload_rechaza_pdf_demasiado_grande(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    from agentic_poc.api.dependencies import get_rag_service

    monkeypatch.setattr(settings, "max_pdf_bytes", 16)
    documents_dir = Path(settings.documents_dir)
    documents_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = documents_dir / "grande.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 " + b"0" * 64)
    app.dependency_overrides[get_rag_service] = lambda: _fake_rag(settings)
    try:
        with pdf_path.open("rb") as handle:
            response = client.post(
                "/api/documents",
                files={"file": ("grande.pdf", handle, "application/pdf")},
            )
    finally:
        app.dependency_overrides.pop(get_rag_service, None)
    assert response.status_code == 413
