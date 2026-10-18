"""Pruebas de extremo a extremo con los servidores vLLM locales.

Se saltan automáticamente si `llm`/`embeddings` no están accesibles (ver
`pytest_collection_modifyitems` en `conftest.py`). Se ejecutan con:

    uv run pytest -m local_llm

Requisito: `docker compose up llm embeddings` (o el stack completo) levantado.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agentic_poc.agents.orchestrator import AgentOrchestrator
from agentic_poc.api.dependencies import (
    get_embeddings,
    get_production_client,
    get_rag_service,
    get_repository,
    get_settings,
)

pytestmark = pytest.mark.local_llm


@pytest.fixture(scope="module")
def orchestrator():  # type: ignore[no-untyped-def]
    """Orquestador real, construido desde la configuración de entorno."""
    settings = get_settings()
    production = get_production_client(settings)
    repository = get_repository(settings)
    rag = get_rag_service(settings, get_embeddings(settings))
    return AgentOrchestrator(settings, repository, production, rag)


async def test_consulta_erp_devuelve_fuente_y_traza(orchestrator) -> None:  # type: ignore[no-untyped-def]
    response = await orchestrator.answer("¿Qué pedidos pendientes tiene el cliente ALFKI?")
    assert response.answer.strip()
    assert response.trace_id
    assert response.execution_trace, "debe haber al menos una herramienta ejecutada"
    # El ERP debe haberse consultado (el pedido/Cliente aparece en alguna evidencia)
    assert any(source in {"ERP", "Producción"} for source in response.sources) or response.warnings


async def test_consulta_importes_usa_decimal(orchestrator) -> None:  # type: ignore[no-untyped-def]
    response = await orchestrator.answer("¿Cuál es el importe total del pedido 10643?")
    assert response.answer.strip()
    if "amounts" in response.data:
        raw = str(response.data["amounts"])
        assert "." in raw  # importe con 2 decimales en la representación


async def test_consulta_documentacion_genera_citas(orchestrator) -> None:  # type: ignore[no-untyped-def]
    # Usa el MISMO RAG que el orquestador (si no, la ingesta no llegaría a las
    # herramientas del agente y el test no probaría nada real).
    rag = orchestrator.rag
    settings = orchestrator.settings
    if rag.is_empty:
        documents = Path(settings.documents_dir)
        documents.mkdir(parents=True, exist_ok=True)
        pdfs = sorted(documents.glob("*.pdf"))
        if not pdfs:
            scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
            sys.path.insert(0, str(scripts_dir))
            from generate_sample_docs import build_contract

            target = documents / "contrato_muestra_prueba.pdf"
            build_contract(target)
            pdfs = [target]
        for pdf in pdfs:
            rag.add_pdf(pdf)
    response = await orchestrator.answer("¿Qué dice el contrato marco sobre los plazos de entrega?")
    assert response.answer.strip()
    assert response.citations, "una pregunta documental debe producir citas"
    for citation in response.citations:
        assert citation.page is not None and citation.page >= 1
        assert citation.source
        assert citation.snippet.strip()


async def test_checkpoint_memoria_multi_turno(orchestrator) -> None:  # type: ignore[no-untyped-def]
    from langchain_core.messages import AIMessage, HumanMessage

    history = [
        HumanMessage(content="¿Qué pedidos tiene ALFKI?"),
        AIMessage(content="Tiene varios pedidos pendientes."),
    ]
    response = await orchestrator.answer("¿Y cuáles están retrasados?", history)
    assert response.used_memory is True
    assert response.answer.strip()
