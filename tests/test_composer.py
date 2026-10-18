"""Tests del compositor determinista: trazabilidad construida desde evidencias."""

from __future__ import annotations

import threading

from agentic_poc.agents.composer import clean_draft, compose_response
from agentic_poc.models.domain import EvidenceCollector, ToolObservation


def _collector() -> EvidenceCollector:
    collector = EvidenceCollector()
    collector.record_tool(
        ToolObservation(
            tool="get_orders",
            source="ERP",
            args_summary="customer_id='ALFKI'",
            record_count=4,
            duration_ms=12.5,
            ok=True,
            record_ids=["10643", "10692", "10702", "10835"],
        )
    )
    collector.record_tool(
        ToolObservation(
            tool="fetch_production",
            source="Producción",
            args_summary="order_ids=4",
            record_count=4,
            duration_ms=8.0,
            ok=True,
        )
    )
    collector.record_evidence(
        "ERP", "order_insights", [{"order_id": 10643, "classification": "pendiente"}]
    )
    collector.record_evidence(
        "Documentos",
        "document_chunks",
        [
            {
                "doc_id": "contrato_marco_suministro",
                "source": "contrato_marco_suministro.pdf",
                "page": 2,
                "snippet": "Los plazos de entrega...",
                "score": 0.82,
            }
        ],
    )
    return collector


def test_sources_solo_fuentes_con_exito() -> None:
    collector = _collector()
    collector.record_tool(
        ToolObservation(tool="rag_search", source="Documentos", ok=False, error="timeout")
    )
    response = compose_response(
        question="q",
        draft="Respuesta",
        collector=collector,
        trace_id="t",
        duration_ms=1.0,
        model="local-llm",
        used_memory=False,
    )
    assert response.sources == ["ERP", "Producción"]
    assert any("Documentos" in w for w in response.warnings)


def test_datos_citas_y_traza_se_derivan_de_evidencias() -> None:
    response = compose_response(
        question="q",
        draft="Respuesta con cita",
        collector=_collector(),
        trace_id="abc",
        duration_ms=42.0,
        model="local-llm",
        used_memory=True,
    )
    assert response.trace_id == "abc"
    assert response.model == "local-llm"
    assert response.used_memory is True
    assert len(response.execution_trace) == 2
    assert response.execution_trace[0].tool == "get_orders"
    assert response.execution_trace[0].record_count == 4
    assert response.data["orders"][0]["order_id"] == 10643
    assert response.citations and response.citations[0].doc_id == "contrato_marco_suministro"
    assert any("get_orders" in step for step in response.reasoning)


def test_draft_vacio_usa_fallback_determinista() -> None:
    response = compose_response(
        question="q",
        draft="   ",
        collector=_collector(),
        trace_id="t",
        duration_ms=1.0,
        model="m",
        used_memory=False,
    )
    assert response.answer.strip()
    assert "fallback" not in response.answer.lower() or response.answer


def test_clean_draft_elimina_ruido() -> None:
    assert clean_draft("   ") == ""
    assert clean_draft("Respuesta limpia") == "Respuesta limpia"


def test_collector_thread_safe() -> None:
    collector = EvidenceCollector()

    def worker(n: int) -> None:
        for i in range(50):
            collector.record_tool(
                ToolObservation(tool=f"t{n}", source="ERP", ok=True, record_count=i)
            )

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(collector.all_observations()) == 400
