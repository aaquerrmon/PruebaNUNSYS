"""Compositor de la respuesta estructurada (determinista).

El modelo solo redacta el texto (`draft`). Todo lo demás —fuentes, pasos,
clasificaciones, importes, citas y avisos— se construye aquí a partir de la
evidencia y las observaciones registradas por las herramientas. Si el modelo
no devuelve texto, se genera una respuesta de respaldo a partir de los datos.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from agentic_poc.models.domain import EvidenceCollector
from agentic_poc.models.responses import Citation, ExecutionStep, QueryResponse

_THINKING_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)

_TOOL_PHRASES = {
    "consultar_erp": "Consulta al ERP",
    "consultar_erp_clientes": "Consulta de clientes al ERP",
    "consultar_produccion": "Consulta al sistema de produccion",
    "search_documents": "Busqueda en documentos",
    "list_documents": "Listado de documentos",
}

_MAX_CITATIONS = 8


def clean_draft(text: str) -> str:
    """Elimina bloques de razonamiento y espacios sobrantes del texto del modelo."""
    cleaned = _THINKING_RE.sub("", text or "").strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def _collect_data(collector: EvidenceCollector) -> dict[str, Any]:
    data: dict[str, Any] = {}
    insights: dict[int, dict[str, Any]] = {}
    for record in collector.of_kind("order_insights"):
        for item in record.payload or []:
            if isinstance(item, dict) and "order_id" in item:
                insights[int(item["order_id"])] = item
    if insights:
        data["orders"] = [insights[key] for key in sorted(insights)]

    amounts: dict[str, Any] = {}
    for record in collector.of_kind("order_amounts"):
        if isinstance(record.payload, dict):
            amounts.update(record.payload)
    if amounts:
        data["amounts"] = amounts

    summaries = [record.payload for record in collector.of_kind("analysis_summary")]
    if summaries:
        data["summaries"] = summaries

    documents: list[dict[str, Any]] = []
    for record in collector.of_kind("document_list"):
        documents.extend(record.payload or [])
    if documents:
        data["documents"] = documents
    return data


def _collect_citations(collector: EvidenceCollector) -> list[Citation]:
    best: dict[tuple[str, int | None, int | None], Citation] = {}
    for record in collector.of_kind("document_chunks"):
        for chunk in record.payload or []:
            if not isinstance(chunk, dict):
                continue
            page = chunk.get("page")
            chunk_index = chunk.get("chunk_index")
            key = (
                str(chunk.get("doc_id", "")),
                page if isinstance(page, int) else None,
                chunk_index if isinstance(chunk_index, int) else None,
            )
            citation = Citation(
                doc_id=str(chunk.get("doc_id", "")),
                source=str(chunk.get("source", "documento")),
                page=page if isinstance(page, int) else None,
                snippet=str(chunk.get("snippet", ""))[:300],
                score=float(chunk.get("score", 0.0)),
            )
            existing = best.get(key)
            if existing is None or citation.score > existing.score:
                best[key] = citation
    ordered = sorted(best.values(), key=lambda c: c.score, reverse=True)
    return ordered[:_MAX_CITATIONS]


def _build_reasoning(collector: EvidenceCollector) -> list[str]:
    steps: list[str] = []
    for observation in collector.all_observations():
        if not observation.ok:
            continue
        phrase = _TOOL_PHRASES.get(observation.tool, observation.tool)
        count = observation.record_count
        detail = f"{count} registros" if count is not None else "sin resultados"
        if observation.source:
            steps.append(f"{phrase}: {detail} (fuente {observation.source})")
        else:
            steps.append(f"{phrase}: {detail}")
    insight_count = sum(len(r.payload or []) for r in collector.of_kind("order_insights"))
    if insight_count:
        steps.append(
            f"Fusion ERP x produccion y clasificacion de {insight_count} pedidos "
            "(completado/bloqueado/retrasado/pendiente)"
        )
    citation_count = sum(len(r.payload or []) for r in collector.of_kind("document_chunks"))
    if citation_count:
        steps.append(f"Extraccion de {citation_count} fragmentos de documentos como evidencia")
    return steps


def _build_warnings(collector: EvidenceCollector) -> list[str]:
    warnings: list[str] = []
    seen: set[str] = set()
    for observation in collector.all_observations():
        if observation.ok:
            continue
        error = observation.error or "error desconocido"
        if observation.source:
            message = f"La fuente {observation.source} no pudo consultarse: {error}"
        else:
            message = f"Error durante la ejecucion ({observation.tool}): {error}"
        if message not in seen:
            seen.add(message)
            warnings.append(message)
    return warnings


def _build_trace(collector: EvidenceCollector) -> list[ExecutionStep]:
    return [
        ExecutionStep(
            tool=observation.tool,
            source=observation.source,
            args_summary=observation.args_summary,
            record_count=observation.record_count,
            duration_ms=observation.duration_ms,
            ok=observation.ok,
            error=observation.error,
        )
        for observation in collector.all_observations()
    ]


def _fallback_answer(data: dict[str, Any], citations: list[Citation]) -> str:
    orders = data.get("orders") or []
    if orders:
        counts: dict[str, int] = {}
        for order in orders:
            key = str(order.get("classification", "desconocido"))
            counts[key] = counts.get(key, 0) + 1
        breakdown = ", ".join(f"{value} {key}" for key, value in sorted(counts.items()))
        amount = data.get("amounts") or {}
        total = sum(
            (Decimal(str(item.get("total", "0"))) for item in amount.values()), Decimal("0")
        )
        total_text = f" El importe total consultado asciende a {total} EUR." if amount else ""
        return f"Se analizaron {len(orders)} pedidos: {breakdown}.{total_text}"
    if citations:
        first = citations[0]
        return (
            f"Se recuperaron {len(citations)} fragmentos relevantes. "
            f"El mas relevante proviene de {first.source}"
            + (f", pagina {first.page}." if first.page else ".")
        )
    return "No se pudo construir una respuesta con las fuentes disponibles."


def compose_response(
    *,
    question: str,
    draft: str,
    collector: EvidenceCollector,
    trace_id: str,
    duration_ms: float,
    model: str,
    used_memory: bool,
) -> QueryResponse:
    data = _collect_data(collector)
    citations = _collect_citations(collector)
    answer = clean_draft(draft) or _fallback_answer(data, citations)
    return QueryResponse(
        answer=answer,
        sources=[str(source) for source in collector.sources_consulted],
        reasoning=_build_reasoning(collector),
        data=data,
        citations=citations,
        warnings=_build_warnings(collector),
        execution_trace=_build_trace(collector),
        trace_id=trace_id,
        duration_ms=duration_ms,
        model=model,
        used_memory=used_memory,
    )
