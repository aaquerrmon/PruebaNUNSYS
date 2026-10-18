"""Registro de evidencia y observaciones a partir de los análisis de negocio.

Separa el "qué" (resultado) del "cómo" (traza): cada fuente consultada deja
una observación independiente (ERP, Producción) y la fusión deja evidencia
tipada que el compositor usa para construir la respuesta estructurada.
"""

from __future__ import annotations

from typing import Any

from agentic_poc.models.domain import ErpCustomer, ErpOrder, SourceName
from agentic_poc.services.business import AnalysisBundle
from agentic_poc.tools.context import ToolContext, ToolRecorder

ERP_SOURCE: SourceName = "ERP"
PRODUCTION_SOURCE: SourceName = "Producción"
DOCUMENTS_SOURCE: SourceName = "Documentos"


def record_erp_step(
    context: ToolContext,
    *,
    args: str,
    orders: list[ErpOrder],
    customers: list[ErpCustomer],
) -> None:
    with ToolRecorder(context, "consultar_erp", source=ERP_SOURCE, args=args) as rec:
        context.collector.record_evidence(
            ERP_SOURCE,
            "customers",
            [c.model_dump(mode="json") for c in customers],
        )
        context.collector.record_evidence(
            ERP_SOURCE,
            "orders",
            [o.model_dump(mode="json") for o in orders],
        )
        rec.record_count = len(orders)
        rec.record_ids = [str(o.order_id) for o in orders]


def record_erp_customers(context: ToolContext, *, args: str, customers: list[ErpCustomer]) -> None:
    with ToolRecorder(context, "consultar_erp_clientes", source=ERP_SOURCE, args=args) as rec:
        context.collector.record_evidence(
            ERP_SOURCE,
            "customers",
            [c.model_dump(mode="json") for c in customers],
        )
        rec.record_count = len(customers)
        rec.record_ids = [c.customer_id for c in customers]


def record_production_step(context: ToolContext, *, args: str, bundle: AnalysisBundle) -> None:
    if bundle.production_error:
        with ToolRecorder(
            context, "consultar_produccion", source=PRODUCTION_SOURCE, args=args
        ) as rec:
            rec.record_count = 0
            raise RuntimeError(bundle.production_error)
    with ToolRecorder(context, "consultar_produccion", source=PRODUCTION_SOURCE, args=args) as rec:
        context.collector.record_evidence(
            PRODUCTION_SOURCE,
            "production_records",
            [r.model_dump(mode="json") for r in bundle.production.values()],
        )
        rec.record_count = len(bundle.production)
        rec.record_ids = [str(r.id) for r in bundle.production.values()]


def record_summary_evidence(context: ToolContext, bundle: AnalysisBundle) -> None:
    """Evidencia de la fusion ERP x Produccion (sin fuente propia)."""
    context.collector.record_evidence(
        ERP_SOURCE,
        "order_insights",
        [i.to_dict() for i in bundle.insights],
    )
    if bundle.totals:
        context.collector.record_evidence(
            ERP_SOURCE,
            "order_amounts",
            {
                str(oid): {
                    "line_count": totals.line_count,
                    "gross": str(totals.gross),
                    "discount": str(totals.discount_amount),
                    "total": str(totals.total),
                }
                for oid, totals in bundle.totals.items()
            },
        )
    context.collector.record_evidence(
        ERP_SOURCE,
        "analysis_summary",
        {
            "scope": bundle.scope,
            "order_count": len(bundle.insights),
            "classification_counts": bundle.classification_counts,
            "total_amount": str(bundle.total_amount),
            "production_consulted": not bundle.production_error,
        },
    )


def record_documents_step(
    context: ToolContext,
    *,
    tool: str,
    args: str,
    evidence_kind: str,
    payload: Any,
    record_count: int,
) -> None:
    with ToolRecorder(context, tool, source=DOCUMENTS_SOURCE, args=args) as rec:
        context.collector.record_evidence(DOCUMENTS_SOURCE, evidence_kind, payload)
        rec.record_count = record_count


__all__ = [
    "DOCUMENTS_SOURCE",
    "ERP_SOURCE",
    "PRODUCTION_SOURCE",
    "record_documents_step",
    "record_erp_customers",
    "record_erp_step",
    "record_production_step",
    "record_summary_evidence",
]
