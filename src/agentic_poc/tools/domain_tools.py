"""Herramientas de dominio expuestas al agente.

Cada herramienta:
- consulta una o varias fuentes (ERP, Producción, Documentos);
- registra en el `ToolContext` la evidencia estructurada y la observación
  (duración, nº de registros, éxito/error) para la respuesta trazable;
- devuelve un JSON compacto que el modelo puede leer. El modelo nunca calcula
  cifras: los importes y clasificaciones llegan ya calculados.

Las herramientas se construyen por petición como closures sobre el contexto,
con `StructuredTool.from_function` para fijar el esquema de argumentos.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from agentic_poc.models.domain import OrderClassification
from agentic_poc.services.business import AnalysisBundle, OrderInsight
from agentic_poc.services.rag import RagError
from agentic_poc.tools.context import ToolContext, ToolRecorder, dumps
from agentic_poc.tools.recording import (
    record_documents_step,
    record_erp_customers,
    record_erp_step,
    record_production_step,
    record_summary_evidence,
)
from agentic_poc.tools.schemas import (
    CustomerOrdersArgs,
    EmptyArgs,
    FindCustomersArgs,
    OrderAmountsArgs,
    SearchDocumentsArgs,
)


def _order_payload(insight: OrderInsight) -> dict[str, Any]:
    return {
        "order_id": insight.order.order_id,
        "customer_id": insight.customer_id,
        "company_name": insight.company_name,
        "required_date": insight.order.required_date.isoformat(),
        "shipped": insight.order.shipped_date is not None,
        "classification": insight.classification.value,
        "production_status": insight.production.status.value if insight.production else None,
        "delay_cause": insight.delay_cause,
        "block_reason": insight.block_reason,
        "total": str(insight.totals.total) if insight.totals else None,
    }


def _bundle_payload(bundle: AnalysisBundle) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "scope": bundle.scope,
        "order_count": len(bundle.insights),
        "classification_counts": bundle.classification_counts,
        "orders": [_order_payload(i) for i in bundle.insights],
    }
    if bundle.totals:
        payload["total_amount"] = str(sum((t.total for t in bundle.totals.values()), Decimal("0")))
    if bundle.production_error:
        payload["warning"] = (
            "No se pudo consultar el sistema de produccion: "
            f"{bundle.production_error}. Los estados de produccion no estan disponibles."
        )
    return payload


def _record_analysis(context: ToolContext, bundle: AnalysisBundle, *, erp_args: str) -> None:
    record_erp_step(
        context,
        args=erp_args,
        orders=bundle.orders,
        customers=list(bundle.customers.values()),
    )
    pending_ids = [i.order.order_id for i in bundle.insights if i.order.shipped_date is None]
    record_production_step(context, args=f"erp_order_ids={pending_ids}", bundle=bundle)
    record_summary_evidence(context, bundle)


def _make_find_customers(context: ToolContext) -> StructuredTool:
    async def find_customers(term: str | None = None) -> str:
        customers = context.repository.search_customers(term)
        record_erp_customers(context, args=f"term={term!r}", customers=customers)
        return dumps(
            {
                "count": len(customers),
                "customers": [c.model_dump(mode="json") for c in customers],
                "hint": (
                    "Usa get_customer_orders con el customer_id para ver sus pedidos."
                    if customers
                    else "No hay clientes que coincidan con el termino."
                ),
            }
        )

    return StructuredTool.from_function(
        coroutine=find_customers,
        name="find_customers",
        description=(
            "Busca clientes en el ERP por ID, razon social o persona de contacto. "
            "Necesario para traducir un nombre de cliente a su customer_id."
        ),
        args_schema=FindCustomersArgs,
    )


def _make_customer_orders(context: ToolContext) -> StructuredTool:
    async def get_customer_orders(customer_id: str, only_pending: bool = True) -> str:
        bundle = await context.business.analyze_customer(
            customer_id, only_pending=only_pending, with_totals=True
        )
        _record_analysis(
            context,
            bundle,
            erp_args=f"customer_id={customer_id!r}, only_pending={only_pending}",
        )
        if not bundle.orders:
            return dumps(
                {
                    "scope": bundle.scope,
                    "order_count": 0,
                    "hint": "El cliente no tiene pedidos con ese criterio.",
                }
            )
        return dumps(_bundle_payload(bundle))

    return StructuredTool.from_function(
        coroutine=get_customer_orders,
        name="get_customer_orders",
        description=(
            "Devuelve los pedidos de un cliente (por defecto solo pendientes) con su estado "
            "de produccion, clasificacion y motivo de retraso/bloqueo. Cubre preguntas como "
            "'que pedidos pendientes tiene el cliente X y en que estado estan'."
        ),
        args_schema=CustomerOrdersArgs,
    )


def _make_blocked_orders(context: ToolContext) -> StructuredTool:
    async def get_blocked_orders() -> str:
        bundle = await context.business.blocked_orders()
        _record_analysis(context, bundle, erp_args="solo pendientes")
        blocked = bundle.filter_classification(OrderClassification.BLOCKED)
        return dumps(
            {
                "scope": bundle.scope,
                "blocked_count": len(blocked),
                "blocked_orders": [_order_payload(i) for i in blocked],
                "production_available": bundle.production_error is None,
                "warning": (
                    f"No se pudo consultar produccion: {bundle.production_error}"
                    if bundle.production_error
                    else None
                ),
                "hint": (
                    "Cada pedido bloqueado incluye block_reason (motivo)."
                    if blocked
                    else "No hay pedidos bloqueados."
                ),
            }
        )

    return StructuredTool.from_function(
        coroutine=get_blocked_orders,
        name="get_blocked_orders",
        description=(
            "Lista los pedidos bloqueados y su motivo (block_reason), cruzando ERP y produccion. "
            "Responde a 'que pedidos estan bloqueados y cual es el motivo'."
        ),
        args_schema=EmptyArgs,
    )


def _make_delayed_orders(context: ToolContext) -> StructuredTool:
    async def get_delayed_orders() -> str:
        bundle = await context.business.delayed_orders()
        _record_analysis(context, bundle, erp_args="solo pendientes")
        delayed = bundle.filter_classification(OrderClassification.DELAYED)
        production_origin = [
            i for i in delayed if i.delay_cause and i.delay_cause.startswith("produccion")
        ]
        return dumps(
            {
                "scope": bundle.scope,
                "delayed_count": len(delayed),
                "delayed_by_production": [_order_payload(i) for i in production_origin],
                "delayed_all_causes": [_order_payload(i) for i in delayed],
                "production_available": bundle.production_error is None,
                "warning": (
                    f"No se pudo consultar produccion: {bundle.production_error}"
                    if bundle.production_error
                    else None
                ),
            }
        )

    return StructuredTool.from_function(
        coroutine=get_delayed_orders,
        name="get_delayed_orders",
        description=(
            "Lista los pedidos retrasados indicando la causa (produccion o fecha requerida vencida). "
            "Responde a 'que clientes tienen pedidos retrasados por problemas de produccion'."
        ),
        args_schema=EmptyArgs,
    )


def _make_month_summary(context: ToolContext) -> StructuredTool:
    async def get_month_summary() -> str:
        bundle = await context.business.month_summary()
        _record_analysis(context, bundle, erp_args="mes en curso")
        return dumps(_bundle_payload(bundle))

    return StructuredTool.from_function(
        coroutine=get_month_summary,
        name="get_month_summary",
        description=(
            "Resumen del estado de los pedidos del mes en curso (completados, pendientes, "
            "bloqueados y retrasados) con el importe total. Responde a 'dame un resumen del "
            "estado de los pedidos de este mes'."
        ),
        args_schema=EmptyArgs,
    )


def _make_order_amounts(context: ToolContext) -> StructuredTool:
    async def get_order_amounts(order_ids: list[int]) -> str:
        bundle = await context.business.order_amounts(order_ids)
        _record_analysis(context, bundle, erp_args=f"order_ids={order_ids}")
        found = {i.order.order_id for i in bundle.insights}
        return dumps(
            {
                "scope": bundle.scope,
                "order_count": len(bundle.insights),
                "total_amount": str(bundle.total_amount),
                "amounts": [
                    {
                        "order_id": i.order.order_id,
                        "line_count": i.totals.line_count if i.totals else 0,
                        "gross": str(i.totals.gross) if i.totals else None,
                        "discount": str(i.totals.discount_amount) if i.totals else None,
                        "total": str(i.totals.total) if i.totals else None,
                    }
                    for i in bundle.insights
                ],
                "not_found": [oid for oid in order_ids if oid not in found],
            }
        )

    return StructuredTool.from_function(
        coroutine=get_order_amounts,
        name="get_order_amounts",
        description=(
            "Calcula los importes (bruto, descuento y total, en euros) de una lista de pedidos "
            "del ERP. Usa este dato para el impacto economico; nunca estimes importes."
        ),
        args_schema=OrderAmountsArgs,
    )


def _make_search_documents(context: ToolContext) -> StructuredTool:
    async def search_documents(query: str, k: int = 5, doc_ids: list[str] | None = None) -> str:
        try:
            chunks = context.rag.search_relevant(query, k=k, doc_ids=doc_ids)
        except RagError as exc:
            with ToolRecorder(
                context,
                "search_documents",
                source="Documentos",
                args=f"query={query!r}, k={k}, doc_ids={doc_ids}",
            ) as rec:
                rec.record_count = 0
                raise
            return dumps({"error": str(exc), "count": 0, "chunks": []})
        record_documents_step(
            context,
            tool="search_documents",
            args=f"query={query!r}, k={k}, doc_ids={doc_ids}",
            evidence_kind="document_chunks",
            payload=[
                {
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "page": c.page,
                    "chunk_index": c.chunk_index,
                    "score": c.score,
                    "snippet": c.text[:400],
                }
                for c in chunks
            ],
            record_count=len(chunks),
        )
        return dumps(
            {
                "query": query,
                "count": len(chunks),
                "chunks": [
                    {"source": c.source, "page": c.page, "score": c.score, "text": c.text[:1200]}
                    for c in chunks
                ],
                "note": (
                    "Fragmentos recuperados de documentos indexados. Cita source y pagina."
                    if chunks
                    else "No hay fragmentos relevantes por encima del umbral en los documentos indexados."
                ),
            }
        )

    return StructuredTool.from_function(
        coroutine=search_documents,
        name="search_documents",
        description=(
            "Busca en los PDF indexados (contratos, procedimientos, SLA) y devuelve fragmentos "
            "con su origen y pagina. Usalo para preguntas sobre documentos: plazos, penalizaciones, etc."
        ),
        args_schema=SearchDocumentsArgs,
    )


def _make_list_documents(context: ToolContext) -> StructuredTool:
    async def list_documents() -> str:
        try:
            documents = context.rag.list_documents()
        except RagError as exc:
            with ToolRecorder(
                context, "list_documents", source="Documentos", args="sin argumentos"
            ) as rec:
                rec.record_count = 0
                raise
            return dumps({"error": str(exc), "count": 0, "documents": []})
        record_documents_step(
            context,
            tool="list_documents",
            args="sin argumentos",
            evidence_kind="document_list",
            payload=[
                {"doc_id": d.doc_id, "source": d.source, "chunks": d.chunks, "pages": d.pages}
                for d in documents
            ],
            record_count=len(documents),
        )
        return dumps(
            {
                "count": len(documents),
                "documents": [
                    {"doc_id": d.doc_id, "source": d.source, "chunks": d.chunks, "pages": d.pages}
                    for d in documents
                ],
            }
        )

    return StructuredTool.from_function(
        coroutine=list_documents,
        name="list_documents",
        description="Lista los documentos PDF indexados actualmente (nombre, paginas y fragmentos).",
        args_schema=EmptyArgs,
    )


def build_tools(context: ToolContext) -> list[BaseTool]:
    """Construye el conjunto de herramientas ligado al contexto de la petición."""
    return [
        _make_find_customers(context),
        _make_customer_orders(context),
        _make_blocked_orders(context),
        _make_delayed_orders(context),
        _make_month_summary(context),
        _make_order_amounts(context),
        _make_search_documents(context),
        _make_list_documents(context),
    ]
