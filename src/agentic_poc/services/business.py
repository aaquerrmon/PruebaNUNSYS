"""Lógica de negocio determinista: fusion ERP x Produccion y clasificación.

El modelo de lenguaje nunca calcula cifras ni clasifica pedidos: este servicio
produce los `OrderInsight` (clasificación, causa de retraso, motivo de bloqueo,
importes) y esos mismos datos alimentan tanto la respuesta estructurada como
la traza de ejecución.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from agentic_poc.models.domain import (
    ErpCustomer,
    ErpOrder,
    OrderClassification,
    OrderTotals,
    ProductionRecord,
    ProductionStatus,
    classify_order,
    delay_cause,
)
from agentic_poc.repositories.erp_repository import ErpRepository
from agentic_poc.services.production_client import ProductionClient, ProductionServiceError


@dataclass(frozen=True, slots=True)
class OrderInsight:
    """Pedido enriquecido con la información de producción y su clasificación."""

    order: ErpOrder
    customer_id: str
    company_name: str | None
    production: ProductionRecord | None
    classification: OrderClassification
    delay_cause: str | None
    block_reason: str | None
    totals: OrderTotals | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "order_id": self.order.order_id,
            "customer_id": self.customer_id,
            "company_name": self.company_name,
            "order_date": self.order.order_date.isoformat(),
            "required_date": self.order.required_date.isoformat(),
            "shipped_date": self.order.shipped_date.isoformat()
            if self.order.shipped_date
            else None,
            "classification": self.classification.value,
            "production_status": self.production.status.value if self.production else None,
            "production_line": self.production.production_line if self.production else None,
            "delay_cause": self.delay_cause,
            "block_reason": self.block_reason,
            "total": str(self.totals.total) if self.totals else None,
        }


@dataclass(frozen=True, slots=True)
class AnalysisBundle:
    """Resultado homogéneo de un análisis multi-fuente."""

    scope: str
    insights: list[OrderInsight]
    orders: list[ErpOrder]
    customers: dict[str, ErpCustomer] = field(default_factory=dict)
    production: dict[int, ProductionRecord] = field(default_factory=dict)
    totals: dict[int, OrderTotals] = field(default_factory=dict)
    production_error: str | None = None

    @property
    def classification_counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in OrderClassification}
        for insight in self.insights:
            counts[insight.classification.value] += 1
        return counts

    @property
    def total_amount(self) -> Decimal:
        return sum((t.total for t in self.totals.values()), Decimal("0"))

    def filter_classification(self, classification: OrderClassification) -> list[OrderInsight]:
        return [i for i in self.insights if i.classification == classification]


class BusinessService:
    """Orquestación de repositorio ERP + cliente de producción."""

    def __init__(
        self,
        repository: ErpRepository,
        production: ProductionClient,
        *,
        today: date | None = None,
    ) -> None:
        self._repo = repository
        self._production = production
        self._today = today or date.today()

    @property
    def today(self) -> date:
        return self._today

    # -- análisis principales -------------------------------------------------

    async def analyze_customer(
        self,
        customer_id: str,
        *,
        only_pending: bool = True,
        with_totals: bool = True,
    ) -> AnalysisBundle:
        customer = self._repo.get_customer(customer_id)
        orders = self._repo.get_orders(customer_id=customer_id, only_pending=only_pending)
        customers = {customer.customer_id: customer} if customer else {}
        return await self._build(
            scope=f"cliente {customer_id}" + (" (solo pendientes)" if only_pending else ""),
            orders=orders,
            customers=customers,
            with_totals=with_totals,
        )

    async def blocked_orders(self) -> AnalysisBundle:
        orders = self._repo.get_orders(only_pending=True)
        bundle = await self._build(scope="pedidos pendientes (bloqueos)", orders=orders)
        return bundle

    async def delayed_orders(self) -> AnalysisBundle:
        orders = self._repo.get_orders(only_pending=True)
        return await self._build(scope="pedidos pendientes (retrasos)", orders=orders)

    async def month_summary(self) -> AnalysisBundle:
        orders = self._repo.get_month_orders(self._today)
        all_orders = self._repo.get_orders(order_ids=[o.order_id for o in orders]) or orders
        return await self._build(
            scope="pedidos del mes en curso", orders=all_orders, with_totals=True
        )

    async def order_amounts(self, order_ids: list[int]) -> AnalysisBundle:
        orders = self._repo.get_orders(order_ids=order_ids)
        return await self._build(scope="importes de pedidos", orders=orders, with_totals=True)

    # -- construcción ----------------------------------------------------------

    async def _build(
        self,
        *,
        scope: str,
        orders: list[ErpOrder],
        customers: dict[str, ErpCustomer] | None = None,
        with_totals: bool = False,
    ) -> AnalysisBundle:
        customer_ids = sorted({order.customer_id for order in orders})
        resolved_customers = (
            customers if customers is not None else self._repo.get_customers(customer_ids)
        )
        pending_ids = [order.order_id for order in orders if order.shipped_date is None]

        production: dict[int, ProductionRecord] = {}
        production_error: str | None = None
        if pending_ids:
            try:
                production = await self._production.fetch_records(pending_ids)
            except ProductionServiceError as exc:
                production_error = str(exc)

        totals = self._repo.get_totals([order.order_id for order in orders]) if with_totals else {}
        insights = self._classify(orders, production, resolved_customers, totals)
        return AnalysisBundle(
            scope=scope,
            insights=insights,
            orders=orders,
            customers=resolved_customers,
            production=production,
            totals=totals,
            production_error=production_error,
        )

    def _classify(
        self,
        orders: list[ErpOrder],
        production: dict[int, ProductionRecord],
        customers: dict[str, ErpCustomer],
        totals: dict[int, OrderTotals],
    ) -> list[OrderInsight]:
        insights: list[OrderInsight] = []
        for order in orders:
            record = production.get(order.order_id)
            classification = classify_order(order, record, self._today)
            customer = customers.get(order.customer_id)
            insights.append(
                OrderInsight(
                    order=order,
                    customer_id=order.customer_id,
                    company_name=customer.company_name if customer else None,
                    production=record,
                    classification=classification,
                    delay_cause=delay_cause(order, record, self._today),
                    block_reason=(
                        record.block_reason
                        if record and record.status == ProductionStatus.BLOCKED
                        else None
                    ),
                    totals=totals.get(order.order_id),
                )
            )
        return insights
