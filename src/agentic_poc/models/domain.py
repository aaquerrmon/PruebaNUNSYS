"""Modelos de dominio: estados, entidades de ERP/producción y evidencias.

Definiciones de dominio (documentadas y aplicadas de forma determinista):

- Pedido **pendiente**: el ERP no tiene fecha de envío.
- Pedido **completado**: el ERP tiene fecha de envío.
- Pedido **bloqueado**: el ERP no lo ha enviado y producción está en ``BLOCKED``.
- Pedido **retrasado**: el ERP no lo ha enviado y se cumple alguna de:
  producción en ``DELAYED`` con ``delay_flag`` (retraso de origen productivo),
  o ``required_date`` vencida (retraso por fechas). Un pedido no se considera
  retrasado únicamente por estar pendiente.

Prioridad de clasificación mutuamente excluyente:
``completado`` > ``bloqueado`` > ``retrasado`` > ``pendiente``.
"""

from __future__ import annotations

import threading
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SourceName = Literal["ERP", "Producción", "Documentos"]

MONEY_QUANTUM = Decimal("0.01")


class ErpOrderStatus(StrEnum):
    """Estado derivado del ERP (fecha de envío)."""

    PENDING = "pendiente"
    COMPLETED = "completado"


class ProductionStatus(StrEnum):
    """Estados del sistema de producción (API REST simulada)."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    BLOCKED = "BLOCKED"
    DELAYED = "DELAYED"
    COMPLETED = "COMPLETED"


class OrderClassification(StrEnum):
    """Clasificación de dominio resultante de la union ERP x produccion."""

    PENDING = "pendiente"
    BLOCKED = "bloqueado"
    DELAYED = "retrasado"
    COMPLETED = "completado"


# --------------------------------------------------------------------------
# Entidades ERP (SQLite Northwind de demostración)
# --------------------------------------------------------------------------


class ErpCustomer(BaseModel):
    model_config = ConfigDict(frozen=True)

    customer_id: str
    company_name: str
    contact_name: str | None = None
    country: str | None = None
    city: str | None = None


class ErpOrder(BaseModel):
    model_config = ConfigDict(frozen=True)

    order_id: int
    customer_id: str
    order_date: date
    required_date: date
    shipped_date: date | None = None
    freight: Decimal = Decimal("0")
    ship_city: str | None = None
    ship_country: str | None = None

    @property
    def erp_status(self) -> ErpOrderStatus:
        return ErpOrderStatus.COMPLETED if self.shipped_date else ErpOrderStatus.PENDING


class OrderLine(BaseModel):
    model_config = ConfigDict(frozen=True)

    order_id: int
    product_id: int
    product_name: str
    unit_price: Decimal
    quantity: int
    discount: Decimal = Decimal("0")

    @property
    def line_total(self) -> Decimal:
        return line_total(self.unit_price, self.quantity, self.discount)


class OrderTotals(BaseModel):
    """Importes calculados de forma determinista (nunca por el modelo)."""

    model_config = ConfigDict(frozen=True)

    order_id: int
    line_count: int
    gross: Decimal
    discount_amount: Decimal
    total: Decimal


# --------------------------------------------------------------------------
# Sistema de producción (API REST simulada)
# --------------------------------------------------------------------------


class ProductionRecord(BaseModel):
    """Registro de una orden de producción asociado a un pedido del ERP."""

    model_config = ConfigDict(frozen=True)

    id: int
    erp_order_id: int
    status: ProductionStatus
    planned_end_date: date
    actual_end_date: date | None = None
    delay_flag: bool = False
    delay_reason: str | None = None
    block_reason: str | None = None
    notes: str | None = None
    production_line: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --------------------------------------------------------------------------
# Lógica determinista de estados e importes
# --------------------------------------------------------------------------


def line_total(unit_price: Decimal, quantity: int, discount: Decimal) -> Decimal:
    """Importe de linea = precio x cantidad x (1 - descuento), a 2 decimales."""
    gross = unit_price * quantity
    net = gross * (Decimal(1) - discount)
    return net.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def order_totals(lines: list[OrderLine]) -> OrderTotals:
    """Totales de un pedido a partir de sus líneas (Decimal, sin coma flotante)."""
    gross = Decimal("0")
    discount_amount = Decimal("0")
    for line in lines:
        base = line.unit_price * line.quantity
        gross += base
        discount_amount += base - line_total(line.unit_price, line.quantity, line.discount)
    gross = gross.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)
    discount_amount = discount_amount.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)
    total = (gross - discount_amount).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)
    return OrderTotals(
        order_id=lines[0].order_id if lines else 0,
        line_count=len(lines),
        gross=gross,
        discount_amount=discount_amount,
        total=total,
    )


def is_delayed(
    order: ErpOrder,
    production: ProductionRecord | None,
    today: date,
) -> bool:
    """True si el pedido está retrasado por producción o por fechas (nunca solo por estar pendiente)."""
    if order.shipped_date is not None:
        return False
    if production is not None and (
        production.status == ProductionStatus.DELAYED or production.delay_flag
    ):
        return True
    return order.required_date < today


def delay_cause(
    order: ErpOrder,
    production: ProductionRecord | None,
    today: date,
) -> str | None:
    """Causa legible del retraso, para respuesta y traza."""
    if not is_delayed(order, production, today):
        return None
    if production is not None and (
        production.status == ProductionStatus.DELAYED or production.delay_flag
    ):
        detail = production.delay_reason or "retraso registrado en producción"
        return f"producción: {detail}"
    return f"fecha requerida vencida el {order.required_date.isoformat()}"


def classify_order(
    order: ErpOrder,
    production: ProductionRecord | None,
    today: date,
) -> OrderClassification:
    """Clasificación mutuamente excluyente (ver docstring del módulo)."""
    if order.shipped_date is not None:
        return OrderClassification.COMPLETED
    if production is not None and production.status == ProductionStatus.BLOCKED:
        return OrderClassification.BLOCKED
    if is_delayed(order, production, today):
        return OrderClassification.DELAYED
    return OrderClassification.PENDING


# --------------------------------------------------------------------------
# Evidencias y observaciones de herramientas (entrada del consolidador)
# --------------------------------------------------------------------------


class ToolObservation(BaseModel):
    """Registro de una herramienta realmente ejecutada durante la petición."""

    model_config = ConfigDict(frozen=True)

    tool: str
    source: SourceName | None = None
    args_summary: str = ""
    record_count: int | None = None
    duration_ms: float = 0.0
    ok: bool = True
    error: str | None = None
    record_ids: list[str] = Field(default_factory=list)


class EvidenceRecord(BaseModel):
    """Evidencia tipada devuelta por una herramienta (payload JSON-segmentable)."""

    model_config = ConfigDict(frozen=True)

    source: SourceName
    kind: str
    payload: Any = None


class EvidenceCollector:
    """Recolector por petición, seguro para hilos (herramientas sincrónicas)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.observations: list[ToolObservation] = []
        self.records: list[EvidenceRecord] = []

    def record_tool(self, observation: ToolObservation) -> None:
        with self._lock:
            self.observations.append(observation)

    def record_evidence(self, source: SourceName, kind: str, payload: Any) -> None:
        with self._lock:
            self.records.append(EvidenceRecord(source=source, kind=kind, payload=payload))

    @property
    def sources_consulted(self) -> list[SourceName]:
        """Fuentes con al menos una herramienta ejecutada con éxito.

        Una fuente cuya llamada falló no aparece aquí (se traslada a ``warnings``).
        """
        ordered: list[SourceName] = []
        with self._lock:
            for obs in self.observations:
                if obs.ok and obs.source and obs.source not in ordered:
                    ordered.append(obs.source)
        return ordered

    def failed_sources(self) -> list[tuple[SourceName, str]]:
        with self._lock:
            return [
                (obs.source, obs.error or "error desconocido")
                for obs in self.observations
                if (not obs.ok) and obs.source
            ]

    def of_kind(self, kind: str) -> list[EvidenceRecord]:
        with self._lock:
            return [record for record in self.records if record.kind == kind]

    def all_observations(self) -> list[ToolObservation]:
        with self._lock:
            return list(self.observations)


def utcnow() -> datetime:
    return datetime.now(UTC)
