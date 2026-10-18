"""Tests unitarios del dominio: clasificación, importes y causas de retraso."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from agentic_poc.models.domain import (
    ErpOrder,
    OrderClassification,
    OrderLine,
    ProductionRecord,
    ProductionStatus,
    classify_order,
    delay_cause,
    is_delayed,
    line_total,
    order_totals,
)


def _order(
    *,
    required: date = date(2026, 10, 1),
    shipped: date | None = None,
    order_id: int = 10643,
) -> ErpOrder:
    return ErpOrder(
        order_id=order_id,
        customer_id="ALFKI",
        order_date=date(2026, 9, 1),
        required_date=required,
        shipped_date=shipped,
        ship_city="Berlin",
        ship_country="Germany",
        freight=Decimal("10.00"),
    )


def _production(status: ProductionStatus, **kwargs) -> ProductionRecord:  # type: ignore[no-untyped-def]
    defaults = {
        "id": 1,
        "erp_order_id": 10643,
        "status": status,
        "planned_end_date": date(2026, 10, 20),
        "actual_end_date": None,
        "delay_flag": False,
        "delay_reason": None,
        "block_reason": None,
        "notes": None,
        "production_line": "L1",
        "updated_at": date(2026, 10, 10),
    }
    defaults.update(kwargs)
    return ProductionRecord(**defaults)  # type: ignore[arg-type]


TODAY = date(2026, 10, 15)


def test_completado_tiene_prioridad_sobre_todo() -> None:
    order = _order(required=date(2026, 9, 1), shipped=date(2026, 9, 5))
    prod = _production(ProductionStatus.BLOCKED)
    assert classify_order(order, prod, TODAY) is OrderClassification.COMPLETED


def test_bloqueado_tiene_prioridad_sobre_retrasado() -> None:
    order = _order(required=date(2026, 9, 1))  # required vencida
    prod = _production(ProductionStatus.BLOCKED)
    assert classify_order(order, prod, TODAY) is OrderClassification.BLOCKED


def test_retrasado_por_produccion_delayed() -> None:
    order = _order(required=date(2026, 12, 1))
    prod = _production(ProductionStatus.DELAYED, delay_flag=True, delay_reason="falta de pieza")
    assert classify_order(order, prod, TODAY) is OrderClassification.DELAYED
    assert delay_cause(order, prod, TODAY) == "producción: falta de pieza"


def test_retrasado_por_fecha_vencida_sin_produccion() -> None:
    order = _order(required=date(2026, 9, 1))
    assert classify_order(order, None, TODAY) is OrderClassification.DELAYED
    cause = delay_cause(order, None, TODAY)
    assert cause is not None and "fecha requerida vencida" in cause


def test_pendiente_nunca_es_retrasado() -> None:
    order = _order(required=date(2026, 12, 1))
    prod = _production(ProductionStatus.IN_PROGRESS)
    assert classify_order(order, prod, TODAY) is OrderClassification.PENDING
    assert not is_delayed(order, prod, TODAY)


def test_delay_flag_con_status_no_delayed_tambien_retrasa() -> None:
    order = _order(required=date(2026, 12, 1))
    prod = _production(ProductionStatus.IN_PROGRESS, delay_flag=True, delay_reason="cuello")
    assert is_delayed(order, prod, TODAY) is True
    assert delay_cause(order, prod, TODAY) == "producción: cuello"


def test_line_total_rounding_half_up() -> None:
    assert line_total(Decimal("10.005"), 1, Decimal("0")) == Decimal("10.01")
    assert line_total(Decimal("19.99"), 3, Decimal("0.15")) == Decimal("50.97")


def test_order_totals_orden_10643() -> None:
    lines = [
        OrderLine(
            order_id=10643,
            product_id=1,
            product_name="A",
            unit_price=Decimal("14.40"),
            quantity=12,
            discount=Decimal("0.05"),
        ),
        OrderLine(
            order_id=10643,
            product_id=2,
            product_name="B",
            unit_price=Decimal("9.80"),
            quantity=10,
            discount=Decimal("0"),
        ),
    ]
    totals = order_totals(lines)
    # 14.40*12 = 172.80 (con 5% dto = 164.16) + 9.80*10 = 98.00  ->  262.16
    assert totals.gross == Decimal("270.80")
    assert totals.discount_amount == Decimal("8.64")
    assert totals.total == Decimal("262.16")
    assert totals.line_count == 2
