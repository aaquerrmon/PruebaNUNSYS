"""Fixtures del sistema de producción simulado (espejo de los pedidos ERP).

Estos registros son datos de DEMOSTRACION: cada uno referencia un OrderID del
seed del ERP (`infrastructure.erp_seed`). Los estados reproducen los casos de
la especificacion (pendiente, bloqueado con motivo, retrasado con causa,
completado) incluyendo los ejemplos tipicos del cliente ALFKI.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from agentic_poc.models.domain import ProductionRecord, ProductionStatus


def build_fixtures(today: date | None = None) -> list[ProductionRecord]:
    """Genera los registros de produccion anclados a `today` (idempotente)."""
    today = today or date.today()
    now = datetime.now(UTC)

    def rec(
        fixture_id: int,
        erp_order_id: int,
        status: ProductionStatus,
        planned_end: int,
        *,
        actual_end_days: int | None = None,
        delay_flag: bool = False,
        delay_reason: str | None = None,
        block_reason: str | None = None,
        notes: str | None = None,
        line: str = "L1",
    ) -> ProductionRecord:
        return ProductionRecord(
            id=fixture_id,
            erp_order_id=erp_order_id,
            status=status,
            planned_end_date=today + timedelta(days=planned_end),
            actual_end_date=(today - timedelta(days=actual_end_days)) if actual_end_days else None,
            delay_flag=delay_flag,
            delay_reason=delay_reason,
            block_reason=block_reason,
            notes=notes,
            production_line=line,
            updated_at=now,
        )

    return [
        # --- ALFKI (casos de la especificacion) ---------------------------
        rec(1, 10643, ProductionStatus.IN_PROGRESS, 6, notes="montaje en curso", line="L2"),
        rec(2, 10692, ProductionStatus.COMPLETED, -40, actual_end_days=56),
        rec(
            3,
            10702,
            ProductionStatus.BLOCKED,
            3,
            block_reason="falta de componentes",
            notes="espera de PCBs del proveedor principal",
            line="L1",
        ),
        rec(4, 10835, ProductionStatus.COMPLETED, -25, actual_end_days=33),
        rec(
            5,
            11011,
            ProductionStatus.DELAYED,
            -4,
            delay_flag=True,
            delay_reason="proveedor de materiales",
            notes="entrega de materia prima reprogramada",
            line="L3",
        ),
        # --- Otros clientes con incidencias --------------------------------
        rec(
            6,
            10690,
            ProductionStatus.BLOCKED,
            4,
            block_reason="averia de maquina",
            notes="parada no planificada en L2",
            line="L2",
        ),
        rec(
            7,
            10691,
            ProductionStatus.DELAYED,
            -2,
            delay_flag=True,
            delay_reason="error de calidad",
            notes="retrabajo tras inspeccion",
            line="L3",
        ),
        rec(8, 10693, ProductionStatus.PENDING, 12, notes="programado para la proxima semana"),
        rec(9, 10694, ProductionStatus.PENDING, 9, notes="sin incidencias en produccion"),
        rec(10, 10695, ProductionStatus.COMPLETED, -8, actual_end_days=13),
        rec(
            11,
            10696,
            ProductionStatus.DELAYED,
            -1,
            delay_flag=True,
            delay_reason="plazo ampliado del proveedor",
            line="L1",
        ),
        rec(
            12,
            10697,
            ProductionStatus.BLOCKED,
            8,
            block_reason="retencion de calidad",
            notes="pendiente de validacion del cliente",
            line="L2",
        ),
        rec(13, 10698, ProductionStatus.COMPLETED, -5, actual_end_days=10),
        rec(14, 10699, ProductionStatus.PENDING, 15),
        # --- Pedidos del mes en curso ---------------------------------------
        rec(15, 20001, ProductionStatus.PENDING, 20, line="L1"),
        rec(16, 20002, ProductionStatus.COMPLETED, 5, actual_end_days=2, line="L2"),
        rec(
            17,
            20003,
            ProductionStatus.DELAYED,
            10,
            delay_flag=True,
            delay_reason="retraso en corte de material",
            line="L3",
        ),
        rec(
            18,
            20004,
            ProductionStatus.BLOCKED,
            11,
            block_reason="falta de componentes",
            line="L1",
        ),
        rec(19, 20005, ProductionStatus.COMPLETED, 10, actual_end_days=4, line="L2"),
        rec(20, 20006, ProductionStatus.PENDING, 18, line="L3"),
    ]
