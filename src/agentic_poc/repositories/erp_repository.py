"""Repositorio de solo lectura sobre el ERP Northwind (SQLAlchemy Core).

Todas las consultas usan parametros enlazados (nunca SQL interpolado) y un
tope de filas por consulta para evitar que un agente pueda saturar la base.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from sqlalchemy import Engine, and_, or_, select

from agentic_poc.infrastructure.erp_seed import customers, order_details, orders, products
from agentic_poc.models.domain import (
    ErpCustomer,
    ErpOrder,
    OrderLine,
    OrderTotals,
)
from agentic_poc.models.domain import (
    order_totals as compute_totals,
)


class ErpRepository:
    """Acceso de lectura al ERP. Un repositorio por proceso, reutilizable."""

    def __init__(self, engine: Engine, *, max_rows: int = 200) -> None:
        self._engine = engine
        self.max_rows = max_rows

    def _limit(self, requested: int | None) -> int:
        if requested is None:
            return self.max_rows
        return max(1, min(requested, self.max_rows))

    # -- clientes ----------------------------------------------------------

    def search_customers(
        self, term: str | None = None, *, limit: int | None = None
    ) -> list[ErpCustomer]:
        stmt = select(customers)
        if term:
            pattern = f"%{term.strip()}%"
            stmt = stmt.where(
                or_(
                    customers.c.CustomerID.ilike(pattern),
                    customers.c.CompanyName.ilike(pattern),
                    customers.c.ContactName.ilike(pattern),
                )
            )
        stmt = stmt.order_by(customers.c.CustomerID).limit(self._limit(limit))
        with self._engine.connect() as conn:
            return [self._row_to_customer(row) for row in conn.execute(stmt)]

    def get_customer(self, customer_id: str) -> ErpCustomer | None:
        stmt = select(customers).where(customers.c.CustomerID == customer_id)
        with self._engine.connect() as conn:
            row = conn.execute(stmt).first()
        return self._row_to_customer(row) if row else None

    def get_customers(self, customer_ids: Sequence[str]) -> dict[str, ErpCustomer]:
        ids = sorted(set(customer_ids))
        if not ids:
            return {}
        stmt = select(customers).where(customers.c.CustomerID.in_(ids)).limit(self.max_rows)
        with self._engine.connect() as conn:
            return {
                customer.customer_id: customer
                for customer in (self._row_to_customer(row) for row in conn.execute(stmt))
            }

    def customer_ids_for_orders(self, order_ids: Sequence[int]) -> dict[int, str]:
        if not order_ids:
            return {}
        stmt = select(orders.c.OrderID, orders.c.CustomerID).where(
            orders.c.OrderID.in_(list(order_ids))
        )
        with self._engine.connect() as conn:
            return {int(r[0]): str(r[1]) for r in conn.execute(stmt)}

    # -- pedidos ------------------------------------------------------------

    def get_orders(
        self,
        *,
        customer_id: str | None = None,
        only_pending: bool = False,
        required_before: date | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        order_ids: Sequence[int] | None = None,
        limit: int | None = None,
    ) -> list[ErpOrder]:
        clauses = []
        if customer_id:
            clauses.append(orders.c.CustomerID == customer_id)
        if only_pending:
            clauses.append(orders.c.ShippedDate.is_(None))
        if required_before:
            clauses.append(orders.c.RequiredDate < required_before)
        if date_from:
            clauses.append(orders.c.OrderDate >= date_from)
        if date_to:
            clauses.append(orders.c.OrderDate <= date_to)
        if order_ids:
            clauses.append(orders.c.OrderID.in_(list(order_ids)))
        stmt = select(orders)
        if clauses:
            stmt = stmt.where(and_(*clauses))
        stmt = stmt.order_by(orders.c.OrderDate.desc(), orders.c.OrderID).limit(self._limit(limit))
        with self._engine.connect() as conn:
            return [self._row_to_order(row) for row in conn.execute(stmt)]

    def get_month_orders(self, today: date) -> list[ErpOrder]:
        """Pedidos con OrderDate dentro del mes en curso."""
        month_start = today.replace(day=1)
        stmt = (
            select(orders)
            .where(orders.c.OrderDate >= month_start, orders.c.OrderDate <= today)
            .order_by(orders.c.OrderID)
            .limit(self.max_rows)
        )
        with self._engine.connect() as conn:
            return [self._row_to_order(row) for row in conn.execute(stmt)]

    def pending_order_ids(self) -> list[int]:
        """IDs de pedidos sin enviar (para cruzar con produccion)."""
        stmt = (
            select(orders.c.OrderID)
            .where(orders.c.ShippedDate.is_(None))
            .order_by(orders.c.OrderID)
        )
        with self._engine.connect() as conn:
            return [int(r[0]) for r in conn.execute(stmt)]

    # -- lineas e importes ---------------------------------------------------

    def get_lines(self, order_ids: Sequence[int]) -> dict[int, list[OrderLine]]:
        if not order_ids:
            return {}
        stmt = (
            select(order_details, products.c.ProductName)
            .join(products, products.c.ProductID == order_details.c.ProductID)
            .where(order_details.c.OrderID.in_(list(order_ids)))
            .order_by(order_details.c.OrderID, order_details.c.ProductID)
            .limit(self.max_rows * 5)
        )
        result: dict[int, list[OrderLine]] = {}
        with self._engine.connect() as conn:
            for row in conn.execute(stmt):
                mapping = row._mapping
                line = OrderLine(
                    order_id=int(mapping[order_details.c.OrderID]),
                    product_id=int(mapping[order_details.c.ProductID]),
                    product_name=str(mapping[products.c.ProductName]),
                    unit_price=mapping[order_details.c.UnitPrice],
                    quantity=int(mapping[order_details.c.Quantity]),
                    discount=mapping[order_details.c.Discount],
                )
                result.setdefault(line.order_id, []).append(line)
        return result

    def get_totals(self, order_ids: Sequence[int]) -> dict[int, OrderTotals]:
        lines = self.get_lines(order_ids)
        return {oid: compute_totals(ols) for oid, ols in lines.items() if ols}

    # -- mapeo ---------------------------------------------------------------

    @staticmethod
    def _row_to_customer(row) -> ErpCustomer:  # type: ignore[no-untyped-def]
        mapping = row._mapping
        return ErpCustomer(
            customer_id=str(mapping[customers.c.CustomerID]),
            company_name=str(mapping[customers.c.CompanyName]),
            contact_name=mapping[customers.c.ContactName],
            country=mapping[customers.c.Country],
            city=mapping[customers.c.City],
        )

    @staticmethod
    def _row_to_order(row) -> ErpOrder:  # type: ignore[no-untyped-def]
        mapping = row._mapping
        return ErpOrder(
            order_id=int(mapping[orders.c.OrderID]),
            customer_id=str(mapping[orders.c.CustomerID]),
            order_date=mapping[orders.c.OrderDate],
            required_date=mapping[orders.c.RequiredDate],
            shipped_date=mapping[orders.c.ShippedDate],
            freight=mapping[orders.c.Freight],
            ship_city=mapping[orders.c.ShipCity],
            ship_country=mapping[orders.c.ShipCountry],
        )
