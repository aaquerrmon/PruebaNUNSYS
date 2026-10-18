"""Esquema y seed reproducible del ERP Northwind de demostración.

Datos de DEMOSTRACIÓN: clientes clásicos de Northwind + pedidos sintéticos
etiquetados como tales. Los identificadores de pedido coinciden con los
fixtures del servicio de producción simulado.

El script es idempotente: vuelve a crear las tablas y regenera los datos.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from sqlalchemy import Column, Date, ForeignKey, Integer, MetaData, Numeric, String, Table
from sqlalchemy.engine import Engine
from sqlalchemy.sql import func

metadata = MetaData()

customers = Table(
    "Customers",
    metadata,
    Column("CustomerID", String(5), primary_key=True),
    Column("CompanyName", String(80), nullable=False),
    Column("ContactName", String(60)),
    Column("ContactTitle", String(60)),
    Column("Country", String(40)),
    Column("City", String(40)),
    Column("PostalCode", String(10)),
    Column("Phone", String(30)),
)

orders = Table(
    "Orders",
    metadata,
    Column("OrderID", Integer, primary_key=True),
    Column("CustomerID", String(5), ForeignKey("Customers.CustomerID"), nullable=False),
    Column("OrderDate", Date, nullable=False),
    Column("RequiredDate", Date, nullable=False),
    Column("ShippedDate", Date),
    Column("ShipCity", String(40)),
    Column("ShipCountry", String(40)),
    Column("Freight", Numeric(10, 2), nullable=False, default=0),
    Column("IsSynthetic", Integer, nullable=False, default=0),
)

products = Table(
    "Products",
    metadata,
    Column("ProductID", Integer, primary_key=True),
    Column("ProductName", String(80), nullable=False),
    Column("UnitPrice", Numeric(10, 2), nullable=False),
    Column("Discontinued", Integer, nullable=False, default=0),
)

order_details = Table(
    "OrderDetails",
    metadata,
    Column("OrderID", Integer, ForeignKey("Orders.OrderID"), primary_key=True),
    Column("ProductID", Integer, ForeignKey("Products.ProductID"), primary_key=True),
    Column("UnitPrice", Numeric(10, 2), nullable=False),
    Column("Quantity", Integer, nullable=False),
    Column("Discount", Numeric(5, 4), nullable=False, default=0),
)

# --------------------------------------------------------------------------
# Fixtures de demostración
# --------------------------------------------------------------------------

_FIXTURE_CUSTOMERS = [
    ("ALFKI", "Alfreds Futterkiste", "Maria Anders", "Sales Representative", "Germany", "Berlin"),
    ("ANATR", "Ana Trujillo Emparedados helados", "Ana Trujillo", "Owner", "Mexico", "México D.F."),
    ("ANTON", "Antonio Moreno Taquería", "Antonio Moreno", "Owner", "Mexico", "México D.F."),
    ("BERGS", "Berglunds snabbköp", "Christina Berglund", "Order Administrator", "Sweden", "Luleå"),
    (
        "BLAUS",
        "Blauer See Delikatessen",
        "Hanna Moos",
        "Sales Representative",
        "Germany",
        "Mannheim",
    ),
    ("BONAP", "Bon app'", "Laurence Lebihan", "Owner", "France", "Marseille"),
    (
        "BOTTM",
        "Bottom-Dollar Markets",
        "Elizabeth Lincoln",
        "Accounting Manager",
        "Canada",
        "Tsawassen",
    ),
    (
        "CENTC",
        "Centro comercial Moctezuma",
        "Francisco Chang",
        "Marketing Manager",
        "Mexico",
        "México D.F.",
    ),
    ("CHOPS", "Chop-suey Chinese", "Yang Wang", "Owner", "Switzerland", "Bern"),
    ("CONSH", "Consolidated Holdings", "Elizabeth Brown", "Sales Representative", "UK", "London"),
]

_FIXTURE_PRODUCTS = [
    (1, "Chai", Decimal("18.00")),
    (2, "Chang", Decimal("19.00")),
    (3, "Aniseed Syrup", Decimal("10.00")),
    (4, "Chef Anton's Cajun Seasoning", Decimal("22.00")),
    (5, "Chef Anton's Gumbo Mix", Decimal("21.35")),
    (6, "Grandma's Boysenberry Spread", Decimal("25.00")),
    (7, "Uncle Bob's Organic Dried Pears", Decimal("30.00")),
    (8, "Northwoods Cranberry Sauce", Decimal("40.00")),
    (9, "Mishi Kobe Niku", Decimal("97.00")),
    (11, "Queso Cabrales", Decimal("21.00")),
]


@dataclass(frozen=True, slots=True)
class _OrderFixture:
    order_id: int
    customer_id: str
    order_date: date
    required_date: date
    shipped_date: date | None
    ship_city: str
    ship_country: str
    freight: Decimal
    # líneas: (product_id, unit_price, quantity, discount)
    lines: tuple[tuple[int, Decimal, int, Decimal], ...]


def _build_order_fixtures(today: date) -> list[_OrderFixture]:
    """Pedidos fijos + pedidos del mes en curso (regenerados en cada seed)."""
    this_month_start = today.replace(day=1)
    fixtures: list[_OrderFixture] = [
        # --- ALFKI: casos de la especificación -------------------------------
        # 10643: pendiente, en producción
        _OrderFixture(
            10643,
            "ALFKI",
            today - timedelta(days=45),
            today + timedelta(days=10),
            None,
            "Berlin",
            "Germany",
            Decimal("41.23"),
            ((1, Decimal("18.00"), 10, Decimal("0.05")), (2, Decimal("19.00"), 5, Decimal("0"))),
        ),
        # 10692: completado (enviado)
        _OrderFixture(
            10692,
            "ALFKI",
            today - timedelta(days=80),
            today - timedelta(days=50),
            today - timedelta(days=55),
            "Berlin",
            "Germany",
            Decimal("120.50"),
            ((3, Decimal("10.00"), 6, Decimal("0")),),
        ),
        # 10702: pendiente y BLOQUEADO (falta de componentes)
        _OrderFixture(
            10702,
            "ALFKI",
            today - timedelta(days=30),
            today + timedelta(days=5),
            None,
            "Berlin",
            "Germany",
            Decimal("65.80"),
            ((4, Decimal("22.00"), 8, Decimal("0.10")), (6, Decimal("25.00"), 4, Decimal("0"))),
        ),
        # 10835: completado
        _OrderFixture(
            10835,
            "ALFKI",
            today - timedelta(days=60),
            today - timedelta(days=30),
            today - timedelta(days=32),
            "Berlin",
            "Germany",
            Decimal("30.00"),
            ((7, Decimal("30.00"), 3, Decimal("0")),),
        ),
        # 11011: pendiente y RETRASADO (proveedor de materiales, delay_flag)
        _OrderFixture(
            11011,
            "ALFKI",
            today - timedelta(days=55),
            today - timedelta(days=7),
            None,
            "Berlin",
            "Germany",
            Decimal("88.40"),
            (
                (8, Decimal("40.00"), 5, Decimal("0")),
                (11, Decimal("21.00"), 6, Decimal("0.05")),
            ),
        ),
        # --- Otros clientes con problemas de producción ----------------------
        # 10690: BERGS bloqueado (avería de máquina)
        _OrderFixture(
            10690,
            "BERGS",
            today - timedelta(days=35),
            today + timedelta(days=8),
            None,
            "Luleå",
            "Sweden",
            Decimal("55.10"),
            ((9, Decimal("97.00"), 2, Decimal("0")),),
        ),
        # 10691: BONAP retrasado por producción (error de calidad)
        _OrderFixture(
            10691,
            "BONAP",
            today - timedelta(days=50),
            today - timedelta(days=3),
            None,
            "Marseille",
            "France",
            Decimal("44.60"),
            ((5, Decimal("21.35"), 12, Decimal("0.15")),),
        ),
        # 10693: CHOPS pendiente normal (sin incidencias)
        _OrderFixture(
            10693,
            "CHOPS",
            today - timedelta(days=20),
            today + timedelta(days=15),
            None,
            "Bern",
            "Switzerland",
            Decimal("65.60"),
            ((2, Decimal("19.00"), 7, Decimal("0")),),
        ),
        # 10694: BOTTM pendiente con fecha vencida (retraso por fechas)
        _OrderFixture(
            10694,
            "BOTTM",
            today - timedelta(days=70),
            today - timedelta(days=5),
            None,
            "Tsawassen",
            "Canada",
            Decimal("103.20"),
            ((6, Decimal("25.00"), 6, Decimal("0.05")),),
        ),
        # 10695: ANATR completado
        _OrderFixture(
            10695,
            "ANATR",
            today - timedelta(days=40),
            today - timedelta(days=10),
            today - timedelta(days=12),
            "México D.F.",
            "Mexico",
            Decimal("30.40"),
            ((4, Decimal("22.00"), 4, Decimal("0")),),
        ),
        # 10696: CENTC retrasado por producción (plazo ampliado del proveedor)
        _OrderFixture(
            10696,
            "CENTC",
            today - timedelta(days=48),
            today - timedelta(days=2),
            None,
            "México D.F.",
            "Mexico",
            Decimal("34.90"),
            ((7, Decimal("30.00"), 5, Decimal("0")),),
        ),
        # 10697: CONSH bloqueado (retención de calidad)
        _OrderFixture(
            10697,
            "CONSH",
            today - timedelta(days=25),
            today + timedelta(days=12),
            None,
            "London",
            "UK",
            Decimal("19.70"),
            ((11, Decimal("21.00"), 9, Decimal("0")),),
        ),
        # 10698: BLAUS completado
        _OrderFixture(
            10698,
            "BLAUS",
            today - timedelta(days=15),
            today + timedelta(days=5),
            today - timedelta(days=9),
            "Mannheim",
            "Germany",
            Decimal("22.00"),
            ((1, Decimal("18.00"), 3, Decimal("0")),),
        ),
        # 10699: ANTON pendiente normal
        _OrderFixture(
            10699,
            "ANTON",
            today - timedelta(days=8),
            today + timedelta(days=20),
            None,
            "México D.F.",
            "Mexico",
            Decimal("15.60"),
            ((3, Decimal("10.00"), 8, Decimal("0")),),
        ),
        # --- Pedidos del mes en curso (para "resumen del estado de este mes") -
        _OrderFixture(
            20001,
            "ALFKI",
            this_month_start + timedelta(days=1),
            this_month_start + timedelta(days=30),
            None,
            "Berlin",
            "Germany",
            Decimal("48.00"),
            ((2, Decimal("19.00"), 4, Decimal("0")),),
        ),
        _OrderFixture(
            20002,
            "BERGS",
            this_month_start + timedelta(days=2),
            this_month_start + timedelta(days=25),
            this_month_start + timedelta(days=8),
            "Luleå",
            "Sweden",
            Decimal("72.10"),
            ((8, Decimal("40.00"), 3, Decimal("0")),),
        ),
        _OrderFixture(
            20003,
            "BONAP",
            this_month_start + timedelta(days=3),
            this_month_start + timedelta(days=28),
            None,
            "Marseille",
            "France",
            Decimal("18.90"),
            ((5, Decimal("21.35"), 6, Decimal("0.10")),),
        ),
        _OrderFixture(
            20004,
            "CHOPS",
            this_month_start + timedelta(days=5),
            this_month_start + timedelta(days=20),
            None,
            "Bern",
            "Switzerland",
            Decimal("33.50"),
            ((6, Decimal("25.00"), 4, Decimal("0")),),
        ),
        _OrderFixture(
            20005,
            "BOTTM",
            this_month_start + timedelta(days=6),
            this_month_start + timedelta(days=26),
            this_month_start + timedelta(days=11),
            "Tsawassen",
            "Canada",
            Decimal("27.30"),
            ((7, Decimal("30.00"), 2, Decimal("0")),),
        ),
        _OrderFixture(
            20006,
            "CONSH",
            this_month_start + timedelta(days=7),
            this_month_start + timedelta(days=30),
            None,
            "London",
            "UK",
            Decimal("14.20"),
            ((4, Decimal("22.00"), 3, Decimal("0")),),
        ),
    ]
    return fixtures


def seed(engine: Engine, today: date | None = None) -> dict[str, int]:
    """Crea el esquema e inserta los fixtures. Idempotente (borra y regenera)."""
    today = today or date.today()
    metadata.drop_all(engine)
    metadata.create_all(engine)

    with engine.begin() as conn:
        conn.execute(
            customers.insert(),
            [
                {
                    "CustomerID": cid,
                    "CompanyName": name,
                    "ContactName": contact,
                    "ContactTitle": title,
                    "Country": country,
                    "City": city,
                    "PostalCode": "00000",
                    "Phone": "+34-000-000000",
                }
                for cid, name, contact, title, country, city in _FIXTURE_CUSTOMERS
            ],
        )
        conn.execute(
            products.insert(),
            [
                {
                    "ProductID": pid,
                    "ProductName": name,
                    "UnitPrice": price,
                    "Discontinued": 0,
                }
                for pid, name, price in _FIXTURE_PRODUCTS
            ],
        )

        fixtures = _build_order_fixtures(today)
        conn.execute(
            orders.insert(),
            [
                {
                    "OrderID": fx.order_id,
                    "CustomerID": fx.customer_id,
                    "OrderDate": fx.order_date,
                    "RequiredDate": fx.required_date,
                    "ShippedDate": fx.shipped_date,
                    "ShipCity": fx.ship_city,
                    "ShipCountry": fx.ship_country,
                    "Freight": fx.freight,
                    "IsSynthetic": 1 if fx.order_id >= 20000 else 0,
                }
                for fx in fixtures
            ],
        )
        detail_rows: list[dict[str, object]] = []
        for fx in fixtures:
            for product_id, unit_price, quantity, discount in fx.lines:
                detail_rows.append(
                    {
                        "OrderID": fx.order_id,
                        "ProductID": product_id,
                        "UnitPrice": unit_price,
                        "Quantity": quantity,
                        "Discount": discount,
                    }
                )
        conn.execute(order_details.insert(), detail_rows)

    with engine.connect() as conn:
        from sqlalchemy import select

        order_count = len(conn.execute(select(orders.c.OrderID)).fetchall())
        detail_count = len(conn.execute(select(order_details.c.OrderID)).fetchall())
        customer_count = len(conn.execute(select(customers.c.CustomerID)).fetchall())
    return {
        "customers": customer_count,
        "orders": order_count,
        "order_details": detail_count,
    }


def seed_if_configured(engine: Engine) -> dict[str, int]:
    """Utilidad para scripts: si la base no existe o está vacía, la siembra."""
    from sqlalchemy import inspect

    if not inspect(engine).has_table("Orders"):
        return seed(engine)
    with engine.connect() as conn:
        from sqlalchemy import select

        row = conn.execute(select(func.count()).select_from(orders)).scalar()
    if not row:
        return seed(engine)
    return {"customers": -1, "orders": int(row), "order_details": -1}


def default_db_path() -> Path:
    return Path("data/erp/northwind.db")


if __name__ == "__main__":  # pragma: no cover
    from agentic_poc.infrastructure.database import make_writable_engine

    url = f"sqlite:///{default_db_path().resolve()}"
    stats = seed(make_writable_engine(url))
    print(f"Seed completado: {stats}")
