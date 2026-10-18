"""Acceso a SQLite: motor del ERP (solo lectura en la aplicación) y de la app.

El script de seed abre la base en lectura-escritura; la aplicación solo lee
(modo ``mode=ro`` de SQLite + ``PRAGMA query_only``), de modo que ninguna
consulta del agente puede modificar el ERP.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.pool import StaticPool


def _sqlite_path(database_url: str) -> Path | None:
    """Devuelve la ruta local de un ``sqlite:///`` o ``None`` si no es SQLite."""
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite":
        return None
    if url.database in (None, "", ":memory:"):
        return None
    # Con URI (file:...) el path puede venir ya con el esquema
    raw = url.database
    if raw.startswith("file:"):
        raw = raw[len("file:") :]
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def make_readonly_engine(database_url: str, *, timeout_seconds: float = 10.0) -> Engine:
    """Motor SQLite de solo lectura con ``busy_timeout`` y ``query_only``."""
    path = _sqlite_path(database_url)
    if path is None:
        raise ValueError(
            f"URL de base de datos no soportada en modo solo lectura: {database_url!r}"
        )
    # El dialecho pysqlite reconoce `uri=true` en la URL, lo retira y pasa
    # `uri=True` a sqlite3.connect; el resto de parámetros van a la URI.
    uri = f"file:{path}?mode=ro&uri=true"
    engine = create_engine(
        f"sqlite:///{uri}",
        connect_args={"timeout": timeout_seconds, "check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _configure(dbapi_connection, _record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA query_only = ON")
        cursor.execute(f"PRAGMA busy_timeout = {int(timeout_seconds * 1000)}")
        cursor.close()

    return engine


def make_writable_engine(database_url: str, *, timeout_seconds: float = 10.0) -> Engine:
    """Motor SQLite de lectura-escritura (solo para seed/scripts)."""
    path = _sqlite_path(database_url)
    if path is None:
        # memoria u otras URLs: motor normal
        return create_engine(database_url, future=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        f"sqlite:///{path}",
        connect_args={"timeout": timeout_seconds, "check_same_thread": False},
        future=True,
    )


def make_app_engine(app_db_path: Path, *, timeout_seconds: float = 10.0) -> Engine:
    """Motor de la aplicación (memoria conversacional y trazas), lectura/escritura propio."""
    app_db_path = Path(app_db_path)
    app_db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        f"sqlite:///{app_db_path.resolve()}",
        connect_args={"timeout": timeout_seconds, "check_same_thread": False},
        future=True,
    )
