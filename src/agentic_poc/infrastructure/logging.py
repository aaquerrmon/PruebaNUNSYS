"""Logging estructurado con correlación por ``trace_id``.

- Formato ``text`` para desarrollo, ``json`` para máquinas.
- Los logs operativos se separan de las evidencias que se devuelven al cliente.
- Nunca se registran claves API, tokens ni el contenido completo de la pregunta
  del usuario salvo que ``PRIVACY_LOG_QUESTIONS=true`` (queda truncado + hash).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)

_RESERVED_LOG_KEYS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)


class JsonFormatter(logging.Formatter):
    """Formatea los registros como una línea JSON (un evento por línea)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace_id = trace_id_var.get()
        if trace_id:
            payload["trace_id"] = trace_id
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_KEYS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Formateo legible para desarrollo, con trace_id cuando existe."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        trace_id = trace_id_var.get()
        return f"[{trace_id}] {base}" if trace_id else base


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Configura el logger raíz una sola vez (idempotente)."""
    handler = logging.StreamHandler(sys.stdout)
    if fmt.lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TextFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Los clientes HTTP de terceros muy ruidosos se dejan en WARNING
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def question_fingerprint(question: str, *, keep: int = 120) -> dict[str, str]:
    """Huella de la pregunta para logs: truncado + SHA-256, nunca el texto íntegro."""
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    return {"question_truncated": question[:keep], "question_sha256_16": digest}


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
