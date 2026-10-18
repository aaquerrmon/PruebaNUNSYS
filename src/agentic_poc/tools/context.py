"""Contexto compartido por las herramientas de una petición.

Cada petición construye su propio `ToolContext` con un `EvidenceCollector`
nuevo. Las herramientas cierran sobre este contexto, de modo que la evidencia
queda ligada a la petición sin usar `contextvars` (las tools sincrónicas se
ejecutan en hilos del executor).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from agentic_poc.models.domain import EvidenceCollector, SourceName, ToolObservation
from agentic_poc.repositories.erp_repository import ErpRepository
from agentic_poc.services.business import BusinessService
from agentic_poc.services.production_client import ProductionClient
from agentic_poc.services.rag import RagService


@dataclass
class ToolContext:
    repository: ErpRepository
    production: ProductionClient
    rag: RagService
    collector: EvidenceCollector = field(default_factory=EvidenceCollector)
    today: date = field(default_factory=date.today)
    doc_ids: list[str] | None = None

    @property
    def business(self) -> BusinessService:
        return BusinessService(self.repository, self.production, today=self.today)


def dumps(payload: Any) -> str:
    """Serializa la respuesta de una herramienta (el modelo lee este texto)."""
    return json.dumps(payload, ensure_ascii=False, default=str)


class ToolRecorder:
    """Context manager que mide, registra y protege una llamada a herramienta.

    Uso::

        with ToolRecorder(ctx, "get_blocked_orders", source="ERP", args="") as rec:
            ...
            rec.record_count = len(result)
            rec.record_ids = [...]
            return dumps(result)
    """

    def __init__(
        self,
        context: ToolContext,
        tool: str,
        *,
        source: SourceName | None,
        args: str = "",
    ) -> None:
        self._context = context
        self._tool = tool
        self._source = source
        self._args = args
        self._start = 0.0
        self.record_count: int | None = None
        self.record_ids: list[str] = []
        self.error: str | None = None

    def __enter__(self) -> ToolRecorder:
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
        duration_ms = round((time.perf_counter() - self._start) * 1000, 2)
        if exc is not None:
            self.error = f"{exc_type.__name__ if exc_type else 'Error'}: {exc}"
        self._context.collector.record_tool(
            ToolObservation(
                tool=self._tool,
                source=self._source,
                args_summary=self._args,
                record_count=self.record_count,
                duration_ms=duration_ms,
                ok=exc is None,
                error=self.error,
                record_ids=self.record_ids[:20],
            )
        )
        return exc is not None
