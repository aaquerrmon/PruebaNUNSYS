"""Modelos de respuesta de la API (contrato público)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Referencia verificable a un fragmento de documento usado en la respuesta."""

    doc_id: str
    source: str
    page: int | None = None
    snippet: str
    score: float


class ExecutionStep(BaseModel):
    """Paso realmente ejecutado por el agente (fuente, herramienta, resultado)."""

    tool: str
    source: str | None = None
    args_summary: str = ""
    record_count: int | None = None
    duration_ms: float = 0.0
    ok: bool = True
    error: str | None = None


class QueryResponse(BaseModel):
    """Respuesta estructurada y trazable a una pregunta en lenguaje natural."""

    answer: str
    sources: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    execution_trace: list[ExecutionStep] = Field(default_factory=list)
    trace_id: str
    duration_ms: float
    model: str
    used_memory: bool = False
    doc_ids: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    """Petición del endpoint /api/query."""

    question: str = Field(min_length=3, max_length=2000)
    conversation_id: str | None = Field(
        default=None,
        description="Identificador de conversación para memoria multi-turno (opcional).",
    )


class IngestResponse(BaseModel):
    doc_id: str
    source: str
    pages: int
    chunks: int
    skipped_pages: int


class DocumentSummary(BaseModel):
    doc_id: str
    source: str
    chunks: int
    pages: int


class HealthResponse(BaseModel):
    status: str
    llm: str
    embeddings: str
    production_api: str
    erp: str
    documents: int
