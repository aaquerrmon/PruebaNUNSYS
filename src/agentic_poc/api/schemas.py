"""Esquemas Pydantic para la API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    conversation_id: str | None = Field(default=None)


class CitationResponse(BaseModel):
    doc_id: str
    source: str
    page: int | None = None
    snippet: str
    score: float


class ExecutionStepResponse(BaseModel):
    tool: str
    source: str | None = None
    args_summary: str = ""
    record_count: int | None = None
    duration_ms: float = 0.0
    ok: bool = True
    error: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    citations: list[CitationResponse] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    execution_trace: list[ExecutionStepResponse] = Field(default_factory=list)
    trace_id: str
    duration_ms: float
    model: str
    used_memory: bool = False


class SSEEvent(BaseModel):
    event: str
    data: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    llm: str = "unknown"
    embeddings: str = "unknown"
    production_api: str = "unknown"
    erp: str = "unknown"
    documents: int = 0


class ConversationMessage(BaseModel):
    role: str
    content: str
    ts: float


class ConversationHistory(BaseModel):
    conversation_id: str
    messages: list[ConversationMessage] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


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
