"""API FastAPI para consultar el agente multi-fuente.

Endpoints:
- GET /health: estado de dependencias (LLM, embeddings, producción, ERP, documentos).
- POST /api/query: responde en formato JSON {answer, sources, reasoning, ...}.
- POST /api/query/stream: SSE de eventos (paso de herramienta, pensamiento resumen, respuesta).
- GET /api/conversations/{id}/history: historia reciente para memoria.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from agentic_poc.agents.orchestrator import AgentOrchestrator
from agentic_poc.api.dependencies import (
    ConversationStore,
    _get_conversation_store,
    get_conversation_store,
    get_orchestrator,
    get_production_client,
    get_rag_service,
    get_repository,
    get_settings,
)
from agentic_poc.api.memory import serialize_history
from agentic_poc.api.schemas import (
    ConversationHistory,
    DocumentSummary,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SSEEvent,
)
from agentic_poc.config import Settings
from agentic_poc.repositories.erp_repository import ErpRepository
from agentic_poc.services.production_client import ProductionClient
from agentic_poc.services.rag import RagService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    settings.assert_local_inference()
    # Crear tablas de memoria conversacional al arrancar (una sola vez por proceso)
    _get_conversation_store()
    app.state.ready = True
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="POC Agente IA (ERP + Producción + RAG)",
        version="0.1.0",
        description="Inferencia 100 % local con Deep Agents sobre vLLM",
        lifespan=lifespan,
        openapi_url=None if settings.is_production else "/openapi.json",
        docs_url=None if settings.is_production else "/docs",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = create_app()


@app.get("/health", response_model=HealthResponse)
async def health(
    settings: Settings = Depends(get_settings),
    repository: ErpRepository = Depends(get_repository),
    production: ProductionClient = Depends(get_production_client),
    rag: RagService = Depends(get_rag_service),
) -> HealthResponse:
    from agentic_poc.services.llm import probe_endpoint

    def _probe(url: str) -> str:
        try:
            probe_endpoint(url)
        except Exception as exc:
            logger.warning("health probe error (%s): %s", url, exc)
            return "fail"
        return "ok"

    llm_status = _probe(settings.llm_base_url)
    embeddings_status = _probe(settings.embeddings_base_url)

    try:
        production_status = "ok" if await production.health() else "fail"
    except Exception as exc:
        logger.warning("health production error: %s", exc)
        production_status = "fail"

    try:
        repository.get_orders(limit=1)
        erp_status = "ok"
    except Exception as exc:
        logger.warning("health erp error: %s", exc)
        erp_status = "fail"

    try:
        documents_count = len(rag.list_documents())
    except Exception as exc:
        logger.warning("health docs error: %s", exc)
        documents_count = 0

    degraded = "fail" in (llm_status, erp_status, production_status)
    return HealthResponse(
        status="degraded" if degraded else "ok",
        llm=llm_status,
        embeddings=embeddings_status,
        production_api=production_status,
        erp=erp_status,
        documents=documents_count,
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    store: ConversationStore = Depends(get_conversation_store),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    history = []
    if request.conversation_id:
        history = await store.get_history(
            request.conversation_id, limit=settings.memory_max_turns * 2
        )
    response = await orchestrator.answer(request.question, history, doc_ids=request.doc_ids)
    if request.conversation_id:
        await store.save_turn(request.conversation_id, request.question, response.answer)
    # map to API schema
    from agentic_poc.api.schemas import CitationResponse, ExecutionStepResponse
    from agentic_poc.api.schemas import QueryResponse as ApiQueryResponse

    return ApiQueryResponse(
        answer=response.answer,
        sources=response.sources,
        reasoning=response.reasoning,
        data=response.data,
        citations=[CitationResponse(**c.model_dump()) for c in response.citations],
        warnings=response.warnings,
        execution_trace=[ExecutionStepResponse(**e.model_dump()) for e in response.execution_trace],
        trace_id=response.trace_id,
        duration_ms=response.duration_ms,
        model=response.model,
        used_memory=response.used_memory,
        doc_ids=response.doc_ids,
    )


@app.post("/api/query/stream")
async def query_stream(
    request: QueryRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    store: ConversationStore = Depends(get_conversation_store),
    settings: Settings = Depends(get_settings),
) -> EventSourceResponse:
    async def event_generator() -> AsyncGenerator[str, None]:
        yield SSEEvent(event="start", data={"question": request.question}).model_dump_json()
        history = []
        if request.conversation_id:
            history = await store.get_history(
                request.conversation_id, limit=settings.memory_max_turns * 2
            )
        try:
            response = await orchestrator.answer(request.question, history, doc_ids=request.doc_ids)
            if request.conversation_id:
                await store.save_turn(request.conversation_id, request.question, response.answer)
            yield SSEEvent(event="answer", data=response.model_dump()).model_dump_json()
            yield SSEEvent(event="done", data={"trace_id": response.trace_id}).model_dump_json()
        except Exception as exc:
            logger.warning("query stream error: %s", exc)
            yield SSEEvent(event="error", data={"error": str(exc)}).model_dump_json()

    return EventSourceResponse(event_generator())


@app.get("/api/conversations/{conversation_id}/history", response_model=ConversationHistory)
async def get_conversation_history(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversation_store),
) -> ConversationHistory:
    messages = await store.get_history(conversation_id)
    history_msgs = serialize_history(messages)
    return ConversationHistory(conversation_id=conversation_id, messages=history_msgs)


@app.post("/api/documents", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    settings: Settings = Depends(get_settings),
    rag: RagService = Depends(get_rag_service),
) -> IngestResponse:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Solo se admiten archivos PDF")
    content = await file.read()
    if len(content) > settings.max_pdf_bytes:
        raise HTTPException(status_code=413, detail="El PDF supera el tamano maximo")
    documents_dir = Path(settings.documents_dir)
    documents_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "documento.pdf").name
    target = documents_dir / safe_name
    target.write_bytes(content)
    try:
        result = rag.add_pdf(target)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return IngestResponse(
        doc_id=result.doc_id,
        source=result.source,
        pages=result.pages,
        chunks=result.chunks,
        skipped_pages=result.skipped_pages,
    )


@app.get("/api/documents", response_model=list[DocumentSummary])
async def list_documents(rag: RagService = Depends(get_rag_service)) -> list[DocumentSummary]:
    # Construcción explícita: DocumentInfo es un dataclass con slots (sin __dict__).
    return [
        DocumentSummary(doc_id=d.doc_id, source=d.source, chunks=d.chunks, pages=d.pages)
        for d in rag.list_documents()
    ]
