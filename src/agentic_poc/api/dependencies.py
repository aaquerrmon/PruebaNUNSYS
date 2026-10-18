"""Dependencias FastAPI (singletones por request/lifecycle).

El `RagService`, el `ConversationStore` y los embeddings se crean una sola vez
por proceso (singleton perezoso) para evitar que cada petición abra una nueva
colección Chroma sobre el mismo `persist_directory` (causa de tablas
duplicadas y fallos del vector store) y para garantizar que las tablas de
memoria se creen una vez.
"""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy.engine import Engine

from agentic_poc.agents.orchestrator import AgentOrchestrator
from agentic_poc.api.memory import ConversationStore
from agentic_poc.config import Settings, get_settings
from agentic_poc.infrastructure.database import make_app_engine, make_readonly_engine
from agentic_poc.repositories.erp_repository import ErpRepository
from agentic_poc.services.embeddings import LocalE5Embeddings
from agentic_poc.services.production_client import ProductionClient
from agentic_poc.services.rag import RagService

_ENGINE_APP: Engine | None = None
_ENGINE_ERP: Engine | None = None
_CONVERSATION_STORE: ConversationStore | None = None
_EMBEDDINGS: LocalE5Embeddings | None = None
_RAG_SERVICE: RagService | None = None


def clear_caches() -> None:
    """Reinicia los singletones (usado por los tests para aislar cada prueba)."""
    global _ENGINE_APP, _ENGINE_ERP, _CONVERSATION_STORE, _EMBEDDINGS, _RAG_SERVICE
    if _ENGINE_APP is not None:
        _ENGINE_APP.dispose()
        _ENGINE_APP = None
    _ENGINE_ERP = None
    _CONVERSATION_STORE = None
    _EMBEDDINGS = None
    _RAG_SERVICE = None
    get_settings.cache_clear()


def _get_engine_app() -> Engine:
    global _ENGINE_APP
    if _ENGINE_APP is None:
        _ENGINE_APP = make_app_engine(get_settings().app_db_path)
    return _ENGINE_APP


def _get_engine_erp() -> Engine:
    global _ENGINE_ERP
    if _ENGINE_ERP is None:
        _ENGINE_ERP = make_readonly_engine(get_settings().erp_database_url)
    return _ENGINE_ERP


def _get_embeddings() -> LocalE5Embeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        settings = get_settings()
        _EMBEDDINGS = LocalE5Embeddings(
            settings.embeddings_base_url,
            model=settings.embeddings_model,
            api_key=settings.embeddings_api_key,
            dimensions=settings.embeddings_dimensions,
            timeout_seconds=settings.embeddings_timeout_seconds,
            batch_size=settings.embeddings_batch_size,
            passage_prefix=settings.embeddings_passage_prefix,
            query_prefix=settings.embeddings_query_prefix,
        )
    return _EMBEDDINGS


def _get_rag_service() -> RagService:
    global _RAG_SERVICE
    if _RAG_SERVICE is None:
        settings = get_settings()
        _RAG_SERVICE = RagService(
            _get_embeddings(),
            persist_directory=settings.vector_store_path,
            top_k=settings.rag_top_k,
            score_threshold=settings.rag_score_threshold,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            max_pdf_bytes=settings.max_pdf_bytes,
            max_pdf_pages=settings.max_pdf_pages,
        )
    return _RAG_SERVICE


def _get_conversation_store() -> ConversationStore:
    global _CONVERSATION_STORE
    if _CONVERSATION_STORE is None:
        _CONVERSATION_STORE = ConversationStore(_get_engine_app())
    return _CONVERSATION_STORE


def get_repository(settings: Settings = Depends(get_settings)) -> ErpRepository:
    return ErpRepository(_get_engine_erp(), max_rows=settings.erp_max_rows)


def get_production_client(settings: Settings = Depends(get_settings)) -> ProductionClient:
    return ProductionClient(
        settings.production_api_url,
        timeout_seconds=settings.production_timeout_seconds,
        max_retries=settings.production_max_retries,
    )


def get_embeddings(settings: Settings = Depends(get_settings)) -> LocalE5Embeddings:
    return _get_embeddings()


def get_rag_service(
    settings: Settings = Depends(get_settings),
    embeddings: LocalE5Embeddings = Depends(get_embeddings),
) -> RagService:
    return _get_rag_service()


def get_conversation_store(settings: Settings = Depends(get_settings)) -> ConversationStore:
    return _get_conversation_store()


def get_orchestrator(
    settings: Settings = Depends(get_settings),
    repository: ErpRepository = Depends(get_repository),
    production: ProductionClient = Depends(get_production_client),
    rag: RagService = Depends(get_rag_service),
) -> AgentOrchestrator:
    return AgentOrchestrator(
        settings=settings,
        repository=repository,
        production=production,
        rag=rag,
    )
