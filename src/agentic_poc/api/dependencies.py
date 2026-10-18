"""Dependencias FastAPI (singletones por request/lifecycle)."""

from __future__ import annotations

from functools import lru_cache

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


@lru_cache(maxsize=1)
def _get_engine_app() -> Engine:
    return make_app_engine(get_settings().app_db_path)


@lru_cache(maxsize=1)
def _get_engine_erp() -> Engine:
    return make_readonly_engine(get_settings().erp_database_url)


def get_repository(settings: Settings = Depends(get_settings)) -> ErpRepository:
    return ErpRepository(_get_engine_erp(), max_rows=settings.erp_max_rows)


def get_production_client(settings: Settings = Depends(get_settings)) -> ProductionClient:
    return ProductionClient(
        settings.production_api_url,
        timeout_seconds=settings.production_timeout_seconds,
        max_retries=settings.production_max_retries,
    )


def get_embeddings(settings: Settings = Depends(get_settings)) -> LocalE5Embeddings:
    return LocalE5Embeddings(
        settings.embeddings_base_url,
        model=settings.embeddings_model,
        api_key=settings.embeddings_api_key,
        dimensions=settings.embeddings_dimensions,
        timeout_seconds=settings.embeddings_timeout_seconds,
        batch_size=settings.embeddings_batch_size,
        passage_prefix=settings.embeddings_passage_prefix,
        query_prefix=settings.embeddings_query_prefix,
    )


def get_rag_service(
    settings: Settings = Depends(get_settings),
    embeddings: LocalE5Embeddings = Depends(get_embeddings),
) -> RagService:
    return RagService(
        embeddings,
        persist_directory=settings.vector_store_path,
        top_k=settings.rag_top_k,
        score_threshold=settings.rag_score_threshold,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        max_pdf_bytes=settings.max_pdf_bytes,
        max_pdf_pages=settings.max_pdf_pages,
    )


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


def get_conversation_store(settings: Settings = Depends(get_settings)) -> ConversationStore:
    return ConversationStore(_get_engine_app())
