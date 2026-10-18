"""Orquestador que construye el agente, invoca la pregunta y compone la respuesta.

Encapsula la creación del contexto, herramientas, registro del perfil del
modelo y el paso determinista de composición de la respuesta estructurada.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import date

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agentic_poc.agents.agent_factory import build_agent, register_local_profile
from agentic_poc.agents.composer import compose_response
from agentic_poc.config import Settings
from agentic_poc.models.domain import EvidenceCollector, ToolObservation
from agentic_poc.models.responses import QueryResponse
from agentic_poc.repositories.erp_repository import ErpRepository
from agentic_poc.services.llm import create_chat_model
from agentic_poc.services.production_client import ProductionClient
from agentic_poc.services.rag import RagService
from agentic_poc.tools.context import ToolContext
from agentic_poc.tools.domain_tools import build_tools


def context_today_factory(settings: Settings) -> Callable[[], date]:
    return date.today


logger = logging.getLogger(__name__)


class AgentExecutionError(Exception):
    """Fallo no recuperable al ejecutar el agente (LLM no disponible, etc.)."""


def _last_ai_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        parts.append(str(block.get("text", "")))
                    else:
                        parts.append(str(block))
                return "\n".join(parts).strip()
    return ""


class AgentOrchestrator:
    """Fachada para ejecutar una pregunta multi-fuente con Deep Agents."""

    def __init__(
        self,
        settings: Settings,
        repository: ErpRepository,
        production: ProductionClient,
        rag: RagService,
        *,
        chat_model: ChatOpenAI | None = None,
        recursion_limit: int = 30,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._production = production
        self._rag = rag
        self._chat_model = chat_model
        self._recursion_limit = recursion_limit

        register_local_profile(self._settings.llm_model)

    @property
    def settings(self) -> Settings:
        """Configuración con la que opera (lectura; la usan tests y diagnóstico)."""
        return self._settings

    @property
    def rag(self) -> RagService:
        """Servicio RAG del orquestador (mismo al que acceden sus herramientas)."""
        return self._rag

    async def answer(
        self,
        question: str,
        history: Sequence[BaseMessage] | None = None,
        doc_ids: list[str] | None = None,
    ) -> QueryResponse:
        started = time.perf_counter()
        history = tuple(history or ())
        trace_id = uuid.uuid4().hex

        model = create_chat_model(self._settings) if self._chat_model is None else self._chat_model

        collector = EvidenceCollector()
        context = ToolContext(
            repository=self._repository,
            production=self._production,
            rag=self._rag,
            collector=collector,
            today=context_today_factory(self._settings)(),
            doc_ids=doc_ids,
        )

        tools = build_tools(context)
        agent = build_agent(model, tools, model_name=self._settings.llm_model)

        messages: list[BaseMessage] = []
        messages.extend(history)
        messages.append(HumanMessage(content=question))

        draft = ""
        try:
            result = await agent.ainvoke(
                {"messages": messages}, config={"recursion_limit": self._recursion_limit}
            )
            draft = _last_ai_text(result.get("messages", []))
        except Exception as exc:
            collector.record_tool(
                ToolObservation(
                    tool="agente_llm",
                    source=None,
                    args_summary=question[:120],
                    record_count=0,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                    ok=False,
                    error=str(exc),
                )
            )
            logger.exception("Fallo ejecutando el agente: %s", exc)
            draft = ""

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response = compose_response(
            question=question,
            draft=draft,
            collector=collector,
            trace_id=trace_id,
            duration_ms=duration_ms,
            model=self._settings.llm_model,
            used_memory=bool(history),
            doc_ids=doc_ids,
        )
        return response
