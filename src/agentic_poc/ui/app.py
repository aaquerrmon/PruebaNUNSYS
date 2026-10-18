"""Interfaz Chainlit para la POC agéntica.

La UI no habla con el LLM directamente: llama siempre a la API FastAPI
(``POST /api/query``), de modo que la trazabilidad y la memoria viven en un
único punto y la UI es solo un presentador.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import chainlit as cl
import httpx

from agentic_poc.api.schemas import QueryRequest
from agentic_poc.config import get_settings

logger = logging.getLogger(__name__)

_STARTERS = [
    "Resume el estado del pedido 10643",
    "¿Qué pedidos de ALFKI están pendientes?",
    "¿Cuánto factura la orden 10692?",
    "¿Qué dice el contrato sobre los plazos de entrega?",
    "¿Cuál es el motivo de retraso de los pedidos retenidos en producción?",
]


@cl.on_chat_start
async def on_chat_start() -> None:
    settings = get_settings()
    settings.assert_local_inference()
    cl.user_session.set("conversation_id", uuid.uuid4().hex)
    cl.user_session.set("client", httpx.AsyncClient(base_url=settings.backend_url, timeout=120.0))

    status = await _health(settings.backend_url)
    await cl.Message(
        content=(
            "## POC Agéntica NUNSYS\n"
            "Asistente con inferencia **local** (vLLM) sobre el ERP Northwind, "
            "la API de producción y la documentación en PDF.\n\n"
            f"Estado de servicios: {status}"
        ),
        author="sistema",
    ).send()

    cl.set_starters(_make_starters)


async def _make_starters(user: cl.User | None = None) -> list[cl.Starter]:
    return [cl.Starter(label=s, message=s) for s in _STARTERS]


@cl.on_chat_end
async def on_chat_end() -> None:
    client: httpx.AsyncClient | None = cl.user_session.get("client")
    if client:
        await client.aclose()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    client: httpx.AsyncClient | None = cl.user_session.get("client")
    conversation_id: str = cl.user_session.get("conversation_id", "")
    if client is None:
        await cl.Message(
            content="Sesión no inicializada; recarga la página.", author="sistema"
        ).send()
        return

    answer_msg = cl.Message(content="", author="asistente")
    await answer_msg.send()

    try:
        payload = QueryRequest(question=message.content, conversation_id=conversation_id)
        async with client.stream(
            "POST",
            "/api/query/stream",
            json=payload.model_dump(),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = _parse_sse(line[6:])
                if event is None:
                    continue
                await _handle_event(event, answer_msg)
    except httpx.HTTPStatusError as exc:
        answer_msg.content = f"Error HTTP {exc.response.status_code} al consultar la API."
        await answer_msg.update()
    except httpx.HTTPError as exc:
        logger.warning("error de red: %s", exc)
        answer_msg.content = f"No pude alcanzar la API local ({exc.__class__.__name__})."
        await answer_msg.update()


def _parse_sse(raw: str) -> dict[str, Any] | None:
    try:
        import json

        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


async def _handle_event(event: dict[str, Any], answer_msg: cl.Message) -> None:
    kind = event.get("event")
    data = event.get("data") or {}

    if kind == "start":
        return

    if kind == "answer" and isinstance(data, dict):
        answer_msg.content = str(data.get("answer") or "")
        extras: list[str] = []

        reasoning = data.get("reasoning")
        if reasoning:
            step = cl.Step(name="Razonamiento", type="llm")
            async with step:
                step.output = "\n".join(f"- {r}" for r in reasoning)

        sources = data.get("sources") or []
        if sources:
            extras.append("**Fuentes consultadas:** " + ", ".join(str(s) for s in sources))

        warnings = data.get("warnings") or []
        if warnings:
            extras.append("**Advertencias:** " + "; ".join(str(w) for w in warnings))

        trace = data.get("execution_trace") or []
        if trace:
            step = cl.Step(name="Traza de ejecución", type="run")
            async with step:
                lines = []
                for i, entry in enumerate(trace, start=1):
                    if isinstance(entry, dict):
                        name = entry.get("tool", f"paso {i}")
                        state = (
                            "ok" if entry.get("ok", True) else f"error: {entry.get('error', '?')}"
                        )
                        count = entry.get("record_count")
                        detail = f" ({count} registros)" if count is not None else ""
                        duration = entry.get("duration_ms")
                        suffix = f" — {duration:.0f} ms" if duration else ""
                        lines.append(f"{i}. {name}: {state}{detail}{suffix}")
                    else:
                        lines.append(f"{i}. {entry}")
                step.output = "\n".join(lines)

        meta = []
        if data.get("trace_id"):
            meta.append(f"trace `{data['trace_id']}`")
        if data.get("duration_ms") is not None:
            meta.append(f"{data['duration_ms']:.0f} ms")
        if meta:
            extras.append(" · ".join(meta))

        if extras:
            answer_msg.content = (answer_msg.content + "\n\n---\n" + "\n\n".join(extras)).strip()
        await answer_msg.update()
        return

    if kind == "error":
        answer_msg.content = f"Error: {data.get('error', 'desconocido')}"
        await answer_msg.update()


async def _health(base_url: str) -> str:
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
            response = await client.get("/health")
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return f"API **no disponible** ({exc.__class__.__name__})."
    parts = [f"{k}: {v}" for k, v in payload.items() if k != "status"]
    estado = payload.get("status", "desconocido")
    return f"`{estado}` — " + ", ".join(parts)
