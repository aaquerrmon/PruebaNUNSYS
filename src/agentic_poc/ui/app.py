"""Interfaz Chainlit para la POC agéntica.

La UI no habla con el LLM directamente: llama siempre a la API FastAPI
(``POST /api/query`` y ``POST /api/documents``), de modo que la trazabilidad,
la memoria y la gestion de documentos viven en un unico punto y la UI es solo
un presentador.

Diseno visual (aproximacion 1:3 dentro de las posibilidades de Chainlit):

- Panel superior colapsable "Documentos indexados": lista de PDFs con
  acciones de seleccion/deseleccion para filtrar el RAG por documento.
- Area de chat principal: mensajes, razonamiento, traza y citas.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
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
    # Por defecto no hay filtro de documentos (None = buscar en todos).
    cl.user_session.set("selected_doc_ids", [])
    cl.user_session.set("client", httpx.AsyncClient(base_url=settings.backend_url, timeout=120.0))

    status = await _health(settings.backend_url)

    welcome = (
        "## POC Agéntica NUNSYS\n"
        "Asistente con inferencia **local** (vLLM) sobre el ERP Northwind, "
        "la API de producción y la documentación en PDF.\n\n"
        f"Estado de servicios: {status}\n\n"
        "---\n"
        "### Documentos indexados (RAG)\n"
        "• Arrastra un PDF al chat o adjuntalo para indexarlo en el vector store.\n"
        "• Usa el panel de documentos de abajo para seleccionar cuales quieres que "
        "el chatbot consulte.\n"
        "• Escribe `/documentos` para ver la lista completa.\n"
        "• Luego pregunta sobre su contenido (plazos, penalizaciones, SLA, etc.)."
    )
    await cl.Message(content=welcome, author="sistema").send()

    # Mostrar el panel de documentos al iniciar
    await _show_document_panel()

    cl.set_starters(_make_starters)


@cl.on_chat_end
async def on_chat_end() -> None:
    client: httpx.AsyncClient | None = cl.user_session.get("client")
    if client:
        await client.aclose()


@cl.on_file_upload
async def on_file_upload(files: list[cl.File]) -> None:
    """Gestiona la subida de PDF arrastrados/adjuntos al chat."""
    client: httpx.AsyncClient | None = cl.user_session.get("client")
    if client is None:
        await cl.Message(
            content="Sesión no inicializada; recarga la página.", author="sistema"
        ).send()
        return

    results: list[str] = []

    for uploaded in files:
        if not uploaded.name.lower().endswith(".pdf"):
            results.append(f"❌ `{uploaded.name}`: no es un PDF")
            continue

        try:
            file_path = Path(uploaded.path) if uploaded.path else None
            content = file_path.read_bytes() if file_path else b""
            if not content:
                results.append(f"❌ `{uploaded.name}`: no se pudo leer el contenido")
                continue

            response = await client.post(
                "/api/documents",
                files={"file": (uploaded.name, content, "application/pdf")},
                timeout=60.0,
            )
            if response.status_code in (200, 201):
                body = response.json()
                results.append(
                    f"✅ `{uploaded.name}` indexado (doc_id=`{body['doc_id']}`, "
                    f"{body['chunks']} fragmentos, {body['pages']} páginas)"
                )
            else:
                detail = response.json().get("detail", "error desconocido")
                results.append(f"❌ `{uploaded.name}`: {detail} (HTTP {response.status_code})")
        except Exception as exc:
            logger.warning("error subiendo %s: %s", uploaded.name, exc)
            results.append(f"❌ `{uploaded.name}`: error de red ({exc.__class__.__name__})")

    await cl.Message(
        content="\n".join(results) if results else "Sin archivos para procesar.",
        author="sistema",
    ).send()

    # Refrescar el panel de documentos tras la subida
    await _show_document_panel()


@cl.action_callback("select_document")
async def on_select_document(action: cl.Action) -> None:
    """Selecciona o desecciona un documento para filtrar el RAG."""
    doc_id = action.value
    selected: list[str] = cl.user_session.get("selected_doc_ids", [])
    if doc_id in selected:
        selected.remove(doc_id)
        action.label = f"⬜ {doc_id}"
    else:
        selected.append(doc_id)
        action.label = f"✅ {doc_id}"
    cl.user_session.set("selected_doc_ids", selected)
    await action.update()
    await _show_document_panel()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    client: httpx.AsyncClient | None = cl.user_session.get("client")
    conversation_id: str = cl.user_session.get("conversation_id", "")
    if client is None:
        await cl.Message(
            content="Sesión no inicializada; recarga la página.", author="sistema"
        ).send()
        return

    # Comando interno: listar documentos indexados
    if message.content.strip() == "/documentos":
        await _list_documents(client)
        return

    # Documentos seleccionados para filtrar el RAG (vacío = todos)
    selected_doc_ids: list[str] = cl.user_session.get("selected_doc_ids", [])
    doc_ids: list[str] | None = selected_doc_ids if selected_doc_ids else None

    answer_msg = cl.Message(content="", author="asistente")
    await answer_msg.send()

    try:
        payload = QueryRequest(
            question=message.content,
            conversation_id=conversation_id,
            doc_ids=doc_ids,
        )
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


async def _show_document_panel() -> None:
    """Muestra el panel de documentos indexados con acciones de selección.

    Se presenta como un paso colapsable encima del chat para emular un panel
    lateral dentro de las limitaciones de Chainlit.
    """
    client: httpx.AsyncClient | None = cl.user_session.get("client")
    if client is None:
        return

    try:
        response = await client.get("/api/documents", timeout=15.0)
        response.raise_for_status()
        documents = response.json()
    except httpx.HTTPError as exc:
        logger.warning("error listando documentos para el panel: %s", exc)
        return

    selected: list[str] = cl.user_session.get("selected_doc_ids", [])

    lines: list[str] = ["### 📄 Documentos indexados", ""]
    if not documents:
        lines.append("No hay documentos indexados. Sube un PDF arrastrándolo al chat.")
    else:
        for doc in documents:
            is_selected = doc["doc_id"] in selected
            marker = "✅" if is_selected else "⬜"
            lines.append(
                f"{marker} **{doc['source']}** — `{doc['doc_id']}` "
                f"({doc['chunks']} fragmentos, {doc['pages']} páginas)"
            )
        lines.append("")
        lines.append(
            "Haz clic en un documento para seleccionarlo/deseleccionarlo. "
            "Solo los seleccionados se consultan en el RAG. "
            "Si ninguno está seleccionado, se buscan en todos."
        )

    step = cl.Step(name="Documentos indexados (RAG)", type="tool")
    async with step:
        step.output = "\n".join(lines)

    # Acciones de selección (una por documento)
    if documents:
        actions: list[cl.Action] = []
        for doc in documents:
            is_selected = doc["doc_id"] in selected
            label = f"{'✅' if is_selected else '⬜'} {doc['doc_id']}"
            actions.append(
                cl.Action(
                    name="select_document",
                    value=doc["doc_id"],
                    label=label,
                    description=doc["source"],
                )
            )
        await cl.Message(
            content="**Seleccionar documentos:**",
            actions=actions,
            author="sistema",
        ).send()


async def _list_documents(client: httpx.AsyncClient) -> None:
    """Lista los documentos indexados en el vector store."""
    try:
        response = await client.get("/api/documents", timeout=15.0)
        response.raise_for_status()
        documents = response.json()
    except httpx.HTTPError as exc:
        logger.warning("error listando documentos: %s", exc)
        await cl.Message(
            content=f"No se pudieron listar los documentos ({exc.__class__.__name__}).",
            author="sistema",
        ).send()
        return

    selected: list[str] = cl.user_session.get("selected_doc_ids", [])

    if not documents:
        await cl.Message(
            content="No hay documentos indexados en el vector store.",
            author="sistema",
        ).send()
        return

    lines = ["### Documentos indexados", ""]
    for doc in documents:
        marker = "✅" if doc["doc_id"] in selected else "⬜"
        lines.append(
            f"{marker} **{doc['source']}** — doc_id: `{doc['doc_id']}` — "
            f"{doc['chunks']} fragmentos, {doc['pages']} páginas"
        )
    lines.append("")
    lines.append("Usa las acciones de selección para elegir qué documentos consulta el chatbot.")

    await cl.Message(content="\n".join(lines), author="sistema").send()


async def _make_starters(user: cl.User | None = None) -> list[cl.Starter]:
    return [cl.Starter(label=s, message=s) for s in _STARTERS]


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

        # Mostrar qué documentos se filtraron
        doc_ids = data.get("doc_ids") or []
        if doc_ids:
            extras.append("**Documentos filtrados:** " + ", ".join(f"`{d}`" for d in doc_ids))

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

        citations = data.get("citations") or []
        if citations:
            step = cl.Step(name="Citas (RAG)", type="run")
            async with step:
                cite_lines = []
                for c in citations:
                    loc = c.get("source", "documento")
                    if c.get("page") is not None:
                        loc += f", p. {c['page']}"
                    cite_lines.append(
                        f"• `{c.get('doc_id', '?')}` — {loc} (score {c.get('score', 0):.2f})"
                    )
                    cite_lines.append(f'  "{c.get("snippet", "")[:200]}"')
                step.output = "\n".join(cite_lines)

        meta = []
        if data.get("trace_id"):
            meta.append(f"trace `{data['trace_id']}`")
        if data.get("duration_ms") is not None:
            meta.append(f"{data['duration_ms']:.0f} ms")
        if data.get("used_memory"):
            meta.append("memoria conversacional usada")
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
