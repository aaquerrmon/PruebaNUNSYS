"""Memoria conversacional ligera en SQLite.

Almacena los últimos `memory_retention_messages` por conversación. La lógica de
memoria es intencionalmente simple (lista FIFO) para cumplir el requisito
"últimas 5 interacciones" y permitir preguntas encadenadas.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import Column, MetaData, String, Table, select
from sqlalchemy.engine import Engine

from agentic_poc.api.schemas import ConversationMessage

metadata = MetaData()

conversation_turns = Table(
    "conversation_turns",
    metadata,
    Column("conversation_id", String, index=True, nullable=False),
    Column("role", String, nullable=False),
    Column("content", String, nullable=False),
    Column("ts", String, nullable=False),
)


class ConversationStore:
    """Persistencia mínima para el historial conversacional."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        metadata.create_all(self._engine)

    async def get_history(self, conversation_id: str, limit: int = 10) -> list[BaseMessage]:
        """Últimos ``limit`` mensajes, en orden cronológico (los más recientes)."""
        with self._engine.connect() as conn:
            stmt = (
                select(conversation_turns.c.role, conversation_turns.c.content)
                .where(conversation_turns.c.conversation_id == conversation_id)
                .order_by(conversation_turns.c.ts.desc())
                .limit(max(1, limit))
            )
            rows = list(reversed(conn.execute(stmt).fetchall()))
        history: list[BaseMessage] = []
        for role, content in rows:
            if role == "human":
                history.append(HumanMessage(content=content))
            else:
                history.append(AIMessage(content=content))
        return history

    async def save_turn(self, conversation_id: str, human: str, assistant: str) -> None:
        now = str(time.time())
        with self._engine.begin() as conn:
            conn.execute(
                conversation_turns.insert(),
                [
                    {
                        "conversation_id": conversation_id,
                        "role": "human",
                        "content": human,
                        "ts": now,
                    },
                    {
                        "conversation_id": conversation_id,
                        "role": "assistant",
                        "content": assistant,
                        "ts": now + "1",
                    },
                ],
            )


def serialize_history(messages: Sequence[BaseMessage]) -> list[ConversationMessage]:
    result: list[ConversationMessage] = []
    for message in messages:
        role = "human" if isinstance(message, HumanMessage) else "assistant"
        content = message.content if isinstance(message.content, str) else str(message.content)
        result.append(ConversationMessage(role=role, content=content, ts=0.0))
    return result
