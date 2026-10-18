"""Construcción del agente Deep Agents con el perfil de la POC.

Se registra un `HarnessProfile` para el modelo local que:
- fija el prompt de sistema,
- excluye las herramientas de sistema de archivos y `execute` (no aportan a
  la POC y consumen contexto, crítico con un modelo de 4B),
- desactiva el subagente general por defecto (evita el coste del tool `task`).
"""

from __future__ import annotations

import logging
from typing import Any

from deepagents import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from agentic_poc.agents.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_EXCLUDED_TOOLS = frozenset(
    {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute", "task"}
)

_PROFILE = HarnessProfile(
    base_system_prompt=SYSTEM_PROMPT,
    excluded_tools=_EXCLUDED_TOOLS,
    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
)

_REGISTERED = False


def register_local_profile(model_name: str, provider: str = "openai") -> None:
    """Registra (una vez por proceso) el perfil del modelo local.

    Se registra tanto el par `provider:model` como el fallback por proveedor.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    register_harness_profile(f"{provider}:{model_name}", _PROFILE)
    register_harness_profile(provider, _PROFILE)
    _REGISTERED = True
    logger.debug("Perfil de harness registrado para %s:%s", provider, model_name)


def build_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    *,
    model_name: str,
    provider: str = "openai",
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Crea un agente Deep Agents ligado a las herramientas de la petición."""
    from deepagents import create_deep_agent

    register_local_profile(model_name, provider)
    return create_deep_agent(model=model, tools=tools)
