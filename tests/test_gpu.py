"""Comprobaciones de presupuesto de GPU y de que la inferencia es local.

Se ejecutan con `uv run pytest -m gpu` y se saltan si no hay `nvidia-smi`.
Requieren el stack `llm` + `embeddings` cargado en la GPU.
"""

from __future__ import annotations

import subprocess

import pytest

from agentic_poc.config import get_settings

pytestmark = pytest.mark.gpu


VRAM_LIMIT_MB = 8 * 1024


def _nvidia_smi_used_mb() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    first = output.strip().splitlines()[0] if output.strip() else ""
    return int(first) if first.isdigit() else None


def test_vram_dentro_del_presupuesto() -> None:
    used = _nvidia_smi_used_mb()
    if used is None:
        pytest.skip("nvidia-smi no disponible")
    assert used <= VRAM_LIMIT_MB, f"VRAM en uso {used} MB > {VRAM_LIMIT_MB} MB"


@pytest.mark.local_llm
async def test_llm_y_embeddings_responden_y_listan_modelos() -> None:
    import httpx

    settings = get_settings()
    async with httpx.AsyncClient(timeout=15.0) as client:
        llm = await client.get(f"{settings.llm_base_url}/models")
        emb = await client.get(f"{settings.embeddings_base_url}/models")
    assert llm.status_code == 200
    assert emb.status_code == 200
    llm_models = {m["id"] for m in llm.json()["data"]}
    emb_models = {m["id"] for m in emb.json()["data"]}
    assert settings.llm_model in llm_models
    assert settings.embeddings_model in emb_models


def test_configuracion_solo_local() -> None:
    get_settings().assert_local_inference()
