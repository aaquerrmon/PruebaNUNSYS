"""Descarga los modelos de Hugging Face a la caché compartida (solo una vez).

Se ejecuta como servicio `model-loader` en Docker Compose antes de arrancar
vLLM; los volúmenes son compartidos, así que la descarga solo ocurre la primera
vez. La inferencia posterior es 100 % local.
"""

from __future__ import annotations

import sys

from huggingface_hub import snapshot_download

MODELS = ["Qwen/Qwen3-4B-AWQ", "intfloat/multilingual-e5-small"]


def main() -> int:
    for repo in MODELS:
        path = snapshot_download(repo_id=repo)
        print(f"[model-loader] {repo} -> {path}")
    print("[model-loader] completado")
    return 0


if __name__ == "__main__":
    sys.exit(main())
