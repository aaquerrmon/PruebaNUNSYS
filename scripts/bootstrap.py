"""Bootstrap del entorno: siembra el ERP, genera los PDFs de ejemplo e ingiere el RAG.

Uso:  python scripts/bootstrap.py [--skip-rag]

En Docker Compose se ejecuta como servicio `bootstrap` con dependencia sana de
`embeddings` (la ingesta requiere el servidor de embeddings local de vLLM).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy.engine import Engine

from agentic_poc.config import get_settings
from agentic_poc.infrastructure.database import make_writable_engine
from agentic_poc.infrastructure.erp_seed import seed_if_configured


def _erp_writable_engine(erp_database_url: str) -> Engine:
    """Motor de escritura para el ERP configurado, creando el directorio si hace falta.

    Usa `settings.erp_database_url` (no una ruta fija): en Compose la BD vive
    en el volumen montado en /data/erp, no en el directorio de trabajo.
    """
    prefix = "sqlite:///"
    if not erp_database_url.startswith(prefix):
        raise SystemExit(
            f"[bootstrap] ERP_DATABASE_URL no es SQLite: se esperaba un prefijo {prefix}"
        )
    db_path = Path(erp_database_url[len(prefix) :])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return make_writable_engine(f"sqlite:///{db_path.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-rag", action="store_true", help="No ingerir documentos en el vector store"
    )
    args = parser.parse_args()

    settings = get_settings()
    settings.assert_local_inference()

    engine = _erp_writable_engine(settings.erp_database_url)
    try:
        stats = seed_if_configured(engine)
    finally:
        engine.dispose()
    print(f"[bootstrap] ERP: {stats}")

    documents_dir = Path(settings.documents_dir)
    documents_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(documents_dir.glob("*.pdf"))
    if not pdfs:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from generate_sample_docs import (
            build_contract,
            build_production_procedure,
            build_service_levels,
        )

        builders = {
            "contrato_marco_suministro.pdf": build_contract,
            "procedimiento_produccion.pdf": build_production_procedure,
            "acuerdo_niveles_servicio.pdf": build_service_levels,
        }
        for name, builder in builders.items():
            builder(documents_dir / name)
        pdfs = sorted(documents_dir.glob("*.pdf"))
    print(f"[bootstrap] documentos PDF: {len(pdfs)}")

    if args.skip_rag:
        print("[bootstrap] ingestión RAG omitida")
        return 0

    from agentic_poc.services.embeddings import LocalE5Embeddings
    from agentic_poc.services.rag import RagService

    embeddings = LocalE5Embeddings(
        settings.embeddings_base_url,
        model=settings.embeddings_model,
        api_key=settings.embeddings_api_key,
        dimensions=settings.embeddings_dimensions,
        timeout_seconds=settings.embeddings_timeout_seconds,
        batch_size=settings.embeddings_batch_size,
        passage_prefix=settings.embeddings_passage_prefix,
        query_prefix=settings.embeddings_query_prefix,
    )
    rag = RagService(embeddings, persist_directory=settings.vector_store_path)
    for pdf in pdfs:
        try:
            result = rag.add_pdf(pdf)
            print(
                f"[bootstrap] ingestado {result.doc_id}: {result.chunks} chunks, {result.pages} páginas"
            )
        except Exception as e:
            print(f"[bootstrap] error procesando {pdf}: {e}")
    print("[bootstrap] listo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
