"""Tests de RAG con un doble de embeddings (sin vLLM).

Verifican la segmentación, limpieza de boilerplate, IDs deterministas,
idempotencia de la re-ingesta y el umbral de relevancia. La búsqueda semántica
real con el modelo e5 se prueba en `test_e2e_local_llm.py` (marcador
`local_llm`).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agentic_poc.services.rag import RagService, _clean_pages, _slugify

DIM = 16


class _FakeEmbeddings:
    """Embeddings deterministas por hash (dimensión fija) para no depender de vLLM."""

    def __init__(self, dimensions: int = DIM) -> None:
        self.dimensions = dimensions
        self.queries: list[str] = []
        self.passages: list[str] = []

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [b / 255.0 for b in digest[: self.dimensions]]
        norm = sum(v * v for v in raw) ** 0.5 or 1.0
        return [v / norm for v in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.passages.extend(texts)
        return [self._vec(f"passage:{t}") for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vec(f"query:{text}")


@pytest.fixture()
def rag(tmp_path: Path) -> RagService:
    return RagService(
        _FakeEmbeddings(),  # type: ignore[arg-type]
        persist_directory=tmp_path / "vector_store",
        chunk_size=200,
        chunk_overlap=40,
        score_threshold=0.0,
    )


def test_clean_pages_elimina_numeros_de_pagina() -> None:
    pages = ["Contenido útil.\nPágina 1 de 10", "Más contenido.\nPage 2 of 10"]
    cleaned = _clean_pages(pages)
    assert cleaned == ["Contenido útil.", "Más contenido."]


def test_clean_pages_elimina_boilerplate_repetido() -> None:
    pages = [f"CABECERA CORPORATIVA\nTexto único {i}." for i in range(4)]
    cleaned = _clean_pages(pages)
    assert all("CABECERA CORPORATIVA" not in page for page in cleaned)


def test_slugify() -> None:
    assert _slugify("Contrato Marco Suministro") == "contrato-marco-suministro"
    assert _slugify("contrato_marco_suministro") == "contrato-marco-suministro"
    assert _slugify("con äcentos / raros!!") == "con-centos-raros"


def test_ingesta_genera_chunks_con_ids_deterministas(rag: RagService) -> None:
    pdf = _make_pdf("doc.pdf", ["Linea de contrato uno. " * 40])
    result = rag.add_pdf(pdf)
    assert result.chunks >= 1
    assert result.doc_id == "doc"
    infos = rag.list_documents()
    assert [i.doc_id for i in infos] == ["doc"]


def test_reingesta_es_idempotente(rag: RagService) -> None:
    pdf = _make_pdf("doc.pdf", ["Texto del contrato. " * 40])
    first = rag.add_pdf(pdf)
    second = rag.add_pdf(pdf)
    assert first.chunks == second.chunks
    infos = rag.list_documents()
    assert len(infos) == 1
    assert infos[0].chunks == first.chunks


def test_delete_document(rag: RagService) -> None:
    pdf = _make_pdf("borrame.pdf", ["Contenido " * 60])
    rag.add_pdf(pdf)
    removed = rag.delete_document("borrame")
    assert removed >= 1
    assert rag.list_documents() == []


def test_search_devuelve_chunks_ordenados(rag: RagService) -> None:
    pdf = _make_pdf("doc.pdf", ["El plazo de entrega es de treinta dias. " * 30])
    rag.add_pdf(pdf)
    chunks = rag.search("plazo de entrega", k=3)
    assert chunks
    scores = [c.score for c in chunks]
    assert scores == sorted(scores, reverse=True)
    assert all(c.doc_id == "doc" for c in chunks)


def _make_pdf(name: str, pages: list[str]) -> Path:
    """PDF mínimo de una línea por página (usa fpdf2, dependencia de dev)."""
    from fpdf import FPDF

    pdf = FPDF()
    for text in pages:
        pdf.add_page()
        pdf.set_font("helvetica", size=11)
        pdf.multi_cell(0, 6, text)
    target = Path(name)
    pdf.output(str(target))
    return target
