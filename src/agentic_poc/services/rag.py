"""Servicio RAG: ingesta de PDF, indexacion vectorial (Chroma) y busqueda.

- La indexacion es idempotente por documento: re-subir un PDF reemplaza sus
  fragmentos (mismos IDs deterministas).
- Cada fragmento conserva `doc_id`, `source`, `page` y `chunk_index`, de modo
  que las citas de la respuesta siempre son verificables.
- El espacio de la coleccion es coseno; la relevancia se normaliza a
  similitud (1 - distancia) y se filtra por umbral (nunca "sin resultados").
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from agentic_poc.services.embeddings import LocalE5Embeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"


class RagError(RuntimeError):
    """Error de ingesta o de acceso al indice vectorial."""


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    doc_id: str
    source: str
    page: int | None
    chunk_index: int | None
    text: str
    score: float

    def citation(self) -> str:
        location = f"{self.source}"
        if self.page is not None:
            location += f", pag. {self.page}"
        return location


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    doc_id: str
    source: str
    chunks: int
    pages: int


@dataclass(frozen=True, slots=True)
class IngestionResult:
    doc_id: str
    source: str
    pages: int
    chunks: int
    skipped_pages: int


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "documento"


_PAGE_NUMBER_RE = re.compile(
    r"^\s*(pagina|página|page)\s+\d+(\s*(de|of)\s*\d+)?\s*$", re.IGNORECASE
)


def _clean_pages(pages: list[str]) -> list[str]:
    """Elimina encabezados/pies repetidos y numeros de pagina.

    Se considera boilerplate cualquier linea (no vacia) que aparezca en la
    mayoria de las paginas del documento, ademas de las lineas de numeracion.
    """
    if len(pages) < 2:
        return [_strip_page_numbers(page) for page in pages]
    counts: dict[str, int] = {}
    for page in pages:
        for raw_line in page.splitlines():
            line = raw_line.strip()
            if 0 < len(line) < 120:
                counts[line] = counts.get(line, 0) + 1
    threshold = max(2, len(pages) // 2 + 1)
    boilerplate = {line for line, count in counts.items() if count >= threshold}
    cleaned: list[str] = []
    for page in pages:
        kept = [raw_line for raw_line in page.splitlines() if raw_line.strip() not in boilerplate]
        cleaned.append(_strip_page_numbers("\n".join(kept)))
    return cleaned


def _strip_page_numbers(page: str) -> str:
    return "\n".join(line for line in page.splitlines() if not _PAGE_NUMBER_RE.match(line))


class RagService:
    """Fachada de RAG sobre Chroma + embeddings e5 locales."""

    def __init__(
        self,
        embeddings: LocalE5Embeddings,
        *,
        persist_directory: Path,
        top_k: int = 5,
        score_threshold: float = 0.35,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        max_pdf_bytes: int = 10 * 1024 * 1024,
        max_pdf_pages: int = 200,
    ) -> None:
        self._persist_directory = Path(persist_directory)
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.max_pdf_bytes = max_pdf_bytes
        self.max_pdf_pages = max_pdf_pages
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(self._persist_directory),
            collection_metadata={"hnsw:space": "cosine"},
        )

    # -- ingesta ------------------------------------------------------------

    def add_pdf(self, pdf_path: Path, *, doc_id: str | None = None) -> IngestionResult:
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            raise RagError(f"El documento no existe: {pdf_path}")
        size = pdf_path.stat().st_size
        if size > self.max_pdf_bytes:
            raise RagError(f"El PDF supera el tamano maximo ({size} > {self.max_pdf_bytes} bytes)")
        try:
            reader = PdfReader(str(pdf_path))
            total_pages = len(reader.pages)
            if total_pages == 0:
                raise RagError("El PDF no contiene paginas")
            if total_pages > self.max_pdf_pages:
                raise RagError(
                    f"El PDF supera el maximo de paginas ({total_pages} > {self.max_pdf_pages})"
                )
            pages_text: list[str] = []
            for page in reader.pages:
                pages_text.append(page.extract_text() or "")
            pages_text = _clean_pages(pages_text)
        except RagError:
            raise
        except Exception as exc:
            raise RagError(f"No se pudo leer el PDF {pdf_path.name}: {exc}") from exc

        resolved_doc_id = doc_id or _slugify(pdf_path.stem)
        self.delete_document(resolved_doc_id)

        documents: list[Document] = []
        ids: list[str] = []
        skipped = 0
        for page_number, text in enumerate(pages_text, start=1):
            if len(text.strip()) < 20:
                skipped += 1
                continue
            for chunk_index, chunk in enumerate(self._splitter.split_text(text)):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "doc_id": resolved_doc_id,
                            "source": pdf_path.name,
                            "page": page_number,
                            "chunk_index": chunk_index,
                        },
                    )
                )
                ids.append(f"{resolved_doc_id}:p{page_number}:c{chunk_index}")
        if not documents:
            raise RagError(f"El PDF {pdf_path.name} no contiene texto extraible")

        try:
            self._store.add_documents(documents=documents, ids=ids)
        except Exception as exc:
            raise RagError(f"No se pudo indexar {pdf_path.name}: {exc}") from exc
        logger.info(
            "Documento indexado: %s (%s paginas, %s fragmentos)",
            resolved_doc_id,
            total_pages,
            len(documents),
        )
        return IngestionResult(
            doc_id=resolved_doc_id,
            source=pdf_path.name,
            pages=total_pages,
            chunks=len(documents),
            skipped_pages=skipped,
        )

    def delete_document(self, doc_id: str) -> int:
        try:
            existing = self._store.get(where={"doc_id": doc_id})
        except Exception as exc:  # pragma: no cover - defensivo
            raise RagError(f"No se pudo consultar el indice: {exc}") from exc
        ids = list(existing.get("ids") or [])
        if ids:
            self._store.delete(ids=ids)
        return len(ids)

    def list_documents(self) -> list[DocumentInfo]:
        try:
            payload = self._store.get(include=["metadatas"])
        except Exception as exc:  # pragma: no cover - defensivo
            raise RagError(f"No se pudo listar el indice: {exc}") from exc
        metadatas = payload.get("metadatas") or []
        sources: dict[str, str] = {}
        chunk_counts: dict[str, int] = {}
        page_sets: dict[str, set[int]] = {}
        for metadata in metadatas:
            if not metadata:
                continue
            doc_id = str(metadata.get("doc_id", "desconocido"))
            sources.setdefault(doc_id, str(metadata.get("source", doc_id)))
            chunk_counts[doc_id] = chunk_counts.get(doc_id, 0) + 1
            page = metadata.get("page")
            if isinstance(page, int):
                page_sets.setdefault(doc_id, set()).add(page)
        return [
            DocumentInfo(
                doc_id=doc_id,
                source=sources[doc_id],
                chunks=count,
                pages=len(page_sets.get(doc_id, ())),
            )
            for doc_id, count in sorted(chunk_counts.items())
        ]

    # -- busqueda -----------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        limit = k or self.top_k
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        try:
            results = self._store.similarity_search_with_score(query, k=limit, filter=where)
        except Exception as exc:
            raise RagError(f"Fallo la busqueda vectorial: {exc}") from exc
        chunks: list[RetrievedChunk] = []
        for document, distance in results:
            score = 1.0 - float(distance)  # espacio coseno: distancia = 1 - similitud
            metadata = document.metadata or {}
            page = metadata.get("page")
            chunk_index = metadata.get("chunk_index")
            chunks.append(
                RetrievedChunk(
                    doc_id=str(metadata.get("doc_id", "")),
                    source=str(metadata.get("source", "documento")),
                    page=page if isinstance(page, int) else None,
                    chunk_index=chunk_index if isinstance(chunk_index, int) else None,
                    text=document.page_content,
                    score=round(score, 4),
                )
            )
        return chunks

    def search_relevant(
        self,
        query: str,
        *,
        k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Fragmentos por encima del umbral de similitud (puede ser vacio)."""
        return [
            c for c in self.search(query, k=k, doc_ids=doc_ids) if c.score >= self.score_threshold
        ]

    @property
    def is_empty(self) -> bool:
        try:
            return self._store._collection.count() == 0
        except Exception:  # pragma: no cover - defensivo
            return True
