"""
Local vector/chunk store for the RAG index.

Deliberately plain JSON on disk rather than a vector database: the corpus here
is machine manuals and SOPs (hundreds to low thousands of chunks), where a
dependency-free store is faster to operate and easier to inspect. Swap in
Chroma/FAISS later behind the same three methods if the corpus grows.

Each chunk keeps the metadata needed for source-aware answers:
filename, document type, page number, section, chunk index, ingested_at.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from agent import config
from agent.logging_setup import get_logger

log = get_logger("RAG")

_INDEX_FILE = "index.json"


@dataclass
class Chunk:
    text: str
    filename: str = ""
    file_name: Optional[str] = None
    doc_type: str = "document"
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_index: int = 0
    ingested_at: float = 0.0
    embedding: Optional[List[float]] = None

    # Document metadata
    document_id: str = ""
    file_type: str = "markdown"
    version: str = "1.0"
    effective_date: str = "2026-10-08"
    department: str = "Production"

    # Chunk metadata
    chunk_id: str = ""
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    parent_section: Optional[str] = None
    token_count: int = 0

    # Knowledge category
    category: str = "Operations"
    equipment_type: str = "Concrete 3D Printer"
    severity_level: str = "Medium"
    keywords: List[str] = field(default_factory=list)

    # Retrieval & freshness metadata
    is_active: bool = True
    access_roles: List[str] = field(default_factory=lambda: ["Operator", "Supervisor", "Admin"])
    created_at: str = "2026-10-08 00:00:00"
    last_verified_at: str = "2026-10-08 00:00:00"
    verified_by: str = "Lead Engineer"

    def __post_init__(self) -> None:
        if not self.filename and self.file_name:
            self.filename = self.file_name
        if not self.file_name and self.filename:
            self.file_name = self.filename
        if self.page_number is None:
            self.page_number = self.page
        if self.page is None:
            self.page = self.page_number
        if self.section_heading is None:
            self.section_heading = self.section
        if self.section is None:
            self.section = self.section_heading
        if not self.chunk_id:
            clean_fn = self.filename.replace(" ", "_")
            self.chunk_id = f"{clean_fn}_{self.chunk_index}"
        if not self.document_id:
            self.document_id = self.filename
        if self.token_count == 0 and self.text:
            self.token_count = len(self.text.split())

    def citation(self) -> str:
        parts = [self.filename]
        p = self.page_number or self.page
        if p is not None:
            parts.append(f"page {p}")
        sec = self.section_heading or self.section
        if sec:
            parts.append(sec)
        return ", ".join(parts)

    def metadata(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "document_id": self.document_id,
            "doc_type": self.doc_type,
            "file_type": self.file_type,
            "version": self.version,
            "effective_date": self.effective_date,
            "department": self.department,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "page": self.page,
            "page_number": self.page_number,
            "section": self.section,
            "section_heading": self.section_heading,
            "parent_section": self.parent_section,
            "token_count": self.token_count,
            "category": self.category,
            "equipment_type": self.equipment_type,
            "severity_level": self.severity_level,
            "keywords": self.keywords,
            "is_active": self.is_active,
            "access_roles": self.access_roles,
            "ingested_at": self.ingested_at,
            "created_at": self.created_at,
            "last_verified_at": self.last_verified_at,
            "verified_by": self.verified_by,
        }


@dataclass
class DocumentStore:
    chunks: List[Chunk] = field(default_factory=list)
    embedder_name: str = ""
    is_semantic: bool = False

    # --- persistence ---

    @staticmethod
    def _path(index_dir: Optional[str] = None) -> str:
        return os.path.join(index_dir or config.RAG_INDEX_DIR, _INDEX_FILE)

    def save(self, index_dir: Optional[str] = None) -> str:
        directory = index_dir or config.RAG_INDEX_DIR
        os.makedirs(directory, exist_ok=True)
        path = self._path(directory)
        payload = {
            "embedder": self.embedder_name,
            "is_semantic": self.is_semantic,
            "chunks": [asdict(c) for c in self.chunks],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        log.info("Saved %d chunks to %s", len(self.chunks), path)
        return path

    @classmethod
    def load(cls, index_dir: Optional[str] = None) -> "DocumentStore":
        path = cls._path(index_dir)
        if not os.path.exists(path):
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            log.warning("Could not read RAG index %s: %s", path, exc)
            return cls()

        valid_fields = set(Chunk.__dataclass_fields__.keys())
        loaded_chunks = []
        for raw in payload.get("chunks", []):
            try:
                filtered = {k: v for k, v in raw.items() if k in valid_fields}
                loaded_chunks.append(Chunk(**filtered))
            except Exception as e:
                log.warning("Skipping corrupted chunk: %s", e)

        return cls(
            chunks=loaded_chunks,
            embedder_name=payload.get("embedder", ""),
            is_semantic=bool(payload.get("is_semantic")),
        )

    _bm25_corpus_tokens: Optional[List[List[str]]] = field(default=None, init=False, repr=False)
    _bm25_doc_lens: Optional[List[int]] = field(default=None, init=False, repr=False)
    _bm25_avgdl: float = field(default=0.0, init=False, repr=False)

    def invalidate_bm25(self) -> None:
        self._bm25_corpus_tokens = None
        self._bm25_doc_lens = None
        self._bm25_avgdl = 0.0

    # --- content ---

    def add(self, chunks: List[Chunk]) -> None:
        self.chunks.extend(chunks)
        self.invalidate_bm25()

    def remove_file(self, filename: str) -> int:
        """Drops all chunks for one file, so re-ingesting doesn't duplicate it."""
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.filename != filename]
        self.invalidate_bm25()
        return before - len(self.chunks)

    @property
    def texts(self) -> List[str]:
        return [c.text for c in self.chunks]

    def files(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for c in self.chunks:
            counts[c.filename] = counts.get(c.filename, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self.chunks)
