"""
Document ingestion: load -> chunk -> (embed) -> store.

Supported: .txt, .md, .pdf (needs `pypdf`). Page numbers are preserved for PDFs
and Markdown headings become the `section` metadata, so answers can cite where
they came from.
"""

import os
import re
import threading
import time
from typing import List, Optional, Tuple

from agent import config
from agent.logging_setup import get_logger
from agent.rag.embeddings import get_embedder
from agent.rag.store import Chunk, DocumentStore

log = get_logger("RAG")

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".log"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _doc_type(filename: str) -> str:
    """Coarse classification from the name, used as retrieval metadata."""
    lower = filename.lower()
    for keyword, label in (
        ("manual", "manual"),
        ("sop", "sop"),
        ("troubleshoot", "troubleshooting"),
        ("maintenance", "maintenance"),
        ("report", "report"),
        ("spec", "specification"),
    ):
        if keyword in lower:
            return label
    return "document"


def load_text_file(path: str) -> List[Tuple[str, Optional[int]]]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return [(fh.read(), None)]


def load_pdf(path: str) -> List[Tuple[str, Optional[int]]]:
    """Returns [(page_text, page_number)]. Requires pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF ingestion needs pypdf. Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(path)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            log.warning("Could not extract page %d of %s: %s", number, path, exc)
            text = ""
        if text.strip():
            pages.append((text, number))
    return pages


def _section_for(text: str, offset: int) -> Optional[str]:
    """Nearest preceding Markdown heading, used as the section label."""
    last = None
    for match in _HEADING_RE.finditer(text):
        if match.start() > offset:
            break
        last = match.group(2)
    return last


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """
    Splits into overlapping windows, preferring paragraph/sentence boundaries so
    chunks stay readable. Returns [(chunk_text, start_offset)].

    Markdown headings split FIRST: a manual section such as "## Delamination"
    should be its own chunk, otherwise a query for one topic retrieves the
    neighbouring section that happens to share a window.
    """
    size = chunk_size or config.RAG_CHUNK_SIZE
    step_back = overlap if overlap is not None else config.RAG_CHUNK_OVERLAP
    text = text.strip()
    if not text:
        return []

    # Section-aware split on markdown headings.
    headings = list(_HEADING_RE.finditer(text))
    if len(headings) > 1:
        sections: List[Tuple[str, int]] = []
        for index, match in enumerate(headings):
            start = match.start()
            end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            section = text[start:end].strip()
            if not section:
                continue
            if len(section) <= size:
                sections.append((section, start))
            else:
                # Long section: window it, keeping offsets absolute.
                for piece, offset in _window(section, size, step_back):
                    sections.append((piece, start + offset))
        if sections:
            return sections

    if len(text) <= size:
        return [(text, 0)]
    return _window(text, size, step_back)


def _window(text: str, size: int, step_back: int) -> List[Tuple[str, int]]:
    """Overlapping windows that prefer paragraph/sentence boundaries."""

    chunks: List[Tuple[str, int]] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer a paragraph break, then a sentence end, within the tail.
            window = text[start:end]
            for boundary in ("\n\n", ". ", "\n"):
                cut = window.rfind(boundary)
                if cut > size * 0.5:
                    end = start + cut + len(boundary)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append((piece, start))
        if end >= len(text):
            break
        start = max(end - step_back, start + 1)
    return chunks


def ingest_file(path: str, store: DocumentStore) -> int:
    """Ingests one file into the store (replacing prior chunks for it)."""
    filename = os.path.basename(path)
    extension = os.path.splitext(path)[1].lower()

    if extension in PDF_EXTENSIONS:
        pages = load_pdf(path)
    elif extension in TEXT_EXTENSIONS:
        pages = load_text_file(path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    removed = store.remove_file(filename)
    if removed:
        log.info("Replacing %d existing chunks for %s", removed, filename)

    doc_type = _doc_type(filename)
    now = time.time()
    new_chunks: List[Chunk] = []

    for page_text, page_number in pages:
        for index, (piece, offset) in enumerate(chunk_text(page_text)):
            sec = _section_for(page_text, offset)
            low_text = piece.lower()

            # Infer Category
            if "safety" in low_text or "hazard" in low_text or "warning" in low_text:
                cat = "Safety"
            elif "maintenance" in low_text or "clean" in low_text or "flush" in low_text:
                cat = "Maintenance"
            elif "troubleshoot" in low_text or "issue" in low_text or "clog" in low_text or "loss" in low_text:
                cat = "Troubleshooting"
            elif "quality" in low_text or "delamination" in low_text or "crack" in low_text:
                cat = "Quality"
            else:
                cat = "Operations"

            # Infer Severity
            if "critical" in low_text or "danger" in low_text or "stop print" in low_text:
                sev = "Critical"
            elif "warning" in low_text or "high" in low_text or "conflict" in low_text:
                sev = "High"
            elif "medium" in low_text or "moderate" in low_text:
                sev = "Medium"
            else:
                sev = "Low"

            # Extract Keywords
            kw_candidates = [
                "slump", "pump", "nozzle", "delamination", "concrete", "retarder",
                "accelerator", "pressure", "extruder", "mortar", "hopper", "cleaning",
                "layer", "admixture", "flow", "clogging", "setting", "water", "speed"
            ]
            kws = [kw for kw in kw_candidates if kw in low_text]

            new_chunks.append(
                Chunk(
                    text=piece,
                    filename=filename,
                    doc_type=doc_type,
                    file_type="markdown" if extension in (".md", ".markdown") else "text",
                    page=page_number,
                    page_number=page_number,
                    section=sec,
                    section_heading=sec,
                    chunk_index=index,
                    chunk_id=f"{filename}_{index}",
                    token_count=len(piece.split()),
                    category=cat,
                    severity_level=sev,
                    keywords=kws,
                    ingested_at=now,
                )
            )

    store.add(new_chunks)
    log.info("Ingested %s -> %d chunks (%s)", filename, len(new_chunks), doc_type)
    return len(new_chunks)


def embed_store(store: DocumentStore) -> None:
    """Attaches embeddings when a semantic embedder is available."""
    embedder = get_embedder()
    store.embedder_name = embedder.name
    store.is_semantic = embedder.is_semantic

    if not embedder.is_semantic:
        log.info("No embedding API configured -- index will use offline lexical search.")
        return

    vectors = embedder.embed_documents(store.texts)
    if vectors is None:
        log.warning("Embedding failed -- index falls back to lexical search.")
        store.is_semantic = False
        return
    for chunk, vector in zip(store.chunks, vectors):
        chunk.embedding = vector
    log.info("Embedded %d chunks with %s", len(store.chunks), embedder.name)


def ingest_path(path: str, index_dir: Optional[str] = None) -> dict:
    """
    Ingests a file or a whole directory (recursively) and saves the index.
    Returns a summary dict.
    """
    store = DocumentStore.load(index_dir)
    files: List[str] = []

    if os.path.isdir(path):
        for root, _dirs, names in os.walk(path):
            for name in names:
                if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
                    files.append(os.path.join(root, name))
    elif os.path.isfile(path):
        files = [path]
    else:
        return {"ok": False, "error": f"Path not found: {path}", "files": 0, "chunks": 0}

    if not files:
        return {
            "ok": False,
            "error": f"No supported documents in {path} "
                     f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})",
            "files": 0,
            "chunks": 0,
        }

    total = 0
    ingested = []
    errors = []
    for file_path in files:
        try:
            total += ingest_file(file_path, store)
            ingested.append(os.path.basename(file_path))
        except Exception as exc:
            errors.append(f"{os.path.basename(file_path)}: {exc}")
            log.warning("Failed to ingest %s: %s", file_path, exc)

    embed_store(store)
    index_path = store.save(index_dir)

    return {
        "ok": bool(ingested),
        "files": len(ingested),
        "chunks": total,
        "total_chunks_in_index": len(store),
        "ingested": ingested,
        "errors": errors,
        "index_path": index_path,
        "semantic": store.is_semantic,
    }


def learn_and_store_in_rag_brain(
    title: str,
    category: str = "Operations",
    content: str = "",
    problem: Optional[str] = None,
    solution: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    index_dir: Optional[str] = None,
) -> dict:
    """
    Dynamically stores a newly learned problem, fix, SOP or parameter into:
    1. docs/knowledge/learned_knowledge.md (disk persistence)
    2. DocumentStore active in-memory and saved to rag_index (immediate retrieval)
    """
    clean_title = (title or "Learned Operational Knowledge").strip()
    prob_text = (problem or content or "").strip()
    sol_text = (solution or content or "").strip()
    full_text = (
        f"## Issue: {clean_title}\n"
        f"**Category:** {category}\n"
        f"**Problem:** {prob_text}\n"
        f"**Solution / Corrective Action:** {sol_text}\n"
        f"**Verification Status:** Site-Verified\n"
    )

    # 1. Append to docs/knowledge/learned_knowledge.md
    docs_dir = os.path.join(config.RAG_DOCS_DIR, "knowledge")
    os.makedirs(docs_dir, exist_ok=True)
    learned_file = os.path.join(docs_dir, "learned_knowledge.md")

    with open(learned_file, "a", encoding="utf-8") as f:
        f.write("\n" + full_text + "\n")
    log.info("Appended learned knowledge to %s", learned_file)

    # 2. Add chunk to DocumentStore
    from agent.rag.retriever import get_store
    store = get_store(index_dir, refresh=False)

    now = time.time()
    kw_list = keywords or []
    if not kw_list:
        kw_list = [w.lower() for w in re.findall(r"\b\w+\b", f"{clean_title} {prob_text} {sol_text}") if len(w) >= 4]

    chunk_idx = len(store)
    new_chunk = Chunk(
        text=full_text.strip(),
        filename="learned_knowledge.md",
        doc_type="learned_knowledge",
        file_type="markdown",
        page=None,
        page_number=None,
        section=clean_title,
        section_heading=clean_title,
        chunk_index=chunk_idx,
        chunk_id=f"learned_{int(now)}_{chunk_idx}",
        token_count=len(full_text.split()),
        category=category,
        severity_level="Medium",
        keywords=kw_list,
        ingested_at=now,
    )

    store.add([new_chunk])

    # 3. Non-blocking vector embedding and disk save in background thread
    def _bg_embed_and_save():
        try:
            if store.is_semantic:
                embedder = get_embedder()
                vec = embedder.embed_query(new_chunk.text)
                if vec:
                    new_chunk.embedding = vec
            saved_path = store.save(index_dir)
            log.info("Background embedding & index save complete for chunk %s (%s)", new_chunk.chunk_id, saved_path)
        except Exception as exc:
            log.warning("Background embedding failed for %s: %s", new_chunk.chunk_id, exc)

    threading.Thread(target=_bg_embed_and_save, name="rag-learn-bg-embed", daemon=True).start()
    log.info("Appended chunk %s to in-memory RAG brain immediately (<5ms)", new_chunk.chunk_id)

    return {
        "success": True,
        "title": clean_title,
        "category": category,
        "chunk_id": new_chunk.chunk_id,
        "total_chunks": len(store),
        "message": f"Successfully stored and indexed knowledge: '{clean_title}'.",
    }
