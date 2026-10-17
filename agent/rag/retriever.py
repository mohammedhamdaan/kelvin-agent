import math
import re
from typing import Any, Dict, List, Optional

from agent import config
from agent.logging_setup import get_logger
from agent.rag.embeddings import GeminiEmbedder, LexicalEmbedder, get_embedder, _STOPWORDS
from agent.rag.store import Chunk, DocumentStore

log = get_logger("RAG")

_store_cache: Optional[DocumentStore] = None
_cached_dir: Optional[str] = None


def get_store(index_dir: Optional[str] = None, refresh: bool = False) -> DocumentStore:
    global _store_cache, _cached_dir
    directory = index_dir or config.RAG_INDEX_DIR
    if refresh or _store_cache is None or _cached_dir != directory:
        _store_cache = DocumentStore.load(directory)
        _cached_dir = directory
    return _store_cache


def bm25_score(
    query: str,
    chunks: List[Chunk],
    k1: float = 1.5,
    b: float = 0.75,
    store: Optional[DocumentStore] = None,
) -> List[float]:
    """Computes normalized BM25 keyword scores across document chunks with in-memory caching."""
    raw_tokens = [t.lower() for t in re.findall(r"\b\w+\b", query) if len(t) >= 2]
    if not raw_tokens or not chunks:
        return [0.0] * len(chunks)

    informative = [t for t in raw_tokens if t not in _STOPWORDS]
    query_tokens = informative or raw_tokens

    if (
        store is not None
        and getattr(store, "_bm25_corpus_tokens", None) is not None
        and len(store._bm25_corpus_tokens) == len(chunks)
    ):
        corpus_tokens = store._bm25_corpus_tokens
        doc_lens = store._bm25_doc_lens
        avgdl = store._bm25_avgdl
    else:
        corpus_tokens = []
        doc_lens = []
        for c in chunks:
            combined = f"{c.text} {' '.join(c.keywords)} {c.section_heading or ''} {c.category}"
            toks = [t.lower() for t in re.findall(r"\b\w+\b", combined) if len(t) >= 2]
            corpus_tokens.append(toks)
            doc_lens.append(len(toks))

        N = len(chunks)
        avgdl = sum(doc_lens) / max(1, N)
        if store is not None:
            store._bm25_corpus_tokens = corpus_tokens
            store._bm25_doc_lens = doc_lens
            store._bm25_avgdl = avgdl

    N = len(chunks)
    df: Dict[str, int] = {}
    for term in set(query_tokens):
        df[term] = sum(1 for doc in corpus_tokens if term in doc)

    def idf(term: str) -> float:
        n_t = df.get(term, 0)
        return math.log(1.0 + (N - n_t + 0.5) / (n_t + 0.5))

    raw_scores = []
    for i, doc in enumerate(corpus_tokens):
        d_len = doc_lens[i]
        score = 0.0
        tf: Dict[str, int] = {}
        for t in doc:
            tf[t] = tf.get(t, 0) + 1

        for term in query_tokens:
            if term not in tf:
                continue
            f = tf[term]
            term_score = idf(term) * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * (d_len / avgdl)))
            score += term_score
        raw_scores.append(score)

    ceiling = sum(idf(term) * (k1 + 1.0) for term in set(query_tokens))
    if ceiling <= 0.0:
        return [0.0] * len(chunks)
    return [round(s / ceiling, 4) for s in raw_scores]



def search_documentation(
    query: str,
    top_k: Optional[int] = None,
    index_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Searches ingested documentation using hybrid retrieval:
    combining dense semantic embeddings with sparse BM25 keyword matching and reranking.
    """
    store = get_store(index_dir)
    limit = top_k or config.RAG_TOP_K

    if len(store) == 0:
        return {
            "found": False,
            "results": [],
            "message": (
                "No documentation has been ingested yet. Add documents with: "
                "python ingest_docs.py <file-or-folder>"
            ),
        }

    # 1. Sparse BM25 keyword scoring (<1ms using cached in-memory tokens)
    bm25_scores = bm25_score(query, store.chunks, store=store)

    # 2. Dense semantic scoring (with 500ms safety timeout)
    dense_scores: List[float] = []
    has_dense = False
    if store.is_semantic and any(c.embedding for c in store.chunks):
        embedder = get_embedder()
        query_vector = None
        if isinstance(embedder, GeminiEmbedder):
            import concurrent.futures
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(embedder.embed_query, query)
                    query_vector = future.result(timeout=0.5)
            except (concurrent.futures.TimeoutError, Exception) as exc:
                log.info("Embedding query timed out (>500ms) or failed; falling back to in-memory BM25: %s", exc)
                query_vector = None

        if query_vector is not None:
            dense_scores = [
                GeminiEmbedder.cosine(query_vector, c.embedding) if c.embedding else 0.0
                for c in store.chunks
            ]
            has_dense = True

    # 3. Hybrid fusion and reranking
    if has_dense and dense_scores:
        final_scores = [
            round(0.6 * d + 0.4 * b, 4)
            for d, b in zip(dense_scores, bm25_scores)
        ]
        search_mode = "hybrid"
    else:
        # Fallback to lexical if BM25 didn't score (or combine with LexicalEmbedder)
        lex_scores = LexicalEmbedder().score(query, store.texts)
        final_scores = [
            max(b, l) for b, l in zip(bm25_scores, lex_scores)
        ]
        search_mode = "bm25_lexical"

    ranked = sorted(zip(store.chunks, final_scores), key=lambda pair: pair[1], reverse=True)
    hits = [(c, s) for c, s in ranked[:limit] if s >= config.RAG_MIN_SCORE]

    if not hits:
        return {
            "found": False,
            "results": [],
            "message": (
                "That information was not found in the available documentation. "
                f"({len(store)} chunks searched across {len(store.files())} document(s).)"
            ),
        }

    return {
        "found": True,
        "results": [
            {
                "text": chunk.text,
                "source": chunk.citation(),
                "score": round(float(score), 3),
                **chunk.metadata(),
            }
            for chunk, score in hits
        ],
        "message": f"{len(hits)} relevant passage(s) found.",
        "search_mode": search_mode,
    }


def index_summary(index_dir: Optional[str] = None) -> Dict[str, Any]:
    store = get_store(index_dir, refresh=True)
    return {
        "chunks": len(store),
        "documents": store.files(),
        "embedder": store.embedder_name or "none",
        "semantic": store.is_semantic,
    }


def search_rag_brain(
    query: str,
    top_k: int = 4,
    index_dir: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Searches both RAG documentation index and historical shift_issues in the database.
    """
    doc_results = search_documentation(query, top_k=top_k, index_dir=index_dir)

    db_issues = []
    try:
        from agent import db as agent_db
        with agent_db.connect(db_path) as conn:
            q_terms = [t.lower() for t in re.findall(r"\b\w+\b", query) if len(t) >= 3]
            for term in q_terms[:3]:
                pattern = f"%{term}%"
                rows = conn.execute(
                    """
                    SELECT issue_id, problem_type, problem_description, solution, severity, status, resolved_at
                    FROM shift_issues
                    WHERE problem_type LIKE ? OR problem_description LIKE ? OR solution LIKE ?
                    ORDER BY issue_id DESC LIMIT 3
                    """,
                    (pattern, pattern, pattern),
                ).fetchall()
                for r in rows:
                    item = dict(r)
                    if not any(existing["issue_id"] == item["issue_id"] for existing in db_issues):
                        db_issues.append(item)
    except Exception as exc:
        log.warning("Could not query shift_issues for search_rag_brain: %s", exc)

    return {
        "query": query,
        "documentation": doc_results,
        "historical_shift_issues": db_issues,
        "found": doc_results.get("found", False) or bool(db_issues),
    }
