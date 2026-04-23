"""
RAG retrieval for institute Q&A — ChromaDB is the only runtime store for answers.

- Vectors + document text live in a persistent Chroma collection under `data/.rag_index/chroma/`.
- Queries: embed with OpenAI → `collection.query` (cosine) → filter weak hits → top chunks.
- List-style queries use higher effective top_k; non-list defaults favor precision (top_k≈5).
- Ingestion: run `scripts/build_rag_index.py`.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

import chromadb
from openai import OpenAI

from .config import (
    DATA_ROOT,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    openai_http_timeout,
)

INDEX_DIR = DATA_ROOT / ".rag_index"
CHROMA_DIR = INDEX_DIR / "chroma"
META_FILE = INDEX_DIR / "meta.json"
COLLECTION_NAME = "sgsits_knowledge"

# Maximum number of characters taken from one retrieved chunk not too big slow down the model
MAX_CHARS_PER_HIT = 2200
MAX_HITS_CAP = 12#Maximum number of chunks (hits) you will use
# Cosine distance d in [0, 2]; score = 1 - d. Drop very weak matches to reduce hallucination context.
MIN_SCORE = 0.12
# Over-fetch before threshold filter
QUERY_MULTIPLIER = 2
# Initial fetch: 24 chunks
# ↓
# After MIN_SCORE filter: 15 remain
# ↓
# Final selection: top 12
_LIST_QUERY = re.compile(
    r"\b(faculty|faculties|all\s+faculty|list\s+of|complete\s+list|every|all\s+departments?|"
    r"departments?\b|hod\b|head\s+of\s+department|chairperson|chair\s+person|"
    r"syllabus\s+topics?|contact\s+list|phone\s+list|emails?)\b",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _chroma_client() -> chromadb.PersistentClient:#make vector db
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


@lru_cache(maxsize=1)
def _openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY, timeout=openai_http_timeout(), max_retries=1)


def _collection() -> Any:#get access
    client = _chroma_client()
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return None


def _read_meta() -> dict[str, Any]:#Reads meta.json
    if not META_FILE.is_file():
        return {}
    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

#Finds latest modified .json file time
def _data_json_max_mtime(data_root: Any) -> float:
    m = 0.0
    for p in data_root.rglob("*.json"):
        try:
            parts = p.relative_to(data_root).parts
        except ValueError:
            continue
        if parts and parts[0] == "user-data":
            continue
        if ".rag_index" in parts:
            continue
        try:
            m = max(m, p.stat().st_mtime)
        except OSError:
            pass
    return m


def knowledge_base_outdated() -> bool:#do we need to rebuilt the vector db
    """
    True if any source JSON under data/ is newer than the mtime recorded at last index build
    (stored in meta.json as source_data_max_mtime). Legacy meta without that field is treated
    as unknown (no warning) until the index is rebuilt with a current build script.
    """
    meta = _read_meta()
    indexed = meta.get("source_data_max_mtime")
    if indexed is None:
        return False
    try:
        idx_val = float(indexed)
    except (TypeError, ValueError):
        return True
    current = _data_json_max_mtime(DATA_ROOT)
    # small slack avoids float noise on same-second writes
    return current > idx_val + 0.5


def index_exists() -> bool:#do we even have data???
    coll = _collection()
    if coll is None:
        return False
    try:
        return coll.count() > 0
    except Exception:
        return False


def _normalize_query(q: str) -> str:#clean user input
    q = re.sub(r"\s+", " ", q.strip().lower())
    return q[:2000]


@lru_cache(maxsize=256)#if same query come use the cached vector no need to call api
def _embedding_vector(model: str, query_key: str) -> tuple[float, ...]:#convert text->vector
    """Cache embeddings for identical queries in this process (saves API time + avoids duplicate timeouts)."""
    oai = _openai_client()
    r = oai.embeddings.create(model=model, input=query_key)#covert query into vector
    return tuple(r.data[0].embedding)#return vector


def _trim_text(s: str, limit: int) -> tuple[str, bool]:
    s = (s or "").strip()
    if len(s) <= limit:
        return s, False
    # Prefer breaking at newline so list rows stay intact
    cut = s[:limit]
    nl = cut.rfind("\n")
    if nl > limit * 0.55:
        cut = cut[:nl]
    else:
        sp = cut.rfind(" ")
        if sp > limit * 0.65:
            cut = cut[:sp]
    return cut.rstrip() + "…", True


def _effective_top_k(query_norm: str, requested: int) -> int:#how many result
    """
    Default retrieval ~5 for precision; expand to 10–12 for list/HOD/faculty-style queries.
    Callers may pass higher top_k from tools — list queries still bump to at least 10.
    """
    req = max(1, min(int(requested), 20))
    if _LIST_QUERY.search(query_norm) or " list" in query_norm or query_norm.startswith("list "):#detect list queries
        return min(MAX_HITS_CAP, max(10, req))#if than result 10-12
    # Non-list: cap broad tool defaults (e.g. 10) down toward 5 unless list heuristics fire
    return min(8, max(1, min(req, 5)))#else 5


def search_institute_knowledge(query: str | None, top_k: int = 5) -> dict[str, Any]:#main function
    """
    Embed `query` and return passages from Chroma (structured hits: text, score, source, section).
    Applies similarity thresholding and dynamic top_k for list-style questions.
    """
    q_raw = (query or "").strip()
    if not q_raw:
        return {"ok": False, "error": "Empty query.", "hits": []}
    if not OPENAI_API_KEY:
        return {"ok": False, "error": "OPENAI_API_KEY is not set.", "hits": []}

    q = _normalize_query(q_raw)
    if not q:
        return {"ok": False, "error": "Empty query.", "hits": []}

    coll = _collection()
    if coll is None:
        return {
            "ok": False,
            "error": "Chroma collection not found. Rebuild the index.",
            "hint": "Run: python scripts/build_rag_index.py (from project root)",
            "hits": [],
        }
    try:
        n = coll.count()
    except Exception:
        n = 0
    if n == 0:
        return {
            "ok": False,
            "error": "Knowledge index is empty. Rebuild the index.",
            "hint": "Run: python scripts/build_rag_index.py (from project root)",
            "hits": [],
        }

    meta = _read_meta()
    embed_model = meta.get("embedding_model") or OPENAI_EMBEDDING_MODEL#done embedding

    try:
        qv = list(_embedding_vector(embed_model, q))
    except Exception as e:
        return {
            "ok": False,
            "error": f"Embedding request failed: {e}",
            "hint": "Check network, OPENAI_API_KEY, or increase OPENAI_TIMEOUT / OPENAI_CONNECT_TIMEOUT in .env",
            "hits": [],
        }

    k = _effective_top_k(q, top_k)
    fetch_n = max(1, min(n, k * QUERY_MULTIPLIER))

    raw = coll.query(
        query_embeddings=[qv],
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"],
    )

    ids0 = raw.get("ids") or []
    docs0 = raw.get("documents") or []
    meta0 = raw.get("metadatas") or []
    dist0 = raw.get("distances") or []
    ids = ids0[0] if ids0 else []
    documents = docs0[0] if docs0 else []
    metadatas = meta0[0] if meta0 else []
    distances = dist0[0] if dist0 else []

    scored: list[tuple[float, str, dict[str, Any], str, Any]] = []
    for i, doc_id in enumerate(ids):# Prevents index errors (very important for robustness)
        text_raw = documents[i] if i < len(documents) else ""
        md = metadatas[i] if i < len(metadatas) else {}
        dist = distances[i] if i < len(distances) else None
        score = 0.0
        if dist is not None:
            score = round(max(0.0, 1.0 - float(dist)), 4)
        if score < MIN_SCORE:
            continue
        src = ""
        section = ""
        if isinstance(md, dict):
            src = str(md.get("source", "") or "")
            section = str(md.get("section", "") or "")
        scored.append((score, text_raw, md if isinstance(md, dict) else {}, src, doc_id))

    scored.sort(key=lambda x: x[0], reverse=True)
    scored = scored[:k]

    hits: list[dict[str, Any]] = []
    for score, text_raw, md, src, doc_id in scored:
        text, was_trim = _trim_text(text_raw, MAX_CHARS_PER_HIT)
        section = ""
        if isinstance(md, dict):
            section = str(md.get("section", "") or "")
        hits.append(
            {
                "score": score,
                "source": src,
                "section": section,
                "text": text,
                "id": doc_id,
                "truncated": was_trim,
            }
        )

    note = (
        "Answer from these passages only; cite source paths. "
        "Several chunks may be returned — combine them; do not invent facts."
    )
    if not hits:
        return {
            "ok": True,
            "query": q_raw,
            "hits": [],
            "embedding_model": embed_model,
            "vector_store": "chromadb",
            "retrieval": "rag",
            "note": note,
            "warning": "No passages met the similarity threshold — treat as not found in knowledge base.",
        }

    return {
        "ok": True,
        "query": q_raw,
        "hits": hits,
        "embedding_model": embed_model,
        "vector_store": "chromadb",
        "retrieval": "rag",
        "note": note,
    }


def clear_index_cache() -> None:
    _chroma_client.cache_clear()
    _openai_client.cache_clear()
    _embedding_vector.cache_clear()
