"""
Ingest institute text into ChromaDB (the only vector store used at answer time).

Chunking: ~400 token targets (~1600 chars), ~80 token overlap (~320 chars), sentence-safe.
Priority chunks for HOD rows + CSE faculty improve recall for structured questions.

Run from project root:  python scripts/build_rag_index.py
Requires OPENAI_API_KEY in .env (embedding API cost).
"""
from __future__ import annotations
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent#project path
sys.path.insert(0, str(ROOT))#add project root to the python path
os.chdir(ROOT)#Makes project root the current directory

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import chromadb  # noqa: E402
from openai import OpenAI  # noqa: E402

from chatbot.config import openai_http_timeout  # noqa: E402

DATA_ROOT = ROOT / "data"
INDEX_DIR = DATA_ROOT / ".rag_index"
CHROMA_DIR = INDEX_DIR / "chroma"
META_FILE = INDEX_DIR / "meta.json"
COLLECTION_NAME = "sgsits_knowledge"

EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
# ~400 tokens / ~80 tokens overlap (approx 4 chars/token for Latin text)
CHUNK_CHARS = 1600
OVERLAP_CHARS = 320#prevent context break
MIN_STRING = 40
BATCH = 48#Process 48 chunks per API call


def _source_json_max_mtime() -> float:
    """Latest mtime among ingest JSON files (for freshness checks in UI)."""
    m = 0.0
    for p in DATA_ROOT.rglob("*.json"):
        try:
            rel = p.relative_to(DATA_ROOT)
        except ValueError:
            continue
        parts = rel.parts
        if parts and parts[0] == "user-data":
            continue
        if ".rag_index" in parts:
            continue
        try:
            m = max(m, p.stat().st_mtime)#Get last modified time:
        except OSError:
            pass
    return m


def _split_sentences(text: str) -> list[str]:
    """Split into sentence-like units without breaking mid-clause where possible."""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    # Line breaks often separate list items (faculty lines, etc.)
    parts: list[str] = []
    for block in re.split(r"\n\s*\n+", text):
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(line) < 300:
                parts.append(line)
            else:
                sub = re.split(r"(?<=[.!?])\s+", line)
                parts.extend(s.strip() for s in sub if s.strip())
    if len(parts) < 2:
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return parts if parts else [text]


def _chunk_sentence_aware(text: str) -> list[str]:
    """Pack sentences into chunks <= CHUNK_CHARS with OVERLAP_CHARS carry between chunks."""
    units = _split_sentences(text)
    if not units:
        return []
    chunks: list[str] = []
    buf: list[str] = []
    size = 0

    for u in units:
        add = len(u) + (1 if buf else 0)
        if size + add <= CHUNK_CHARS:
            buf.append(u)
            size += add
            continue
        if buf:#If buffer already has content 
            chunks.append(" ".join(buf))
            # Overlap: carry tail sentences until ~OVERLAP_CHARS (keep list lines intact)
            carry: list[str] = []
            cs = 0
            for x in reversed(buf):
                #Collect sentences until overlap limit is reached
                if cs + len(x) > OVERLAP_CHARS and carry:
                    break
                carry.insert(0, x)
                cs += len(x) + 1
            buf = carry + [u]#New chunk begins with:tail of previous chunk (overlap)current sentence
            size = sum(len(x) + 1 for x in buf) - (1 if buf else 0)#Updates current buffer size
        else:
            # single huge unit — hard slice on word boundary
            if len(u) > CHUNK_CHARS:#Huge sentence
                w = 0
                while w < len(u):
                    piece = u[w : w + CHUNK_CHARS]
                    if w + CHUNK_CHARS < len(u):
                        sp = piece.rfind(" ")
                        if sp > CHUNK_CHARS // 2:#if space so split in second half
                            piece = piece[:sp]#Move pointer accordingly
                            w += sp + 1
                        else:
                            w += CHUNK_CHARS#No good space → just cut
                    else:
                        w = len(u)
                    chunks.append(piece.strip())
            else:#start new chunk with this sentence
                buf = [u]
                size = len(u)
    if buf:#Add any remaining buffered text
        chunks.append(" ".join(buf))
    return [c for c in chunks if c.strip()]

#Take all values yielded by the recursive call and pass them upward
#this function traverse the schema 
def _iter_strings(obj: object, min_len: int = MIN_STRING):
    if isinstance(obj, str):
        if len(obj) >= min_len:
            yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v, min_len)#Delegates yielding to another generator (recursive call)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_strings(v, min_len)


def collect_priority_chunks() -> list[dict[str, str]]:
    """
    High-signal chunks for HOD table and CSE faculty lists (better retrieval than blob-only).
    """
    rows: list[dict[str, str]] = []
    admin = DATA_ROOT / "about" / "administration.json"
    if admin.is_file():
        try:
            doc = json.loads(admin.read_text(encoding="utf-8"))
            heads = doc.get("content", {}).get("department_heads") or []
            for i, row in enumerate(heads):
                if not isinstance(row, dict):
                    continue
                dept = str(row.get("Department", "")).strip()
                hod = str(row.get("HOD Name", "")).strip()
                des = str(row.get("Designation", "")).strip()
                if not dept:
                    continue
                text = (
                    f"Head of Department (HOD) for {dept}: {hod}. Designation: {des}. "
                    f"Institute: SGSITS Indore. Keywords: HOD, head of department, department head."
                )
                cid = f"priority__admin_hod_{i}".replace(" ", "_")
                rows.append(
                    {
                        "source": "about/administration.json",
                        "section": "department_heads",
                        "chunk_id": cid,
                        "text": text,
                    }
                )
        except (json.JSONDecodeError, OSError):
            pass

    comp = DATA_ROOT / "departments" / "computer.json"
    if comp.is_file():
        try:
            doc = json.loads(comp.read_text(encoding="utf-8"))
            fac = doc.get("content", {}).get("faculty") or []
            lines: list[str] = []
            for f in fac:
                if not isinstance(f, dict):
                    continue
                nm = str(f.get("name", "")).strip()
                ds = str(f.get("designation", "")).strip()
                if nm:
                    lines.append(f"{nm} — {ds}" if ds else nm)
            if lines:
                blob = (
                    "Computer Engineering (CSE) department — complete faculty list at SGSITS Indore:\n"
                    + "\n".join(lines)
                )
                for i, piece in enumerate(_chunk_sentence_aware(blob)):
                    rows.append(
                        {
                            "source": "departments/computer.json",
                            "section": "faculty",
                            "chunk_id": f"priority__cse_faculty_{i}",
                            "text": piece,
                        }
                    )
        except (json.JSONDecodeError, OSError):
            pass

    return rows


def collect_regular_chunks() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(DATA_ROOT.rglob("*.json")):
        rel = path.relative_to(DATA_ROOT)
        parts = rel.parts
        if parts[0] == "user-data":
            continue
        if ".rag_index" in parts:
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        strings = list(_iter_strings(doc))
        if not strings:
            continue
        blob = "\n\n".join(strings)#Merges all text into one large document
        if len(blob) < 80:#Skip Tiny Content
            continue
        src = str(rel).replace("\\", "/")#Normalize Source Path
        for i, piece in enumerate(_chunk_sentence_aware(blob)):
            cid = f"{src.replace('/', '__')}__{i}"
            rows.append(
                {
                    "source": src,
                    "section": "content",
                    "chunk_id": cid,
                    "text": piece,
                }
            )
    return rows


def collect_chunks() -> list[dict[str, str]]:
    # Priority first so IDs are stable; regular chunks follow
    return collect_priority_chunks() + collect_regular_chunks()


def main() -> None:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        print("Set OPENAI_API_KEY in .env", file=sys.stderr)
        sys.exit(1)

    rows = collect_chunks()
    if not rows:
        print("No text extracted from data/**/*.json", file=sys.stderr)
        sys.exit(1)

    print(f"Chunks to embed: {len(rows)}  model={EMBED_MODEL}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))#Creates a persistent vector database
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    oai = OpenAI(api_key=key, timeout=openai_http_timeout(), max_retries=1)
    total = 0
    source_max = _source_json_max_mtime()

    for i in range(0, len(rows), BATCH):
        batch = rows[i : i + BATCH]#chunks proceeds in batch not all at a time
        texts = [b["text"] for b in batch]
        resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
        ids = [b["chunk_id"] for b in batch]
        embeddings = [resp.data[j].embedding for j in range(len(batch))]
        metadatas = [
            {
                "source": b["source"],
                "section": b.get("section", "content"),
                "chunk_id": b["chunk_id"],
            }
            for b in batch
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        total += len(batch)
        print(f"  added {total}/{len(rows)}")

    meta = {
        "embedding_model": EMBED_MODEL,
        "chunks": total,
        "backend": "chromadb",
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "source_data_max_mtime": source_max,
    }
    META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Chroma collection '{COLLECTION_NAME}' at {CHROMA_DIR} ({total} vectors)")
    print(f"meta.json: indexed_at={meta['indexed_at']} source_data_max_mtime={source_max}")

    legacy = INDEX_DIR / "embeddings.json"
    if legacy.is_file():
        try:
            legacy.unlink()
            print(f"Removed legacy {legacy.name}")
        except OSError:
            pass

    try:
        from chatbot.rag import clear_index_cache

        clear_index_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
