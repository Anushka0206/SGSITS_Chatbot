"""
Institute Q&A: ChromaDB + RAG only.

- Runtime: `search_institute_knowledge` → `rag.search_institute_knowledge` (embed query, Chroma similarity search).
- No tools read `data/*.json` during chat; JSON is ingestion input for `scripts/build_rag_index.py` only.
- `save_user_message` appends to `data/user-data/` (feedback), not to Chroma.
"""
from __future__ import annotations

from typing import Any

from .rag import search_institute_knowledge


def tool_dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "search_institute_knowledge":
        try:
            tk = int(arguments.get("top_k") or 10)
        except (TypeError, ValueError):
            tk = 10
        tk = max(1, min(tk, 20))
        return search_institute_knowledge(arguments.get("query"), tk)
    if name == "save_user_message":
        kind = arguments.get("kind", "feedback")
        text = arguments.get("text", "")
        if kind not in ("feedback", "complaint", "query"):
            kind = "feedback"
        from .user_store import append_entry

        entry = append_entry(kind, text)  # type: ignore[arg-type]
        return {"ok": True, "saved": kind, "entry_id": entry.get("id")}
    return {"ok": False, "error": f"Unknown tool {name}"}


TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "search_institute_knowledge",
            "description": "ONLY way to answer SGSITS questions: Chroma RAG. Pass one strong query per user turn (e.g. 'faculty list Computer Engineering CSE', 'HOD CSE'). Use top_k 10–12 for long lists (faculty, syllabus).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look up, e.g. 'HOD Computer Engineering CSE', 'Machine Learning syllabus CSE', 'placement statistics'"},
                    "top_k": {
                        "type": "integer",
                        "description": "Number of passages to retrieve (default 10; use 12 for faculty/long lists, max 20)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_user_message",
            "description": "Save user feedback, complaint, or general query to user-data JSON files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["feedback", "complaint", "query"]},
                    "text": {"type": "string"},
                },
                "required": ["kind", "text"],
            },
        },
    },
]
