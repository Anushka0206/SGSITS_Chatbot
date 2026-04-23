"""Append-only user submissions: feedback, complaints, queries."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .config import USER_DATA_DIR

Kind = Literal["feedback", "complaint", "query"]


def _ensure_files() -> None:
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("feedback.json", "complaints.json", "queries.json"):
        p = USER_DATA_DIR / name
        if not p.exists():
            p.write_text(
                json.dumps({"entries": []}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


def _path_for(kind: Kind) -> Path:
    _ensure_files()
    mapping = {
        "feedback": USER_DATA_DIR / "feedback.json",
        "complaint": USER_DATA_DIR / "complaints.json",
        "query": USER_DATA_DIR / "queries.json",
    }
    return mapping[kind]


def append_entry(kind: Kind, text: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    path = _path_for(kind)
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = {
        "id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "text": text.strip(),
        **(extra or {}),
    }
    data.setdefault("entries", []).append(entry)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return entry
