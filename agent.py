"""OpenAI chat with tool calling: Chroma RAG + optional user message save."""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .config import OPENAI_API_KEY, OPENAI_MODEL, openai_http_timeout
from .knowledge import TOOLS_OPENAI, tool_dispatch

SYSTEM_PROMPT = """You are the official SGSITS (Shri Govindram Seksaria Institute of Technology and Science, Indore) assistant.

Rules:
- **Any question about the institute** (departments, faculty, HOD, syllabus, placements, facilities, contacts, admissions, notices, etc.) MUST be answered using **search_institute_knowledge** first. Do not answer from memory or the open web.
- You may **only** state facts that appear in the retrieved passages (hits). If passages conflict, say so briefly and prefer the most specific source path.
- For **list-style** questions (faculty lists, all departments, HOD list, complete syllabus topics, contacts), call search_institute_knowledge with **top_k=12** (or 10–12) so enough chunks are retrieved.
- For short factual questions, **one** search with a clear query is enough; use **top_k=5** unless it is clearly a list. Do **not** run a second search unless the first returned ok=false or was empty.
- For feedback / complaints / messages to the institute only: use save_user_message (no RAG required).
- If the tool returns **no hits**, ok=false, or a warning that nothing met the similarity threshold, reply exactly: **Not found in knowledge base** (you may add one short sentence inviting the user to rephrase). Do not guess.
- Be concise and polite. Mention source file paths from the hits when helpful.
"""


def run_turn(
    messages: list[dict[str, Any]], client: OpenAI | None = None
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """
    Append assistant reply to conversation. `messages` is user/assistant history (no system).
    Returns (updated_messages_with_latest_assistant, assistant_text, sidecar).
    sidecar is kept for API compatibility (Chroma-only flow uses empty sidecar).
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env in the SGSITS-CHATBOT folder.")

    client = client or OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=openai_http_timeout(),
        max_retries=1,
    )
    msgs: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
    sidecar: dict[str, Any] = {}

    max_loops = 5
    for _ in range(max_loops):
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            tools=TOOLS_OPENAI,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            msgs.append(
                {
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = tool_dispatch(tc.function.name, args)
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            continue

        text = (msg.content or "").strip()
        new_hist = messages + [{"role": "assistant", "content": text}]
        return new_hist, text, sidecar

    text = "Sorry, the assistant could not finish the request (tool loop limit)."
    return messages + [{"role": "assistant", "content": text}], text, sidecar
