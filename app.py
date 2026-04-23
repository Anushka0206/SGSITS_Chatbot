"""
SGSITS Chatbot — Streamlit UI. Loads OPENAI_API_KEY from .env
What it does: OpenAI answers only from ChromaDB semantic search (search_institute_knowledge) over embedded institute JSON; rebuild index after data changes with scripts/build_rag_index.py.

Run: pip install -r requirements.txt && streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from openai import APIConnectionError, APITimeoutError, RateLimitError

from chatbot.agent import run_turn
from chatbot.config import DATA_ROOT, OPENAI_API_KEY, OPENAI_MODEL
from chatbot.rag import index_exists, knowledge_base_outdated

st.set_page_config(page_title="SGSITS Assistant", page_icon="🎓", layout="centered")

st.title("SGSITS Assistant")
st.caption(
    "**How it works:** All institute answers come from **ChromaDB** only (`search_institute_knowledge`). Rebuild the index with `python scripts/build_rag_index.py` after editing files under `data/`."
)
st.caption(f"Data: `{DATA_ROOT}` · Model: `{OPENAI_MODEL}`")

# Index health: missing/empty Chroma vs stale JSON relative to last build (meta.json)
if not index_exists():
    st.error(
        "Knowledge index missing or empty. Run `python scripts/build_rag_index.py` from the project root, then refresh this page."
    )
elif knowledge_base_outdated():
    st.warning("⚠️ Knowledge base outdated. Rebuild index.")

if not OPENAI_API_KEY:
    st.error("Set `OPENAI_API_KEY` in a `.env` file in the SGSITS-CHATBOT folder, then restart.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "turn_counter" not in st.session_state:
    st.session_state.turn_counter = 0
if "last_sidecar" not in st.session_state:
    st.session_state.last_sidecar = {}

prompt = st.chat_input("Ask about syllabus, HOD, faculty, placements, facilities, gallery, or leave feedback…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt:
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                conv = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                new_hist, reply, sidecar = run_turn(conv)
                st.markdown(reply)
                st.session_state.messages = [
                    {"role": m["role"], "content": m["content"]} for m in new_hist
                ]
                st.session_state.turn_counter += 1
                st.session_state.last_sidecar = sidecar
            except APITimeoutError:
                st.error(
                    "Request timed out (OpenAI). Check your network, try again, or set a higher "
                    "`OPENAI_TIMEOUT` (seconds) in `.env` (default 120)."
                )
            except APIConnectionError as e:
                st.error(f"Connection error: {e}")
            except RateLimitError as e:
                st.error(f"Rate limited: {e}")
            except Exception as e:
                st.error(str(e))

sc = st.session_state.get("last_sidecar") or {}
dl = sc.get("downloads") or []
links = sc.get("pdf_links") or []
if dl or links:
    st.divider()
    st.subheader("Documents (last answer)")
    tc = st.session_state.turn_counter
    for i, d in enumerate(dl):
        p = d.get("path")
        if not p:
            continue
        try:
            with open(p, "rb") as f:
                data = f.read()
            st.download_button(
                label=d.get("label", "Download PDF"),
                data=data,
                file_name=d.get("file_name", "document.pdf"),
                key=f"dl_{tc}_{i}",
                mime="application/pdf",
            )
        except OSError:
            st.caption(f"Could not read: {p}")
    for j, link in enumerate(links):
        st.link_button(link.get("label", "Open PDF"), link.get("url", "#"), key=f"lnk_{tc}_{j}")
