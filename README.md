# SGSITS Chatbot

Streamlit assistant for SGSITS institute Q&A using **OpenAI** + **ChromaDB** retrieval (RAG). Answers are grounded in embedded content from JSON under `data/`.

## Requirements

- Python 3.10+ (recommended)
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Quick start

1. Clone the repository and enter the project folder.

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Copy the environment template and **fill in your own key** (never commit `.env`):

   ```bash
   copy .env.example .env
   ```

4. Build the vector index (required before first run; re-run after you change JSON under `data/`):

   ```bash
   python scripts/build_rag_index.py
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

## Architecture (Chroma + RAG)

| Step | What happens |
|------|----------------|
| **Ingest (offline)** | `python scripts/build_rag_index.py` reads text from `data/**/*.json`, chunks it, embeds with OpenAI, and stores vectors + text in Chroma under `data/.rag_index/`. |
| **Chat (runtime)** | The app embeds the user query, queries Chroma, and the model answers using the returned passages (RAG). |
| **Feedback** | Optional saves go to `data/user-data/*.json` (not part of the vector index). |

## Configuration (`.env`)

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required for chat and embeddings. |
| `OPENAI_MODEL` | Chat model (default: `gpt-4o-mini`). |
| `OPENAI_EMBEDDING_MODEL` | Embedding model (default: `text-embedding-3-small`). |
| `OPENAI_TIMEOUT` | Read timeout in seconds (default: `240`). |
| `OPENAI_CONNECT_TIMEOUT` | Connect timeout in seconds (default: `30`). |

See `.env.example` for a safe template (placeholders only).

---

## What to commit on Git (and what not to) — security

### Safe to commit (typical open-source / coursework upload)

- Application code: `app.py`, `chatbot/`, `scripts/`
- `requirements.txt`, `README.md`, `.gitignore`
- **`.env.example`** with **fake placeholders only** (no real keys)
- Institute **source** knowledge as JSON under `data/` *if* it is non-sensitive public information (syllabi, contacts published on the website, etc.). Review before pushing.

### Do **not** commit (secrets, generated data, private user content)

| Item | Why |
|------|-----|
| **`.env`** | Contains your real API key. |
| **`data/.rag_index/`** | Generated Chroma index; large, reproducible with the build script. |
| **`.venv/` / `venv/`** | Local Python environment; huge and machine-specific. |
| **`__pycache__/`**, `*.pyc` | Bytecode cache. |
| **`data/user-data/`** | May contain user messages or feedback — treat as personal/sensitive unless you have explicit consent to publish. |
| **Private PDFs or internal docs** | Only if not meant to be public. |
| **IDE/OS junk** | e.g. `.idea/`, `.vscode/` with personal tokens, `Thumbs.db`, `.DS_Store` (optional to ignore). |

Before your first public push, run:

```bash
git status
```

Confirm **no** `.env`, **no** `.rag_index`, and **no** real keys inside any tracked file.

### If a key was ever in Git or in a shared file

Revoke it in the [OpenAI API keys](https://platform.openai.com/api-keys) dashboard and create a new key. Keys in history or forks may still be exposed.
