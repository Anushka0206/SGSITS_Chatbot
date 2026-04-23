from pathlib import Path

import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
# Read timeout for OpenAI requests (chat + embeddings). Default raised to reduce "timed out" on slow links.
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "240"))
# Fail fast if TCP/connect hangs (seconds)
OPENAI_CONNECT_TIMEOUT = float(os.getenv("OPENAI_CONNECT_TIMEOUT", "30"))

USER_DATA_DIR = DATA_ROOT / "user-data"


def openai_http_timeout():
    """httpx.Timeout: generous read, bounded connect (used by OpenAI SDK)."""
    try:
        import httpx

        return httpx.Timeout(
            OPENAI_TIMEOUT,
            connect=OPENAI_CONNECT_TIMEOUT,
            read=OPENAI_TIMEOUT,
            write=60.0,
            pool=30.0,
        )
    except Exception:
        return OPENAI_TIMEOUT
