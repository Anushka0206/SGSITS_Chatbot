"""Streamlit entry — delegates to app.py so `streamlit run streamlit_app.py` works."""
from __future__ import annotations

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "app.py"), run_name="__main__")
