"""
Extract text + tables from PDFs into structured JSON for RAG-friendly retrieval.
Uses pdfplumber (vector text + table detection). Image-only pages are flagged for OCR follow-up.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Map PDF filename stem (lowercase) → relative path under data/ to merge into
PDF_MERGE_TARGETS: dict[str, str] = {
    "fee_structure": "admissions/fee-policy.json",
    "fee-policy": "admissions/fee-policy.json",
    "fee_policy": "admissions/fee-policy.json",
    "admissions": "admissions/ug.json",
    "timetable": "academics/exams/timetable.json",
    "exam_timetable": "academics/exams/timetable.json",
    "results": "academics/exams/results.json",
    "exam_notices": "academics/exams/notices.json",
    "notices": "academics/exams/notices.json",
    "regulations": "academics/regulations.json",
    "calendar": "academics/academic-calendar.json",
    "academic_calendar": "academics/academic-calendar.json",
    "library": "facilities/library.json",
    "hostel": "facilities/hostel.json",
    "placement": "placements/training-cell.json",
    "placements": "placements/training-cell.json",
}


def _clean(text: str) -> str:
    text = re.sub(r"Page\s*\d+\s*(of\s*\d+)?", "", text, flags=re.I)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_pdf(pdf_path: Path) -> dict[str, Any]:
    import pdfplumber

    pages_out: list[dict[str, Any]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            tables_raw: list[list[list[str | None]]] = []
            table_text_blocks: list[str] = []
            try:
                for t in page.find_tables() or []:
                    extracted = t.extract()
                    if extracted:
                        tables_raw.append(extracted)
                        rows = []
                        for row in extracted:
                            cells = [str(c).strip() if c else "" for c in row]
                            cells = [c for c in cells if c]
                            if cells:
                                rows.append(" | ".join(cells))
                        if rows:
                            table_text_blocks.append("\n".join(rows))
            except Exception:
                pass

            text = page.extract_text() or ""
            text = _clean(text)
            combined = "\n\n".join([*table_text_blocks, text]).strip()
            low_text = len(re.sub(r"\s+", "", combined)) < 40
            pages_out.append(
                {
                    "page": i + 1,
                    "text": text,
                    "tables": tables_raw,
                    "combined_text": combined,
                    "likely_image_or_scan": low_text and not tables_raw,
                }
            )

    return {
        "source_pdf": pdf_path.name,
        "page_count": len(pages_out),
        "pages": pages_out,
    }


def extract_all_pdfs(pdf_dir: Path, data_root: Path) -> None:
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        return
    out_dir = data_root / "_pdf_extracts"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        stem = pdf_path.stem.lower()
        stem_norm = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
        extraction = extract_pdf(pdf_path)
        merge_into = PDF_MERGE_TARGETS.get(stem_norm) or PDF_MERGE_TARGETS.get(
            stem_norm.replace("_", "-"), ""
        )

        sidecar = {
            "merge_into": merge_into or None,
            "content_block": "pdf_extraction",
            "source_files": [str(pdf_path.name)],
            "extraction": extraction,
        }
        dest = out_dir / f"{stem_norm}.json"
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=2)
        print(f"PDF extracted → {dest.relative_to(data_root.parent)}")


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parent.parent
    pd = root.parent / "1st_aparna" / "sgsits_rag" / "data" / "pdf"
    extract_all_pdfs(pd, root / "data")
