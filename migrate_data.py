"""
Build SGSITS-CHATBOT/data hierarchy from legacy sources (texts/, json/, pdf/).
Run from repo: python scripts/migrate_data.py
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"
# Legacy sgsits_rag bundle (sibling under ml_project)
LEGACY_DATA = PROJECT_ROOT.parent / "1st_aparna" / "sgsits_rag" / "data"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def envelope(
    *,
    doc_id: str,
    category: str,
    title: str,
    content: dict[str, Any],
    tags: list[str] | None = None,
    source_files: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "kind": "sgsits_knowledge",
        "metadata": {
            "id": doc_id,
            "title": title,
            "category": category,
            "tags": tags or [],
            "source_files": source_files or [],
            "last_updated": utc_now(),
        },
        "content": content,
    }


# -----------------------------------------------------------------------------
# Text helpers
# -----------------------------------------------------------------------------

def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def parse_faculty_text(text: str) -> list[dict[str, str]]:
    """Parse faculty listing blocks into structured members."""
    blocks = re.split(r"\n\s*\n", text)
    out: list[dict[str, str]] = []
    section_headers = {
        "professors",
        "associate professors",
        "assistant professors",
        "professor",
    }
    for b in blocks:
        lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        if not lines:
            continue
        if len(lines) == 1 and lines[0].lower() in section_headers:
            continue
        if lines[0].lower() in section_headers:
            lines = lines[1:]
        if len(lines) < 2:
            continue
        name = lines[0]
        designation = lines[1] if len(lines) > 1 else ""
        qualification = " ".join(lines[2:]) if len(lines) > 2 else ""
        out.append(
            {
                "name": name,
                "designation": designation,
                "qualification": qualification.strip(),
            }
        )
    return out


def syllabus_record(stem: str, raw: str) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "source_stem": stem,
        "raw_text": raw,
    }
    m = re.search(r"Course Code:\s*([^\n]+)", raw, re.I)
    if m:
        rec["course_code"] = m.group(1).strip()
    m = re.search(r"Course Name:\s*([^\n]+)", raw, re.I)
    if m:
        rec["course_name"] = m.group(1).strip()
    m = re.search(r"DEPARTMENT:\s*([^\n]+)", raw, re.I)
    if m:
        rec["department"] = m.group(1).strip()
    m = re.search(r"ACADEMIC YEAR:\s*([^\n]+)", raw, re.I)
    if m:
        rec["academic_year"] = m.group(1).strip()
    return rec


def programs_table_lines(raw: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if "|" not in line or line.lower().startswith("program"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            rows.append(
                {
                    "program": parts[0],
                    "duration": parts[1],
                    "intake": parts[2],
                }
            )
    return rows


# -----------------------------------------------------------------------------
# Directory layout (all JSON targets)
# -----------------------------------------------------------------------------

def ensure_hierarchy() -> None:
    rel_dirs = [
        "home",
        "about",
        "admissions",
        "academics",
        "academics/exams",
        "departments",
        "students",
        "facilities",
        "placements",
        "research",
        "media/photo-gallery",
        "contact",
    ]
    for r in rel_dirs:
        (DATA_ROOT / r).mkdir(parents=True, exist_ok=True)


def placeholder(
    doc_id: str,
    category: str,
    title: str,
    note: str = "Awaiting curated content.",
) -> dict[str, Any]:
    return envelope(
        doc_id=doc_id,
        category=category,
        title=title,
        tags=[],
        content={"_note": note},
    )


def write_all_placeholders() -> None:
    """Minimal valid JSON for every leaf in the target tree so retrieval paths exist."""
    mapping: list[tuple[str, str, str, str]] = [
        ("home/notifications.json", "home_notifications", "home", "Notifications"),
        ("home/news-events.json", "home_news_events", "home", "News & events"),
        ("home/gallery.json", "home_gallery", "home", "Gallery highlights"),
        ("about/vision-mission.json", "about_vision_mission", "about", "Vision & mission"),
        ("about/director-message.json", "about_director", "about", "Director message"),
        ("about/policies.json", "about_policies", "about", "Policies"),
        ("admissions/ug.json", "adm_ug", "admissions", "Undergraduate admissions"),
        ("admissions/pg.json", "adm_pg", "admissions", "Postgraduate admissions"),
        ("admissions/phd.json", "adm_phd", "admissions", "Ph.D. admissions"),
        ("admissions/fee-policy.json", "adm_fee_policy", "admissions", "Fee policy"),
        ("academics/programs.json", "acad_programs", "academics", "Programs overview"),
        ("academics/academic-calendar.json", "acad_calendar", "academics", "Academic calendar"),
        ("academics/regulations.json", "acad_regulations", "academics", "Regulations"),
        ("academics/exams/results.json", "acad_exams_results", "academics/exams", "Exam results"),
        ("academics/exams/timetable.json", "acad_exams_timetable", "academics/exams", "Exam timetable"),
        ("academics/exams/notices.json", "acad_exams_notices", "academics/exams", "Exam notices"),
        ("students/activities.json", "stu_activities", "students", "Student activities"),
        ("students/scholarships.json", "stu_scholarships", "students", "Scholarships"),
        ("students/ncc-nss.json", "stu_ncc_nss", "students", "NCC / NSS"),
        ("students/skill-courses.json", "stu_skills", "students", "Skill courses"),
        ("facilities/library.json", "fac_library", "facilities", "Library"),
        ("facilities/hostel.json", "fac_hostel", "facilities", "Hostel"),
        ("facilities/labs.json", "fac_labs", "facilities", "Labs (general)"),
        ("facilities/sports.json", "fac_sports", "facilities", "Sports"),
        ("facilities/medical.json", "fac_medical", "facilities", "Medical"),
        ("research/projects.json", "res_projects", "research", "Research projects"),
        ("research/publications.json", "res_publications", "research", "Publications"),
        ("research/patents.json", "res_patents", "research", "Patents"),
        ("media/photo-gallery/campus-tour.json", "media_campus_tour", "media", "Campus tour"),
        ("media/photo-gallery/events.json", "media_events", "media", "Media events"),
        ("media/videos.json", "media_videos", "media", "Videos"),
        ("contact/address.json", "contact_address", "contact", "Address"),
        ("contact/directory.json", "contact_directory", "contact", "Directory"),
        ("contact/feedback.json", "contact_feedback", "contact", "Feedback"),
    ]
    for rel, doc_id, cat, title in mapping:
        p = DATA_ROOT / rel
        if not p.exists():
            write_json(p, placeholder(doc_id, cat, title))


def ingest_from_legacy() -> None:
    if not LEGACY_DATA.exists():
        print(f"Legacy data not found at {LEGACY_DATA}; skipping ingest.")
        return

    texts = LEGACY_DATA / "texts"
    json_dir = LEGACY_DATA / "json"

    # --- about: institute ---
    about_path = texts / "about_sgsits_general.txt"
    if about_path.exists():
        raw = read_text(about_path)
        write_json(
            DATA_ROOT / "about" / "institute.json",
            envelope(
                doc_id="about_institute",
                category="about",
                title="Institute overview",
                source_files=[str(about_path.relative_to(LEGACY_DATA))],
                tags=["history", "autonomy", "affiliation", "overview"],
                content={
                    "summary_paragraphs": [p.strip() for p in re.split(r"\n\s*\n+", raw) if p.strip()],
                    "raw_text": raw,
                },
            ),
        )

    # --- about: governance (academic council) ---
    gov_path = texts / "academic_council_general.txt"
    if gov_path.exists():
        raw = read_text(gov_path)
        write_json(
            DATA_ROOT / "about" / "governance.json",
            envelope(
                doc_id="about_governance",
                category="about",
                title="Governance & academic council",
                source_files=[str(gov_path.relative_to(LEGACY_DATA))],
                tags=["governance", "academic council", "statutory"],
                content={
                    "academic_council": {
                        "summary": raw.split("Composition of Academic Council:")[0].strip(),
                        "composition_text": raw,
                        "procedures_quotes": re.findall(r'"([^"]+)"', raw),
                    },
                    "raw_text": raw,
                },
            ),
        )

    # --- about: administration (lightweight; same schema family as governance) ---
    write_json(
        DATA_ROOT / "about" / "administration.json",
        envelope(
            doc_id="about_administration",
            category="about",
            title="Administration",
            tags=["administration"],
            content={
                "_note": "Add admin structure, deans, and offices when available.",
            },
        ),
    )

    # --- departments: faculty-only files (merge when multiple sources share one JSON) ---
    dept_map: dict[str, tuple[str, str, str]] = {
        "civil_faculty.txt": ("departments/civil.json", "Civil Engineering", "faculty"),
        "ece_faculty.txt": ("departments/electronics.json", "Electronics & Communication", "faculty_ece"),
        "ei__faculty.txt": ("departments/electronics.json", "Electronics & Instrumentation", "faculty_ei"),
        "eee_faculty.txt": ("departments/electrical.json", "Electrical Engineering", "faculty"),
        "it__faculty.txt": ("departments/it.json", "Information Technology", "faculty"),
        "mechanical__faculty.txt": ("departments/mechanical.json", "Mechanical Engineering", "faculty"),
        "pharmacy__faculty.txt": ("departments/others.json", "Pharmacy", "faculty"),
    }

    for fname, (rel, dept_name, faculty_key) in dept_map.items():
        fp = texts / fname
        if not fp.exists():
            continue
        raw = read_text(fp)
        members = parse_faculty_text(raw)
        path = DATA_ROOT / rel
        rel_legacy = str(fp.relative_to(LEGACY_DATA))
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            content = existing.setdefault("content", {})
            content[faculty_key] = members
            content.setdefault("faculty_source_files", []).append(rel_legacy)
            existing["metadata"]["last_updated"] = utc_now()
            existing["metadata"].setdefault("source_files", []).append(rel_legacy)
            if "department" not in content:
                content["department"] = dept_name
            write_json(path, existing)
        else:
            write_json(
                path,
                envelope(
                    doc_id=f"dept_{rel.replace('/', '_').replace('.json', '')}",
                    category="departments",
                    title=f"{dept_name} — department profile",
                    source_files=[rel_legacy],
                    tags=["department", "faculty", dept_name.lower().replace(" ", "_")],
                    content={
                        "department": dept_name,
                        faculty_key: members,
                        "programs": [],
                        "research_areas": [],
                        "laboratories": [],
                    },
                ),
            )

    # Normalize electronics.json: expose combined faculty for retrieval
    el_path = DATA_ROOT / "departments" / "electronics.json"
    if el_path.exists():
        with open(el_path, "r", encoding="utf-8") as f:
            el = json.load(f)
        c = el.get("content", {})
        combined: list[dict[str, str]] = []
        for key in ("faculty_ece", "faculty_ei"):
            for m in c.get(key, []) or []:
                entry = dict(m)
                entry["subdepartment"] = "ECE" if key == "faculty_ece" else "EI"
                combined.append(entry)
        if combined:
            c["faculty"] = combined
        el["content"] = c
        write_json(el_path, el)

    # --- CSE rich bundle → departments/computer.json ---
    cse_files = {
        "cse_curriculum.txt": ("curriculum",),
        "cse_specializations_overview.txt": ("specializations",),
        "cse_research_areas.txt": ("research_areas",),
        "cse_laboratories.txt": ("laboratories",),
        "cse_faculty.txt": ("faculty",),
    }
    extra: dict[str, Any] = {}
    srcs: list[str] = []
    for fname, keys in cse_files.items():
        fp = texts / fname
        if not fp.exists():
            continue
        raw = read_text(fp)
        srcs.append(str(fp.relative_to(LEGACY_DATA)))
        if "curriculum" in keys:
            extra["curriculum_notes"] = raw
        if "specializations" in keys:
            extra["specializations"] = [ln.strip("- ").strip() for ln in raw.splitlines() if ln.strip().startswith("-")]
        if "research_areas" in keys:
            extra["research_areas"] = [ln.strip("- ").strip() for ln in raw.splitlines() if ln.strip().startswith("-")]
        if "laboratories" in keys:
            extra["laboratories"] = [ln.strip("- ").strip() for ln in raw.splitlines() if ln.strip().startswith("-")]
        if "faculty" in keys:
            extra["faculty"] = parse_faculty_text(raw)

    prog_fp = texts / "cse_programs_overview"
    if prog_fp.exists():
        raw = read_text(prog_fp)
        srcs.append(str(prog_fp.relative_to(LEGACY_DATA)))
        extra["programs_table"] = programs_table_lines(raw)

    comp_path = DATA_ROOT / "departments" / "computer.json"
    write_json(
        comp_path,
        envelope(
            doc_id="dept_computer_engineering",
            category="departments",
            title="Computer Engineering — department profile",
            source_files=srcs,
            tags=["cse", "computer", "department", "faculty", "labs"],
            content={
                "department": "Computer Engineering",
                **extra,
            },
        ),
    )

    # --- academics: syllabus (all course syllabi text files) ---
    syllabus_names = [
        "data_structures_syllabus_cse.txt",
        "discrete_structures_syllabus_cse.txt",
        "operating_system_syllabus_cse.txt",
        "oops_syllabus_cse.txt",
        "computer_networks_syllabus.txt",
        "computer_architecture_syllabus_cse.txt",
    ]
    courses: list[dict[str, Any]] = []
    ssrc: list[str] = []
    for fname in syllabus_names:
        fp = texts / fname
        if not fp.exists():
            continue
        raw = read_text(fp)
        stem = fp.stem
        ssrc.append(str(fp.relative_to(LEGACY_DATA)))
        courses.append(syllabus_record(stem, raw))

    write_json(
        DATA_ROOT / "academics" / "syllabus.json",
        envelope(
            doc_id="academics_syllabus",
            category="academics",
            title="Syllabus repository",
            source_files=ssrc,
            tags=["syllabus", "courses", "cse"],
            content={
                "schema": "syllabus_bundle_v1",
                "courses": courses,
            },
        ),
    )

    # --- placements: split legacy overview ---
    po = json_dir / "placements_overview.json"
    if po.exists():
        with open(po, "r", encoding="utf-8") as f:
            legacy = json.load(f)
        meta = legacy.get("metadata", {})
        content = legacy.get("content", {})

        write_json(
            DATA_ROOT / "placements" / "statistics.json",
            envelope(
                doc_id="placements_statistics",
                category="placements",
                title="Placement statistics",
                source_files=[str(po.relative_to(LEGACY_DATA))],
                tags=["placement", "salary", "statistics"],
                content={
                    "placement_statistics": content.get("placement_statistics", {}),
                    "branch_wise_placements": content.get("branch_wise_placements", []),
                    "overview": content.get("overview", ""),
                },
            ),
        )

        write_json(
            DATA_ROOT / "placements" / "companies.json",
            envelope(
                doc_id="placements_companies",
                category="placements",
                title="Recruiters & companies",
                source_files=[str(po.relative_to(LEGACY_DATA))],
                tags=["companies", "recruiters"],
                content={
                    "top_recruiters": content.get("top_recruiters", {}),
                },
            ),
        )

        write_json(
            DATA_ROOT / "placements" / "training-cell.json",
            envelope(
                doc_id="placements_training_cell",
                category="placements",
                title="Training & placement cell",
                source_files=[str(po.relative_to(LEGACY_DATA))],
                tags=["tpo", "training", "process", "internships"],
                content={
                    "title": meta.get("title", "Training & Placement Cell"),
                    "source_url": meta.get("source", ""),
                    "placement_process": content.get("placement_process", []),
                    "pre_placement_training": content.get("pre_placement_training", []),
                    "internships": content.get("internships", {}),
                    "contact": content.get("contact", {}),
                },
            ),
        )


def merge_pdf_extractions() -> None:
    """If scripts/pdf_extract.py wrote sidecar files, merge into targets."""
    side = DATA_ROOT / "_pdf_extracts"
    if not side.exists():
        return
    for jf in side.glob("*.json"):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue
        target_rel = data.get("merge_into")
        if not target_rel:
            continue
        tp = DATA_ROOT / target_rel
        if not tp.exists():
            continue
        with open(tp, "r", encoding="utf-8") as f:
            target = json.load(f)
        block = data.get("content_block") or "pdf_extractions"
        target.setdefault("content", {})[block] = data.get("extraction", {})
        target["metadata"]["last_updated"] = utc_now()
        if "source_files" in data:
            target["metadata"].setdefault("source_files", []).extend(data["source_files"])
        write_json(tp, target)
        print(f"Merged PDF extraction into {tp.relative_to(PROJECT_ROOT)}")


def main() -> None:
    import sys

    ensure_hierarchy()
    write_all_placeholders()
    ingest_from_legacy()
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from clean_corpus_ingest import ingest_clean_corpus

        ingest_clean_corpus(LEGACY_DATA, DATA_ROOT, utc_now)
    except ImportError as e:
        print(f"clean_corpus_ingest: {e}")
    try:
        from pdf_extract import extract_all_pdfs  # type: ignore

        pdf_dir = LEGACY_DATA / "pdf"
        if pdf_dir.exists():
            extract_all_pdfs(pdf_dir, DATA_ROOT)
            merge_pdf_extractions()
    except ImportError:
        print("pdf_extract not available; install pdfplumber to process PDFs.")
    print(f"Done. Knowledge base root: {DATA_ROOT}")


if __name__ == "__main__":
    main()
