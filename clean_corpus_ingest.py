"""
Ingest structured knowledge from sgsits_rag/data__/data/clean/*.txt (full site scrapes).
This is the primary source for academic calendar, programs list, regulations, admissions, etc.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def strip_leading_meta(text: str) -> str:
    text = re.sub(r"^SOURCE:\s*.+\n", "", text, flags=re.I | re.M)
    text = re.sub(r"^PAGE TITLE:\s*.+\n", "", text, flags=re.I | re.M)
    return text.strip()


def split_h3_sections(text: str) -> dict[str, str]:
    """Split markdown-ish content on ### headings."""
    text = strip_leading_meta(text)
    parts = re.split(r"\n(?=###\s)", text)
    out: dict[str, str] = {}
    for p in parts:
        p = p.strip()
        if not p.startswith("###"):
            continue
        m = re.match(r"###\s+([^\n]+)\n?(.*)$", p, re.DOTALL)
        if m:
            out[m.group(1).strip()] = m.group(2).strip()
    return out


def split_h2_sections(text: str) -> dict[str, str]:
    """Split on ## headings (level-2)."""
    text = strip_leading_meta(text)
    parts = re.split(r"\n(?=##\s)", text)
    out: dict[str, str] = {}
    for p in parts:
        p = p.strip()
        if not p.startswith("##"):
            continue
        m = re.match(r"##\s+([^\n]+)\n?(.*)$", p, re.DOTALL)
        if m:
            out[m.group(1).strip()] = m.group(2).strip()
    return out


def extract_pipe_table_from_text(text: str, header_must_contain: str) -> list[dict[str, str]]:
    """Find a | table whose header line contains the given substring."""
    idx = text.find(header_must_contain)
    if idx < 0:
        return []
    chunk = text[idx : idx + 8000]
    _, rows = parse_pipe_table(chunk)
    return rows


def bullets(text: str) -> list[str]:
    rows: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            rows.append(line[2:].strip())
        elif line.startswith("• "):
            rows.append(line[2:].strip())
    return rows


def parse_pipe_table(block: str) -> tuple[list[str], list[dict[str, str]]]:
    """Parse markdown tables using | as separator."""
    lines = [ln.strip() for ln in block.splitlines() if ln.strip() and "|" in ln]
    if not lines:
        return [], []
    header_line = lines[0]
    if re.match(r"^[-:| ]+$", lines[1] if len(lines) > 1 else ""):
        data_lines = lines[2:]
    else:
        data_lines = lines[1:]
    headers = [c.strip() for c in header_line.split("|") if c.strip()]
    rows_out: list[dict[str, str]] = []
    for ln in data_lines:
        if re.match(r"^[-:| ]+$", ln):
            continue
        cells = [c.strip() for c in ln.split("|")]
        cells = [c for c in cells if c != ""]
        if len(cells) < 2:
            continue
        row: dict[str, str] = {}
        for i, h in enumerate(headers):
            row[h] = cells[i] if i < len(cells) else ""
        rows_out.append(row)
    return headers, rows_out


def parse_semester_calendar(section_text: str) -> dict[str, Any]:
    """Turn Academic Calendar bullets into structured fields."""
    data: dict[str, Any] = {
        "semesters": [],
        "breaks_vacations": [],
        "raw_bullets": bullets(section_text),
    }
    for b in data["raw_bullets"]:
        if ":" in b:
            name, rest = b.split(":", 1)
            entry = {"label": name.strip(), "period": rest.strip()}
            low = name.lower()
            if "semester" in low:
                data["semesters"].append(entry)
            elif "vacation" in low or "break" in low:
                data["breaks_vacations"].append(entry)
        else:
            data.setdefault("other_points", []).append(b)
    return data


def envelope(
    doc_id: str,
    category: str,
    title: str,
    content: dict[str, Any],
    tags: list[str],
    source_files: list[str],
    utc_now_fn,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "kind": "sgsits_knowledge",
        "metadata": {
            "id": doc_id,
            "title": title,
            "category": category,
            "tags": tags,
            "source_files": source_files,
            "last_updated": utc_now_fn(),
        },
        "content": content,
    }


def ingest_clean_corpus(legacy_data: Path, data_root: Path, utc_now_fn) -> None:
    clean = legacy_data.parent / "data__" / "data" / "clean"
    if not clean.exists():
        print(f"Clean corpus not found at {clean}; skipping full-site ingest.")
        return

    rel = lambda p: str(Path(p).relative_to(legacy_data.parent))

    # ------------------------------------------------------------------ academics
    ap = clean / "academics" / "academics.txt"
    if ap.exists():
        raw = ap.read_text(encoding="utf-8", errors="replace")
        src = [rel(ap)]
        h3 = split_h3_sections(raw)
        overview = h3.get("Academic Programs Overview", "")
        programs_body = h3.get("Complete List of Programs", "")
        cal_body = h3.get("Academic Calendar", "")
        exam_body = h3.get("Examination System", "")
        gov_body = h3.get("Academic Governance", "")
        auto_body = h3.get("Autonomous Status Benefits", "")
        schemes_body = h3.get("Schemes and Syllabi", "")

        _, program_rows = parse_pipe_table(programs_body)

        cal_struct = parse_semester_calendar(cal_body)
        data_root.joinpath("academics", "academic-calendar.json").write_text(
            json.dumps(
                envelope(
                    "academics_calendar",
                    "academics",
                    "Academic calendar",
                    {
                        "schema": "academic_calendar_v1",
                        "source_page": "https://www.sgsits.ac.in/academics",
                        "official_links": {
                            "calendar": "https://www.sgsits.ac.in/calendar",
                            "timetable": "https://www.sgsits.ac.in/timetable",
                            "examinations": "https://www.sgsits.ac.in/examinations",
                            "results": "https://www.sgsits.ac.in/results",
                        },
                        "semesters": cal_struct.get("semesters", []),
                        "breaks_and_vacations": cal_struct.get("breaks_vacations", []),
                        "raw_calendar_text": cal_body.strip(),
                        "bullet_points": cal_struct.get("raw_bullets", []),
                    },
                    ["calendar", "semester", "academics"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        exam_headers, exam_table = parse_pipe_table(exam_body)
        exam_info: dict[str, Any] = {
            "raw_examination_section": exam_body.strip(),
            "bullet_points": bullets(exam_body),
        }
        if exam_table:
            exam_info["structured_rows"] = exam_table

        data_root.joinpath("academics", "programs.json").write_text(
            json.dumps(
                envelope(
                    "academics_programs",
                    "academics",
                    "Academic programs",
                    {
                        "schema": "programs_overview_v1",
                        "source_page": "https://www.sgsits.ac.in/academics",
                        "overview": overview.strip(),
                        "programs": program_rows,
                        "schemes_and_syllabi": schemes_body.strip(),
                    },
                    ["programs", "ug", "pg", "phd"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        data_root.joinpath("academics", "regulations.json").write_text(
            json.dumps(
                envelope(
                    "academics_regulations",
                    "academics",
                    "Examination system & academic regulations",
                    {
                        "schema": "academic_regulations_v1",
                        "source_page": "https://www.sgsits.ac.in/academics",
                        "examination_system": exam_info,
                        "academic_governance": {
                            "raw_text": gov_body.strip(),
                            "bullet_points": bullets(gov_body),
                        },
                        "autonomous_status_benefits": {
                            "raw_text": auto_body.strip(),
                            "bullet_points": bullets(auto_body),
                        },
                    },
                    ["examination", "cie", "ese", "cgpa", "regulations"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        syl_path = data_root / "academics" / "syllabus.json"
        if syl_path.exists() and schemes_body.strip():
            doc = json.loads(syl_path.read_text(encoding="utf-8"))
            doc.setdefault("content", {})["schemes_and_syllabi_note"] = schemes_body.strip()
            doc["metadata"]["last_updated"] = utc_now_fn()
            doc["metadata"].setdefault("source_files", []).extend(src)
            syl_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ about / institute & vision
    inst = clean / "about" / "institute_overview.txt"
    if inst.exists():
        raw = inst.read_text(encoding="utf-8", errors="replace")
        src = [rel(inst)]
        body = strip_leading_meta(raw)
        vision_mission: dict[str, Any] = {"vision": "", "mission": [], "highlights": []}
        lines = body.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("To be a centre"):
                vision_mission["vision"] = line.strip()
                for j in range(i + 1, min(i + 10, len(lines))):
                    ln = lines[j].strip()
                    if ln.startswith("- To ") or ln.startswith("- To "):
                        vision_mission["mission"].append(ln.lstrip("- ").strip())
                    elif ln.startswith("- "):
                        vision_mission["mission"].append(ln[2:].strip())
                break
        for line in lines:
            if re.match(r"^- Year of Establishment:", line):
                vision_mission["highlights"].append(line.strip().lstrip("- ").strip())
            if re.match(r"^- Type:", line):
                vision_mission["highlights"].append(line.strip().lstrip("- ").strip())
            if re.match(r"^- NAAC", line) or re.match(r"^- NIRF", line):
                vision_mission["highlights"].append(line.strip().lstrip("- ").strip())

        data_root.joinpath("about", "institute.json").write_text(
            json.dumps(
                envelope(
                    "about_institute_full",
                    "about",
                    "About SGSITS",
                    {
                        "schema": "institute_profile_v1",
                        "source_page": "https://www.sgsits.ac.in/about",
                        "full_text": body,
                        "history_and_establishment": split_h2_sections(body).get(
                            "History and Establishment", ""
                        ),
                        "affiliation_and_governance": split_h2_sections(body).get(
                            "Affiliation and Governance", ""
                        ),
                    },
                    ["about", "history", "institute"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        data_root.joinpath("about", "vision-mission.json").write_text(
            json.dumps(
                envelope(
                    "about_vision_mission",
                    "about",
                    "Vision & mission",
                    {
                        "schema": "vision_mission_v1",
                        "vision": vision_mission.get("vision", ""),
                        "mission": vision_mission.get("mission", []),
                        "key_facts": vision_mission.get("highlights", []),
                        "raw_context": body,
                    },
                    ["vision", "mission", "naac", "nirf"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ administration & director
    adm = clean / "administration" / "administration.txt"
    if adm.exists():
        raw = adm.read_text(encoding="utf-8", errors="replace")
        src = [rel(adm)]
        body = strip_leading_meta(raw)
        h2 = split_h2_sections(body)
        admin_block = h2.get("Administration", "")
        director_intro = admin_block.split("###")[0].strip() if admin_block else ""
        m_dean = re.search(
            r"###\s*Dean Administration\s*\n(.*?)(?=###\s|\Z)",
            admin_block,
            re.DOTALL | re.I,
        )
        m_gov = re.search(
            r"###\s*Governing Body\s*\n(.*?)(?=###\s|\Z)",
            admin_block,
            re.DOTALL | re.I,
        )
        dean_txt = (m_dean.group(1).strip() if m_dean else "")
        gov_txt = (m_gov.group(1).strip() if m_gov else "")
        hod_rows = extract_pipe_table_from_text(body, "Department | HOD Name")
        structure_txt = h2.get("Administrative Structure", "").strip()
        officers_txt = h2.get("Key Administrative Officers", "")

        data_root.joinpath("about", "administration.json").write_text(
            json.dumps(
                envelope(
                    "about_administration_full",
                    "about",
                    "Administration",
                    {
                        "schema": "administration_v1",
                        "source_page": "https://www.sgsits.ac.in/administration",
                        "director_profile": bullets(director_intro),
                        "dean_administration": bullets(dean_txt),
                        "governing_body": gov_txt,
                        "department_heads": hod_rows,
                        "administrative_structure": structure_txt,
                        "key_administrative_officers": bullets(officers_txt),
                        "raw_text": body,
                    },
                    ["administration", "director", "hod"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        data_root.joinpath("about", "director-message.json").write_text(
            json.dumps(
                envelope(
                    "about_director_message",
                    "about",
                    "Director",
                    {
                        "schema": "director_profile_v1",
                        "source_page": "https://www.sgsits.ac.in/administration",
                        "profile_bullets": bullets(director_intro),
                        "raw_administration_intro": director_intro,
                    },
                    ["director", "leadership"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ admissions
    adf = clean / "admissions" / "admissions.txt"
    if adf.exists():
        raw = adf.read_text(encoding="utf-8", errors="replace")
        src = [rel(adf)]
        h3 = split_h3_sections(raw)
        common = {
            "seat_reservation": h3.get("Seat Reservation Policy", ""),
            "important_dates": h3.get("Important Dates (Approximate)", ""),
            "documents_required": h3.get("Documents Required for Admission", ""),
            "contact": h3.get("Contact for Admission Queries", ""),
        }
        data_root.joinpath("admissions", "ug.json").write_text(
            json.dumps(
                envelope(
                    "admissions_ug",
                    "admissions",
                    "Undergraduate admissions",
                    {
                        "schema": "admissions_ug_v1",
                        "source_page": "https://www.sgsits.ac.in/admissions",
                        "ug_sections": {
                            "b_e_b_tech": h3.get("Undergraduate (UG) Programs", ""),
                        },
                        "shared_information": common,
                        "raw_text": strip_leading_meta(raw),
                    },
                    ["jee", "dte", "ug", "btech"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        data_root.joinpath("admissions", "pg.json").write_text(
            json.dumps(
                envelope(
                    "admissions_pg",
                    "admissions",
                    "Postgraduate admissions",
                    {
                        "schema": "admissions_pg_v1",
                        "source_page": "https://www.sgsits.ac.in/admissions",
                        "pg_section": h3.get("Postgraduate (PG) Programs", ""),
                        "shared_information": common,
                        "raw_text": strip_leading_meta(raw),
                    },
                    ["gate", "cmat", "pg", "mtech", "mba", "mca"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        data_root.joinpath("admissions", "phd.json").write_text(
            json.dumps(
                envelope(
                    "admissions_phd",
                    "admissions",
                    "Ph.D. admissions",
                    {
                        "schema": "admissions_phd_v1",
                        "source_page": "https://www.sgsits.ac.in/admissions",
                        "phd_section": h3.get("Ph.D. Programs", ""),
                        "shared_information": common,
                        "raw_text": strip_leading_meta(raw),
                    },
                    ["phd", "research", "admission"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    fee = clean / "admissions" / "fee_structure.txt"
    if fee.exists():
        raw = fee.read_text(encoding="utf-8", errors="replace")
        src = [rel(fee)]
        h3 = split_h3_sections(raw)
        _, fee_rows = parse_pipe_table(h3.get("Course-wise Fee Structure", ""))
        _, hostel_rows = parse_pipe_table(h3.get("Hostel Fees (Per Semester)", ""))

        data_root.joinpath("admissions", "fee-policy.json").write_text(
            json.dumps(
                envelope(
                    "admissions_fee_policy",
                    "admissions",
                    "Fee structure & policy",
                    {
                        "schema": "fee_policy_v1",
                        "source_page": "https://www.sgsits.ac.in/fee-structure",
                        "important_note": h3.get("Important Note", "").strip(),
                        "course_fees": fee_rows,
                        "fee_components": bullets(h3.get("Fee Components", "")),
                        "hostel_fees": hostel_rows,
                        "fee_concession_scholarships": h3.get("Fee Concession / Scholarships", "").strip(),
                        "mode_of_payment": bullets(h3.get("Mode of Payment", "")),
                        "contact": bullets(h3.get("Contact for Fee Queries", "")),
                        "raw_text": strip_leading_meta(raw),
                    },
                    ["fees", "hostel", "scholarship"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ contact
    cpath = clean / "contact" / "contact.txt"
    if cpath.exists():
        raw = cpath.read_text(encoding="utf-8", errors="replace")
        src = [rel(cpath)]
        h3 = split_h3_sections(raw)
        data_root.joinpath("contact", "address.json").write_text(
            json.dumps(
                envelope(
                    "contact_address",
                    "contact",
                    "Address & reach",
                    {
                        "schema": "contact_address_v1",
                        "source_page": "https://www.sgsits.ac.in/contact",
                        "address_block": h3.get("Contact Information", "").strip(),
                        "how_to_reach": bullets(h3.get("How to Reach", "")),
                        "office_hours": bullets(h3.get("Office Hours", "")),
                    },
                    ["address", "indore", "campus"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        phones = h3.get("Phone Numbers", "")
        entries = [ln.strip("- ").strip() for ln in phones.splitlines() if ln.strip().startswith("-")]
        data_root.joinpath("contact", "directory.json").write_text(
            json.dumps(
                envelope(
                    "contact_directory",
                    "contact",
                    "Contact directory",
                    {
                        "schema": "contact_directory_v1",
                        "source_page": "https://www.sgsits.ac.in/contact",
                        "phone_email_directory": entries,
                        "raw_phone_section": phones.strip(),
                    },
                    ["phone", "email", "directory"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ facilities
    fac = clean / "facilities" / "campus_facilities.txt"
    if fac.exists():
        raw = fac.read_text(encoding="utf-8", errors="replace")
        src = [rel(fac)]
        h3 = split_h3_sections(raw)

        def write_fac(rel_path: str, doc_id: str, title: str, key: str, tags: list[str]) -> None:
            block = h3.get(key, "")
            data_root.joinpath(*rel_path.split("/")).write_text(
                json.dumps(
                    envelope(
                        doc_id,
                        "facilities",
                        title,
                        {
                            "schema": "facility_section_v1",
                            "source_page": "https://www.sgsits.ac.in/facilities",
                            "section": key,
                            "content_text": block.strip(),
                            "bullet_points": bullets(block),
                        },
                        tags,
                        src,
                        utc_now_fn,
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        # Hostel section may span multiple h3 — combine Hostel + overview hints
        hostel_combined = "\n\n".join(
            h3.get(k, "")
            for k in ("Campus Overview", "Hostel Facilities")
            if h3.get(k)
        )
        data_root.joinpath("facilities", "hostel.json").write_text(
            json.dumps(
                envelope(
                    "fac_hostel",
                    "facilities",
                    "Hostel",
                    {
                        "schema": "facility_hostel_v1",
                        "source_page": "https://www.sgsits.ac.in/hostel",
                        "content_text": hostel_combined.strip(),
                        "bullet_points": bullets(hostel_combined),
                    },
                    ["hostel", "mess", "accommodation"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        write_fac("facilities/library.json", "fac_library", "Library", "Central Library", ["library", "books"])
        write_fac(
            "facilities/sports.json",
            "fac_sports",
            "Sports",
            "Sports & Recreation Facilities",
            ["sports", "ground"],
        )
        write_fac(
            "facilities/labs.json",
            "fac_labs",
            "Labs & computing",
            "Computing & IT Infrastructure",
            ["labs", "computing", "wifi"],
        )
        write_fac(
            "facilities/medical.json",
            "fac_medical",
            "Medical",
            "Medical & Health Facilities",
            ["medical", "health"],
        )

    # ------------------------------------------------------------------ placements (enrich)
    plc = clean / "placements" / "placements.txt"
    if plc.exists():
        raw = plc.read_text(encoding="utf-8", errors="replace")
        src_txt = rel(plc)
        h3 = split_h3_sections(raw)
        _, stat_rows = parse_pipe_table(h3.get("Placement Statistics (Recent Trends)", ""))
        _, branch_rows = parse_pipe_table(h3.get("Branch-wise Placement Trends", ""))

        for name, fname, merge_key in [
            ("placements_statistics", "statistics.json", "website_scrape"),
            ("placements_companies", "companies.json", "website_scrape"),
            ("placements_training_cell", "training-cell.json", "website_scrape"),
        ]:
            path = data_root / "placements" / fname
            if path.exists():
                doc = json.loads(path.read_text(encoding="utf-8"))
            else:
                doc = {"metadata": {}, "content": {}}
            doc.setdefault("content", {})[merge_key] = {
                "intro": strip_leading_meta(raw).split("###")[0].strip(),
                "sections": {k: v.strip() for k, v in h3.items()},
                "placement_statistics_table": stat_rows,
                "branch_wise_table": branch_rows,
                "top_recruiters_text": h3.get("Top Recruiters", ""),
                "source_clean_file": src_txt,
            }
            doc.setdefault("metadata", {})
            doc["metadata"]["last_updated"] = utc_now_fn()
            doc["metadata"].setdefault("source_files", []).append(src_txt)
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ research
    res = clean / "research" / "research_development.txt"
    if res.exists():
        raw = res.read_text(encoding="utf-8", errors="replace")
        src = [rel(res)]
        h3 = split_h3_sections(raw)
        data_root.joinpath("research", "projects.json").write_text(
            json.dumps(
                envelope(
                    "research_projects",
                    "research",
                    "Research projects & focus",
                    {
                        "schema": "research_projects_v1",
                        "source_page": "https://www.sgsits.ac.in/research",
                        "overview": strip_leading_meta(raw).split("###")[0].strip(),
                        "phd_programs_section": h3.get("Ph.D. Programs", ""),
                        "research_focus": {k: v for k, v in h3.items() if "Research Focus" in k or "Engineering" in k},
                        "research_infrastructure": h3.get("Research Infrastructure", ""),
                        "funded_research_projects": h3.get("Funded Research Projects", ""),
                        "consultancy_services": h3.get("Consultancy Services", ""),
                    },
                    ["research", "dst", "serb", "projects"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        data_root.joinpath("research", "publications.json").write_text(
            json.dumps(
                envelope(
                    "research_publications",
                    "research",
                    "Publications",
                    {
                        "schema": "research_publications_v1",
                        "source_page": "https://www.sgsits.ac.in/research",
                        "publications_section": h3.get("Publications", ""),
                        "bullet_points": bullets(h3.get("Publications", "")),
                    },
                    ["publications", "journals", "ieee"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        data_root.joinpath("research", "patents.json").write_text(
            json.dumps(
                envelope(
                    "research_patents",
                    "research",
                    "Patents",
                    {
                        "schema": "research_patents_v1",
                        "note": "Patents mentioned briefly in publications section of research page.",
                        "publications_section_excerpt": h3.get("Publications", ""),
                    },
                    ["patents", "ip"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ student life
    sl = clean / "student_life" / "student_life.txt"
    if sl.exists():
        raw = sl.read_text(encoding="utf-8", errors="replace")
        src = [rel(sl)]
        h3 = split_h3_sections(raw)
        data_root.joinpath("students", "activities.json").write_text(
            json.dumps(
                envelope(
                    "students_activities",
                    "students",
                    "Student activities & clubs",
                    {
                        "schema": "student_activities_v1",
                        "source_page": "https://www.sgsits.ac.in/student-life",
                        "clubs_and_societies": {k: v for k, v in h3.items() if "Club" in k or "Technical" in k or "Cultural" in k or "Social" in k},
                        "events_and_fests": "\n".join(
                            h3.get(k, "") for k in h3 if "Event" in k or "Fest" in k or "AAYAM" in k
                        ),
                        "major_events": h3.get("Major Events & Fests", ""),
                        "anti_ragging": h3.get("Anti-Ragging Measures", ""),
                        "raw_text": strip_leading_meta(raw),
                    },
                    ["clubs", "fest", "aayam"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        data_root.joinpath("students", "scholarships.json").write_text(
            json.dumps(
                envelope(
                    "students_scholarships",
                    "students",
                    "Scholarships",
                    {
                        "schema": "student_scholarships_v1",
                        "scholarships_section": h3.get("Scholarships Available", ""),
                        "bullet_points": bullets(h3.get("Scholarships Available", "")),
                    },
                    ["scholarship", "tfw"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        data_root.joinpath("students", "ncc-nss.json").write_text(
            json.dumps(
                envelope(
                    "students_ncc_nss",
                    "students",
                    "NCC & NSS",
                    {
                        "schema": "ncc_nss_v1",
                        "from_student_life": True,
                        "social_clubs_excerpt": h3.get("Social Clubs", ""),
                        "bullet_points": bullets(h3.get("Social Clubs", "")),
                    },
                    ["ncc", "nss", "community"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        data_root.joinpath("students", "skill-courses.json").write_text(
            json.dumps(
                envelope(
                    "students_skill_courses",
                    "students",
                    "Student welfare & skills",
                    {
                        "schema": "student_welfare_v1",
                        "student_welfare": h3.get("Student Welfare", ""),
                        "bullet_points": bullets(h3.get("Student Welfare", "")),
                    },
                    ["welfare", "counselling"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ accreditation → policies
    acc = clean / "accreditation" / "accreditation_rankings.txt"
    if acc.exists():
        raw = acc.read_text(encoding="utf-8", errors="replace")
        src = [rel(acc)]
        h3 = split_h3_sections(raw)
        _, nba_rows = parse_pipe_table(h3.get("NBA Accreditation", ""))
        data_root.joinpath("about", "policies.json").write_text(
            json.dumps(
                envelope(
                    "about_policies_accreditation",
                    "about",
                    "Accreditation, quality & policy links",
                    {
                        "schema": "accreditation_quality_v1",
                        "source_page": "https://www.sgsits.ac.in/accreditation",
                        "naac": h3.get("NAAC Accreditation", ""),
                        "nba": {"text": h3.get("NBA Accreditation", ""), "programs_table": nba_rows},
                        "nirf": h3.get("NIRF Rankings", ""),
                        "other_recognitions": h3.get("Other Recognitions", ""),
                        "iqac": h3.get("IQAC (Internal Quality Assurance Cell)", ""),
                        "quality_initiatives": h3.get("Quality Initiatives", ""),
                        "raw_text": strip_leading_meta(raw),
                    },
                    ["naac", "nba", "nirf", "iqac"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ links → home helpers
    lk = clean / "links" / "website_links.txt"
    if lk.exists():
        raw = lk.read_text(encoding="utf-8", errors="replace")
        src = [rel(lk)]
        h3 = split_h3_sections(raw)
        _, main_nav = parse_pipe_table(h3.get("Main Navigation Links", ""))
        _, acad_links = parse_pipe_table(h3.get("Academic & Administrative Links", ""))
        _, stu_links = parse_pipe_table(h3.get("Student & Campus Links", ""))

        data_root.joinpath("home", "notifications.json").write_text(
            json.dumps(
                envelope(
                    "home_portal_links",
                    "home",
                    "Official portal quick links",
                    {
                        "schema": "portal_links_v1",
                        "main_navigation": main_nav,
                        "academic_administrative": acad_links,
                        "note": "Use these URLs for latest notices and downloads.",
                    },
                    ["links", "portal"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        news_rows = [r for r in stu_links if "news" in str(r).lower() or "event" in str(r).lower() or "gallery" in str(r).lower()]
        data_root.joinpath("home", "news-events.json").write_text(
            json.dumps(
                envelope(
                    "home_news_events",
                    "home",
                    "News, events & gallery links",
                    {
                        "schema": "news_events_links_v1",
                        "student_campus_links": stu_links,
                        "filtered_news_events_gallery": news_rows,
                    },
                    ["news", "events", "gallery"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        data_root.joinpath("home", "gallery.json").write_text(
            json.dumps(
                envelope(
                    "home_gallery",
                    "home",
                    "Gallery",
                    {
                        "schema": "gallery_link_v1",
                        "gallery_url": "https://www.sgsits.ac.in/gallery",
                        "related_links": news_rows,
                    },
                    ["gallery", "media"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Exam-related JSONs from same link table
        data_root.joinpath("academics", "exams", "timetable.json").write_text(
            json.dumps(
                envelope(
                    "acad_exams_timetable",
                    "academics/exams",
                    "Timetable",
                    {
                        "schema": "exam_resource_v1",
                        "official_url": "https://www.sgsits.ac.in/timetable",
                        "portal_context": acad_links,
                    },
                    ["timetable", "exams"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        data_root.joinpath("academics", "exams", "results.json").write_text(
            json.dumps(
                envelope(
                    "acad_exams_results",
                    "academics/exams",
                    "Results",
                    {
                        "schema": "exam_resource_v1",
                        "official_url": "https://www.sgsits.ac.in/results",
                        "portal_context": acad_links,
                    },
                    ["results", "exams"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        data_root.joinpath("academics", "exams", "notices.json").write_text(
            json.dumps(
                envelope(
                    "acad_exams_notices",
                    "academics/exams",
                    "Notices",
                    {
                        "schema": "exam_resource_v1",
                        "official_url": "https://www.sgsits.ac.in/notices",
                        "portal_context": acad_links,
                    },
                    ["notices", "exams"],
                    src,
                    utc_now_fn,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print("Clean corpus ingest complete (data__/data/clean).")
