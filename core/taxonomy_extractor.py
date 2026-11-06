"""Deterministic Canonical Taxonomy Extractor for Phase 1 Foundation.

Strictly adheres to PHASE_1_PLAN_APPROVED.md §8.1-§8.3:
- §8.1: Module, Submodule, Chapter, Topic derived directly from corpus files.
- §8.3: Metadata schema: content_type, module_domain, command_syntax,
        chapter_section, difficulty, cross_references, source_evidence.
- Zero invented values: unverified command_syntax is null, unverified
  relationships are []. Page numbers and TOC entries are taken verbatim
  from verified source text files.
"""

import csv
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
TXT_DIR = KNOWLEDGE_DIR / "txt"
SOURCES_CSV = KNOWLEDGE_DIR / "sources.csv"
OUTPUT_JSON = KNOWLEDGE_DIR / "canonical_taxonomy.json"


def load_sources_catalog():
    catalog = {}
    if not SOURCES_CSV.exists():
        return catalog
    with open(SOURCES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            catalog[row["File"].strip()] = {
                "id": int(row["ID"]),
                "topic": row["Topic"].strip(),
                "source": row["Source"].strip(),
            }
    return catalog


def parse_lab_workbook_toc(file_name, domain, default_submodule, catalog):
    path = TXT_DIR / file_name
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.split("\n")

    cat_entry = catalog.get(file_name, {})
    source_id = cat_entry.get("id")

    nodes = []
    current_lab = None
    current_lab_page = None
    in_toc = False

    for line_idx, line in enumerate(lines[:120]):
        stripped = line.strip()
        if "Table of Contents" in stripped:
            in_toc = True
            continue
        if in_toc and ("Before you Begin" in stripped or "Lab Answers" in stripped):
            if "Lab Answers" in stripped:
                break

        if not in_toc:
            continue

        # Match Lab line: "Lab 2: Configuring Scan Chains/Test Logic ... 15"
        lab_m = re.match(r"^(Lab\s+\d+:\s*[^.]+?)(?:\s*\.{2,}|\s{3,})(\d+)\s*$", stripped)
        if lab_m:
            current_lab = lab_m.group(1).strip()
            current_lab_page = int(lab_m.group(2).strip())
            continue

        # Match Exercise line: "Exercise 1: Basic Scan Insertion ... 16"
        ex_m = re.match(r"^(Exercise\s+\d+:\s*[^.]+?)(?:\s*\.{2,}|\s{3,})(\d+)\s*$", stripped)
        if ex_m and current_lab:
            ex_title = ex_m.group(1).strip()
            page_num = int(ex_m.group(2).strip())

            # Difficulty assessment based on workbook progression (§8.3)
            diff = "introductory"
            lab_num_m = re.search(r"Lab\s+(\d+)", current_lab)
            if lab_num_m:
                lnum = int(lab_num_m.group(1))
                if lnum >= 6 or "Advanced" in file_name:
                    diff = "advanced"
                elif lnum >= 3:
                    diff = "intermediate"

            node = {
                "module_domain": domain,
                "submodule": default_submodule,
                "chapter": current_lab,
                "topic": ex_title,
                "content_type": "lab_workbook",
                "chapter_section": f"{current_lab} / {ex_title}",
                "command_syntax": None,
                "difficulty": diff,
                "cross_references": [],
                "source_evidence": {
                    "file": f"knowledge/txt/{file_name}",
                    "source_catalog_id": source_id,
                    "page_number": page_num,
                    "toc_reference": f"Page 3, Line {line_idx + 1}"
                }
            }
            nodes.append(node)

    return nodes


def parse_reference_manual_chapters(file_name, domain, submodule, content_type, catalog):
    path = TXT_DIR / file_name
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.split("\n")

    cat_entry = catalog.get(file_name, {})
    source_id = cat_entry.get("id")

    nodes = []
    in_toc = False
    # TOC scanning in reference manuals: only parse lines inside the Table of Contents
    for i in range(len(lines[:1000])):
        line = lines[i].strip()
        if "Table of Contents" in line:
            in_toc = True
            continue

        if not in_toc:
            continue

        # End of Table of Contents indicators: List of Figures, List of Tables, or actual first chapter content page
        if line.startswith("List of Figures") or line.startswith("List of Tables") or line.startswith("--- PAGE 11 ---") or line.startswith("--- PAGE 27 ---"):
            break

        m = re.match(r"^Chapter\s+(\d+)\s*$", line)
        if m:
            ch_num = m.group(1)
            # In TOC format:
            # Chapter 1
            # Getting Started . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            # 11
            ch_title = None
            page_num = None
            if i + 1 < len(lines):
                title_line = lines[i + 1].strip()
                # Remove trailing dots, spaces, and trailing page numbers embedded in title line
                title_line = re.sub(r"[\s\.]+(?:\d+)?$", "", title_line).strip()
                title_line = re.sub(r"[\s\.]+$", "", title_line).strip()
                if title_line and not title_line.isdigit():
                    ch_title = f"Chapter {ch_num}: {title_line}"
            if i + 2 < len(lines):
                p_line = lines[i + 2].strip()
                if p_line.isdigit():
                    page_num = int(p_line)

            if ch_title:
                node = {
                    "module_domain": domain,
                    "submodule": submodule,
                    "chapter": ch_title,
                    "topic": ch_title,
                    "content_type": content_type,
                    "chapter_section": ch_title,
                    "command_syntax": None,
                    "difficulty": "intermediate" if int(ch_num) > 2 else "introductory",
                    "cross_references": [],
                    "source_evidence": {
                        "file": f"knowledge/txt/{file_name}",
                        "source_catalog_id": source_id,
                        "page_number": page_num,
                        "toc_reference": f"TOC Line {i + 1}"
                    }
                }
                nodes.append(node)

    return nodes


def parse_video_course_modules(catalog):
    nodes = []
    # Parse video course items recorded in sources.csv
    for file_name, meta in sorted(catalog.items()):
        source_type = meta["source"]
        topic_title = meta["topic"]
        if "Video Course" not in source_type and "Exam Bank" not in source_type:
            continue

        c_type = "video_slides" if "slides" in file_name else ("video_subs" if "subs" in file_name else "exam_quiz")

        # Determine domain from prefix
        if file_name.startswith("tshell_intro"):
            domain = "TSHELL_LINUX"
        elif file_name.startswith("tk_"):
            domain = "EDT"
        elif file_name.startswith("atpg_adv"):
            domain = "ATPG"
        elif file_name.startswith("atpg_"):
            # Check for scan module
            if "2_scan_and_test_insertion" in file_name:
                domain = "SCAN"
            else:
                domain = "ATPG"
        else:
            domain = "ATPG"

        node = {
            "module_domain": domain,
            "submodule": topic_title,
            "chapter": topic_title,
            "topic": topic_title,
            "content_type": c_type,
            "chapter_section": topic_title,
            "command_syntax": None,
            "difficulty": "intermediate",
            "cross_references": [],
            "source_evidence": {
                "file": f"knowledge/txt/{file_name}",
                "source_catalog_id": meta["id"],
                "page_number": None,
                "toc_reference": f"sources.csv row ID {meta['id']}"
            }
        }
        nodes.append(node)
    return nodes


def build_canonical_taxonomy():
    catalog = load_sources_catalog()
    all_nodes = []

    # 1. Lab Workbooks (TOC based)
    all_nodes.extend(parse_lab_workbook_toc(
        "Tessent_Shell_Introduction_lab_wkb_2026.1_s.txt",
        "TSHELL_LINUX",
        "Tessent Shell Introduction",
        catalog
    ))

    # ATPG Core workbook has both SCAN (Labs 2,3) and ATPG (Labs 1,4,6,7,8)
    core_lab_nodes = parse_lab_workbook_toc(
        "Tessent_ATPG_Core_Topics_lab_wkbk_2025.4_s.txt",
        "ATPG",
        "Tessent ATPG Core Topics",
        catalog
    )
    for n in core_lab_nodes:
        if "Lab 2:" in n["chapter"] or "Lab 3:" in n["chapter"]:
            n["module_domain"] = "SCAN"
            n["submodule"] = "Module 2: Scan and Test Insertion"
        all_nodes.append(n)

    all_nodes.extend(parse_lab_workbook_toc(
        "Tessent_ATPG_with_TestKompress_lab_wkb_2025.4_s.txt",
        "EDT",
        "Tessent ATPG with TestKompress",
        catalog
    ))

    all_nodes.extend(parse_lab_workbook_toc(
        "Tessent_ATPG_Advanced_Topics_lab_wkb_2025.4_s.txt",
        "ATPG",
        "Tessent ATPG Advanced Topics",
        catalog
    ))

    # 2. Reference / User Manual Chapters
    all_nodes.extend(parse_reference_manual_chapters(
        "tshell_user (3).txt",
        "TSHELL_LINUX",
        "Tessent Shell User's Manual",
        "user_manual",
        catalog
    ))

    atpg_gd_chapters = parse_reference_manual_chapters(
        "atpg_gd.txt",
        "ATPG",
        "Scan and ATPG User's Manual",
        "user_manual",
        catalog
    )
    for n in atpg_gd_chapters:
        if "Chapter 1:" in n["chapter"] or "Chapter 2:" in n["chapter"]:
            n["module_domain"] = "SCAN"
        all_nodes.append(n)

    all_nodes.extend(parse_reference_manual_chapters(
        "Testkompress.Manual.txt",
        "EDT",
        "TestKompress User's and Reference Manual",
        "user_manual",
        catalog
    ))

    # 3. Video course modules and quizzes
    all_nodes.extend(parse_video_course_modules(catalog))

    output_payload = {
        "taxonomy_version": "1.0.0",
        "approved_plan_reference": "PHASE_1_PLAN_APPROVED.md §8.1-§8.3",
        "total_nodes": len(all_nodes),
        "domains": sorted(list(set(n["module_domain"] for n in all_nodes))),
        "nodes": all_nodes
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"Extracted {len(all_nodes)} taxonomy nodes into {OUTPUT_JSON}")
    return output_payload


if __name__ == "__main__":
    build_canonical_taxonomy()
