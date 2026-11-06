"""Build lab/exercise taxonomy nodes and a versioned taxonomy artifact (WP-5).

The historical extractor (``core/taxonomy_extractor.py``) requires a page number at
the end of every TOC line. The OCR'd lab workbooks end TOC lines with dot leaders and
no page number, so ``parse_lab_workbook_toc()`` returns zero nodes for all four
workbooks. As a result ``knowledge/canonical_taxonomy.json`` contains no lab nodes at
all and Theory/Lab export can never produce anything.

This module builds the missing lab nodes *additively*:

* TOC parsing tolerates a missing page number.
* Each node's page and content excerpt are located in the workbook body, so the
  citation is real text evidence rather than a TOC guess.
* ``build_taxonomy_v2()`` copies the existing 78 nodes unchanged, adds an explicit
  ``classification`` field to them, appends the new lab nodes, and writes
  ``knowledge/canonical_taxonomy_v2.json``. The original artifact is never modified.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
TXT_DIR = KNOWLEDGE_DIR / "txt"
BASE_TAXONOMY = KNOWLEDGE_DIR / "canonical_taxonomy.json"
V2_TAXONOMY = KNOWLEDGE_DIR / "canonical_taxonomy_v2.json"
SOURCES_CSV = KNOWLEDGE_DIR / "sources.csv"

TAXONOMY_VERSION = "2.0.0"

# (workbook text file, canonical domain, course submodule name)
LAB_WORKBOOKS = (
    ("Tessent_ATPG_Core_Topics_lab_wkbk_2025.4_s.txt", "ATPG", "ATPG Core Topics Lab Workbook"),
    ("Tessent_ATPG_with_TestKompress_lab_wkb_2025.4_s.txt", "EDT", "ATPG with TestKompress Lab Workbook"),
    ("Tessent_Shell_Introduction_lab_wkb_2026.1_s.txt", "TSHELL_LINUX", "Tessent Shell Introduction Lab Workbook"),
    ("Tessent_ATPG_Advanced_Topics_lab_wkb_2025.4_s.txt", "ATPG", "ATPG Advanced Topics Lab Workbook"),
)

# ATPG Core Labs 2 and 3 are scan-insertion labs; every other lab is ATPG.
# Evidence: knowledge/crossref_audit_multimodule.json and
# core/taxonomy_extractor.py's existing split.
_SCAN_LAB_NUMBERS = {"2", "3"}

_TOC_LAB_RE = re.compile(r"^(Lab\s+\d+)\s*[:\-]\s*(.+?)(?:\s*\.{2,}\s*(\d+)?)?\s*$", re.IGNORECASE)
_TOC_EX_RE = re.compile(r"^(Exercise\s+\d+)\s*[:\-]\s*(.+?)(?:\s*\.{2,}\s*(\d+)?)?\s*$", re.IGNORECASE)
_PAGE_RE = re.compile(r"--- PAGE (\d+) ---")
_BODY_LAB_RE = re.compile(r"^(Lab\s+\d+)\s*:?\s*(.*)$", re.IGNORECASE)
_BODY_EX_RE = re.compile(r"^(Exercise\s+\d+)\s*:?\s*(.*)$", re.IGNORECASE)

_EXCERPT_LIMIT = 6000


def _load_catalog() -> Dict[str, int]:
    """sources.csv file name -> catalog id (used for source_evidence)."""
    catalog: Dict[str, int] = {}
    if not SOURCES_CSV.exists():
        return catalog
    import csv

    with open(SOURCES_CSV, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("File") or "").strip()
            try:
                catalog[name] = int((row.get("ID") or "").strip())
            except (TypeError, ValueError):
                continue
    return catalog


def _cleaned_title(raw: str) -> str:
    return re.sub(r"\s*\.{2,}\s*\d*\s*$", "", raw).strip().rstrip(".").strip()


def parse_toc(lines: List[str], limit: int = 200) -> List[Dict[str, Any]]:
    """Lab/exercise entries from the workbook TOC (page number optional)."""
    entries: List[Dict[str, Any]] = []
    in_toc = False
    current_lab: Optional[str] = None
    for line in lines[:limit]:
        stripped = line.strip()
        if "table of contents" in stripped.lower():
            in_toc = True
            continue
        if not in_toc:
            continue
        if stripped.lower().startswith(("before you begin", "lab answers")):
            if stripped.lower().startswith("lab answers"):
                break
            continue
        lab_match = _TOC_LAB_RE.match(stripped)
        if lab_match and "..." in stripped:
            current_lab = f"{lab_match.group(1).strip()}: {_cleaned_title(lab_match.group(2))}"
            entries.append(
                {"kind": "lab", "title": current_lab, "toc_page": lab_match.group(3)}
            )
            continue
        ex_match = _TOC_EX_RE.match(stripped)
        if ex_match and current_lab and "..." in stripped:
            entries.append(
                {
                    "kind": "exercise",
                    "title": f"{ex_match.group(1).strip()}: {_cleaned_title(ex_match.group(2))}",
                    "lab": current_lab,
                    "toc_page": ex_match.group(3),
                }
            )
    return entries


def _page_at(page_spans: List[tuple], offset: int) -> Optional[int]:
    page = None
    for start, number in page_spans:
        if start <= offset:
            page = number
        else:
            break
    return page


def _locate_body_entry(text: str, title: str) -> Optional[int]:
    """Character offset of *title* in the workbook body, tolerating OCR spacing."""
    wanted = re.sub(r"\s+", " ", title).strip().lower()
    body_start = text.lower().find("table of contents")
    haystack_start = body_start + 1 if body_start != -1 else 0
    for match in re.finditer(r"(?m)^(.+)$", text[haystack_start:]):
        line = re.sub(r"\s+", " ", match.group(1)).strip().lower()
        if line.startswith(wanted) or (wanted.startswith(line.rstrip(":")) and len(line) > 6):
            return haystack_start + match.start()
    short = " ".join(wanted.split()[:3])
    if len(short) >= 8:
        for match in re.finditer(r"(?m)^(.+)$", text[haystack_start:]):
            line = re.sub(r"\s+", " ", match.group(1)).strip().lower()
            if line.startswith(short):
                return haystack_start + match.start()
    return None


def _next_heading_offset(text: str, start: int) -> int:
    tail = text[start + 1:]
    for match in re.finditer(r"(?m)^(?:Lab\s+\d+|Exercise\s+\d+|Assessment\b|Knowledge Check\b)", tail):
        return start + 1 + match.start()
    return min(len(text), start + _EXCERPT_LIMIT)


def build_lab_nodes(txt_dir: Path = TXT_DIR) -> List[Dict[str, Any]]:
    """Lab and exercise nodes for all four workbooks, with page and content excerpt."""
    catalog = _load_catalog()
    nodes: List[Dict[str, Any]] = []

    for file_name, domain, submodule in LAB_WORKBOOKS:
        path = txt_dir / file_name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        page_spans = [(m.end(), int(m.group(1))) for m in _PAGE_RE.finditer(text)]
        entries = parse_toc(text.split("\n"))

        for entry in entries:
            if entry["kind"] != "lab":
                continue
            chapter = entry["title"]
            offset = _locate_body_entry(text, chapter)
            if offset is None:
                continue
            end = _next_heading_offset(text, offset)
            page = _page_at(page_spans, offset)

            zone_domain = domain
            lab_number = re.match(r"Lab\s+(\d+)", chapter, re.IGNORECASE)
            if domain == "ATPG" and lab_number and lab_number.group(1) in _SCAN_LAB_NUMBERS:
                zone_domain = "SCAN"

            nodes.append(
                _lab_node(file_name, catalog, zone_domain, submodule, chapter, chapter,
                          "workbook TOC entry under 'Table of Contents'", page, text[offset:end])
            )

            for sub in entries:
                if sub["kind"] != "exercise" or sub.get("lab") != chapter:
                    continue
                sub_offset = _locate_body_entry(text, sub["title"])
                if sub_offset is None:
                    continue
                nodes.append(
                    _lab_node(file_name, catalog, zone_domain, submodule, chapter, sub["title"],
                              "workbook TOC exercise entry", _page_at(page_spans, sub_offset),
                              text[sub_offset:_next_heading_offset(text, sub_offset)])
                )
    return nodes


def _lab_node(file_name, catalog, domain, submodule, chapter, topic, evidence, page, body) -> Dict[str, Any]:
    return {
        "module_domain": domain,
        "submodule": submodule,
        "chapter": chapter,
        "topic": topic,
        "content_type": "lab_workbook",
        "classification": "Lab",
        "export_classification": "Lab",
        "classification_evidence": evidence,
        "chapter_section": topic,
        "command_syntax": None,
        "difficulty": None,
        "difficulty_unavailable": True,
        "cross_references": [],
        "source_evidence": {
            "file": f"knowledge/txt/{file_name}",
            "source_catalog_id": catalog.get(file_name),
            "page_number": page,
            "toc_reference": "Table of Contents",
        },
        "content_excerpt": re.sub(r"[ \t]+", " ", body).strip()[:_EXCERPT_LIMIT],
        "content_boundary": "workbook heading to next lab/exercise heading",
    }


# Content types that are course/theory material. Exam material is neither theory nor
# a lab, so it deliberately stays unclassified rather than being forced into a bucket.
_THEORY_CONTENT_TYPES = ("user_manual", "video_slides", "video_subs")

CLASSIFICATION_POLICY = (
    "Explicit metadata only (Policy A). A node is 'Theory' when its content_type is one "
    "of user_manual / video_slides / video_subs; a node is 'Lab' when it was parsed from a "
    "lab workbook TOC with body evidence. Exam nodes and any other content type stay "
    "unclassified instead of being inferred."
)


# Course (module) each corpus file belongs to. Derived from the file name only, so the
# mapping is evidence-bearing and reproducible rather than a guess about content.
_COURSE_BY_FILE_PREFIX = (
    ("tshell_ref", "Tessent Shell"),
    ("tshell_user", "Tessent Shell"),
    ("Tessent_Shell_Introduction", "Tessent Shell Introduction"),
    ("atpg_gd", "ATPG Core Topics"),
    ("Tessent_ATPG_Core_Topics", "ATPG Core Topics"),
    ("atpg_video_", "ATPG Core Topics"),
    ("Testkompress.Manual", "ATPG with TestKompress"),
    ("Tessent_ATPG_with_TestKompress", "ATPG with TestKompress"),
    ("tk_video_", "ATPG with TestKompress"),
    ("Tessent_ATPG_Advanced_Topics", "ATPG Advanced Topics"),
    ("atpg_adv_video_", "ATPG Advanced Topics"),
    ("tshell_intro_video_", "Tessent Shell Introduction"),
    ("tessent_lib_user", "Library User Manual"),
)


def course_for_file(file_path: str) -> str:
    """Course (module) that owns a corpus file, derived from its basename."""
    name = str(file_path or "").replace("\\", "/").rsplit("/", 1)[-1]
    for prefix, course in _COURSE_BY_FILE_PREFIX:
        if name.startswith(prefix):
            return course
    return "Other corpus material"


def build_taxonomy_v2(
    base_path: Path = BASE_TAXONOMY,
    output_path: Path = V2_TAXONOMY,
    txt_dir: Path = TXT_DIR,
) -> Dict[str, Any]:
    """Write the additive v2 taxonomy: existing nodes unchanged + lab nodes."""
    base = json.loads(base_path.read_text(encoding="utf-8"))
    existing = [dict(node) for node in base.get("nodes", [])]
    theory_classified = 0
    for node in existing:
        if node.get("content_type") in _THEORY_CONTENT_TYPES:
            node["classification"] = "Theory"
            node["export_classification"] = "Theory"
            node["classification_evidence"] = (
                f"content_type={node.get('content_type')} (policy A, explicit metadata)"
            )
            theory_classified += 1
        node["course"] = course_for_file(node.get("source_evidence", {}).get("file", ""))
        node["course_evidence"] = "source filename prefix"

    lab_nodes = build_lab_nodes(txt_dir)
    for node in lab_nodes:
        node["course"] = course_for_file(node["source_evidence"]["file"])
        node["course_evidence"] = "source filename prefix"
    payload = {
        "taxonomy_version": TAXONOMY_VERSION,
        "supersedes": "knowledge/canonical_taxonomy.json (preserved byte-for-byte)",
        "approved_plan_reference": base.get("approved_plan_reference"),
        "classification_policy": CLASSIFICATION_POLICY,
        "total_nodes": len(existing) + len(lab_nodes),
        "existing_nodes_preserved": len(existing),
        "existing_nodes_theory_classified": theory_classified,
        "lab_nodes_added": len(lab_nodes),
        "domains": sorted({n["module_domain"] for n in existing + lab_nodes if n.get("module_domain")}),
        "nodes": existing + lab_nodes,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def validate_taxonomy_v2(
    v2: Dict[str, Any],
    base_path: Path = BASE_TAXONOMY,
    minimum_lab_nodes: int = 20,
) -> List[str]:
    """Problems that would make the v2 artifact unsafe to use; empty means valid."""
    problems: List[str] = []
    base = json.loads(base_path.read_text(encoding="utf-8"))
    by_key = {
        (n.get("module_domain"), n.get("submodule"), n.get("chapter"), n.get("topic")): n
        for n in v2.get("nodes", [])
    }

    for node in base.get("nodes", []):
        key = (node.get("module_domain"), node.get("submodule"), node.get("chapter"), node.get("topic"))
        candidate = by_key.get(key)
        if candidate is None:
            problems.append(f"existing node missing from v2: {key}")
        elif candidate.get("source_evidence") != node.get("source_evidence"):
            problems.append(f"source evidence changed for existing node: {key}")

    lab_nodes = [n for n in v2.get("nodes", []) if n.get("classification") == "Lab"]
    if len(lab_nodes) < minimum_lab_nodes:
        problems.append(f"only {len(lab_nodes)} lab nodes (expected >= {minimum_lab_nodes})")
    for node in lab_nodes:
        evidence = node.get("source_evidence") or {}
        if not evidence.get("file") or not evidence.get("page_number"):
            problems.append(f"lab node missing page evidence: {node.get('topic')}")
        if not (node.get("content_excerpt") or "").strip():
            problems.append(f"lab node has no content excerpt: {node.get('topic')}")

    theory = [n for n in v2.get("nodes", []) if n.get("classification") == "Theory"]
    if not theory:
        problems.append("no Theory-classified nodes")
    return problems


def main() -> int:
    payload = build_taxonomy_v2()
    problems = validate_taxonomy_v2(payload)
    print(f"existing nodes preserved : {payload['existing_nodes_preserved']}")
    print(f"theory classified        : {payload['existing_nodes_theory_classified']}")
    print(f"lab nodes added          : {payload['lab_nodes_added']}")
    print(f"total nodes              : {payload['total_nodes']}")
    if problems:
        print("validation problems:")
        for problem in problems[:15]:
            print("  -", problem)
        return 1
    print("validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
