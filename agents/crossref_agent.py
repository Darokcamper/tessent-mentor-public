"""Cross-Reference Agent - explicit lab<->manual cross-referencing engine.

Builds a deterministic bridge between the lab exercises and the Tessent manuals:

1. extract_lab_commands(lab_display) - ordered, deduped list of Tessent commands
   (verb_noun pattern) from the lab slides OCR text, the lab workbook and any
   TCL files inside the lab folder.
2. crossref_lab(lab_display) - for every extracted command, attach the FULL
   manual definition block (usage/options, with page citations) via
   rag_builder.retrieve_command_definition, plus where-else-used info.
3. where_used(command) - reverse direction: which lab exercises use a command.

Mostly LLM-free: deterministic text matching over local files, so there is no
hallucination surface and no API cost for the core table.
"""
import re
from pathlib import Path

from core.rag_builder import retrieve_command_definition

KNOWLEDGE_TXT_DIR = Path("knowledge") / "txt"
LAB_DIR = Path("knowledge") / "Tessent Atpg Core Topics" / "10.Labs"
LAB_WORKBOOK = KNOWLEDGE_TXT_DIR / "Tessent_ATPG_Core_Topics_lab_wkbk_2025.4_s.txt"
LAB_SLIDES_FILE = KNOWLEDGE_TXT_DIR / "atpg_video_slides__10_labs.txt"

# Same verb_noun command pattern as rag_builder.retrieve() (~line 366) so
# extraction and retrieval agree on what a "Tessent command" is.
CMD_RE = re.compile(
    r"\b(?:set|add|create|remove|delete|write|read|report|get|put|check|run|"
    r"save|do|exit|source|define)_[a-z0-9_]+\b"
)

# Flow-critical commands the POC asks about; also the allowlist that filters
# OCR fused-word artifacts (e.g. read_verilogdesign, add_scan_groupsgrpl).
KNOWN_COMMANDS = (
    "set_current_design", "add_scan_mode", "analyze_scan_chains",
    "insert_test_logic", "check_design_rules", "build_model",
    "set_system_mode", "set_context", "read_verilog",
    "report_scan_chains", "report_scan_cells", "report_scan_elements",
    "report_test_logic", "create_patterns", "write_patterns",
    "verify_scan_chains", "run_drc", "read_cell_library",
    "write_design", "add_scan_chains", "add_scan_groups",
    "add_black_box", "add_black_boxes", "add_clocks",
    "set_scan_insertion_options", "set_insertion_options",
    "set_scan_signals", "set_edt_options", "set_fault_type",
    "set_atpg_limits", "set_output_masks", "set_transition_hold",
    "add_faults", "read_patterns", "report_statistics",
    "delete_design", "read_cell_lib", "add_cell_models",
    "set_scan", "set_system", "set_test_logic",
    "write_atpg_setup", "write_tcl_library", "report_memory_bists",
)
EXTRA_COMMANDS = KNOWN_COMMANDS  # backwards-compatible alias


def _read_text(path, cap=None):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    if cap and len(text) > cap:
        half = cap // 2
        return text[:half] + "\n[... middle omitted ...]\n" + text[-half:]
    return text


def _lab_keys(lab_display):
    head = lab_display.split(" / ")[0].lower().replace(" ", "")
    sub = lab_display.split(" / ")[1].lower().replace(" ", "") if " / " in lab_display else ""
    return head, sub


def _split_pages(text):
    """Splits ingest text into (label, body) tuples at [SOURCE: ...] markers."""
    pages = []
    marker = "[SOURCE:"
    idx = text.find(marker)
    while idx != -1:
        end = text.find("]", idx)
        nxt = text.find(marker, idx + 1)
        if end == -1:
            end = idx + 10
        label = text[idx + 1:end].strip()
        body = text[end + 1:nxt if nxt != -1 else len(text)]
        pages.append((label, body.strip()))
        idx = nxt
    return pages


def _lab_scope_text(lab_display):
    """Lab-slides OCR pages that belong to this lab exercise."""
    head, sub = _lab_keys(lab_display)
    pages = _split_pages(_read_text(LAB_SLIDES_FILE, cap=400000))
    picked = []
    for label, body in pages:
        blob = (label + "\n" + body).lower().replace(" ", "")
        if (head and head in blob) or (sub and sub in blob):
            picked.append(body)
    return "\n\n".join(picked)


def _lab_folder_text(lab_display):
    """Concatenated TCL/text file contents inside the lab folder on disk."""
    target = LAB_DIR / lab_display.replace(" / ", "/")
    if not target.exists():
        return ""
    if target.is_file():
        return _read_text(target, cap=60000)
    parts = []
    for p in sorted(target.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".tcl", ".txt", ".do", ".f", ".sdc"):
            parts.append(_read_text(p, cap=30000))
    return "\n".join(parts)


def extract_lab_commands(lab_display):
    """Ordered, deduped list of Tessent commands used in ONE lab exercise.

    Regex hits are filtered through KNOWN_COMMANDS to drop OCR fused-word
    artifacts (read_verilogdesign etc.); a substring pass then catches flow
    commands whose OCR text got mangled beyond the exact form.
    """
    scoped = _lab_scope_text(lab_display)
    folder_text = _lab_folder_text(lab_display)
    blob = (scoped + "\n" + folder_text).lower()

    known = set(KNOWN_COMMANDS)
    found = []
    seen = set()
    for m in CMD_RE.finditer(blob):
        cmd = m.group(0).lower()
        if cmd in known and cmd not in seen:
            seen.add(cmd)
            found.append(cmd)
    for cmd in KNOWN_COMMANDS:
        if cmd not in seen and cmd in blob:
            seen.add(cmd)
            found.append(cmd)
    # Drop substring shadows: set_scan (shadowed by set_scan_insertion_options),
    # set_system, read_cell_lib, add_black_box, etc.
    found = [c for c in found if not any(o != c and o.startswith(c) for o in found)]
    return found


# Page-body cache so reverse lookups do not re-read the slides file per call.
_page_body_cache = {}


def _warm_page_cache():
    if not _page_body_cache and LAB_SLIDES_FILE.exists():
        for label, body in _split_pages(_read_text(LAB_SLIDES_FILE, cap=400000)):
            _page_body_cache[label] = body


def where_used(command):
    """Reverse lookup: which lab exercises use a command (from lab slides OCR)."""
    _warm_page_cache()
    cmd = command.lower().strip()
    hits = []
    for label, body in _page_body_cache.items():
        blob = (label + "\n" + body).lower()
        if re.search(r"\b" + re.escape(cmd) + r"\b", blob):
            # label: SOURCE: 10_labs/Lab4/EX2/Screenshot (123).png
            #        SOURCE: 10_labs/Lab2/Screenshot 2026-08-20 095703.png
            segs = [p for p in label.replace("SOURCE:", "").strip().split("/") if p]
            if segs and segs[0].lower() == "10_labs":
                segs = segs[1:]
            if len(segs) >= 3:
                entry = segs[0] + " / " + segs[1]
            elif len(segs) == 2:
                entry = segs[0] + " / (lab-level)"
            else:
                continue
            if entry not in hits:
                hits.append(entry)
    return hits


def crossref_lab(lab_display, max_commands=60):
    """Full cross-reference rows for ONE lab exercise.

    Returns a list of dicts: command, manual_defs -> [{source, page, text}],
    used_in -> [lab/ex strings].
    """
    _warm_page_cache()
    commands = extract_lab_commands(lab_display)[:max_commands]
    rows = []
    for cmd in commands:
        rows.append({
            "command": cmd,
            "manual_defs": retrieve_command_definition(cmd),
            "used_in": where_used(cmd),
        })
    return rows


def format_crossref_table(rows):
    """Markdown table + per-command manual definition blocks."""
    if not rows:
        return "No Tessent commands were found in this lab exercise's material."
    lines = ["## Lab <-> Manual Cross-Reference", ""]
    lines.append("| # | Command | Manual Def? | Also Used In |")
    lines.append("|---|---------|-------------|--------------|")
    for i, r in enumerate(rows, 1):
        n = len(r["manual_defs"])
        def_flag = (str(n) + " block(s)") if n else "not found in manuals"
        used = ", ".join(r["used_in"][:4]) if r["used_in"] else "-"
        lines.append("| %d | `%s` | %s | %s |" % (i, r["command"], def_flag, used))

    for r in rows:
        if not r["manual_defs"]:
            continue
        lines.append("")
        lines.append("### `" + r["command"] + "`")
        for d in r["manual_defs"][:2]:
            cite = "%s, Page %s" % (d.get("source", "?"), d.get("page", "?"))
            lines.append("")
            lines.append("> [Source: %s]" % cite)
            lines.append(">")
            body = str(d.get("text", "")).strip()
            if len(body) > 1200:
                body = body[:1200] + " ..."
            for ln in body.split("\n"):
                lines.append("> " + ln)
    return "\n".join(lines)
