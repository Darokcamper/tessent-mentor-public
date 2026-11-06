"""
core/command_indexer.py - Canonical Command Index Generation and Search Engine

Generates and manages the canonical command index (knowledge/command_index.json)
mapping Siemens Tessent DFT commands to verified syntax, module domains, flow steps,
manual page citations, and workbook lab occurrences.

Plan Reference: PHASE_1_PLAN_APPROVED.md §5.D, §8.2, §8.3
Stage Reference: PHASE_2_PREFLIGHT_REPORT.md Stage 2.2
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
COMMAND_INDEX_FILE = KNOWLEDGE_DIR / "command_index.json"

# WP-2: additive, corpus-derived superset built by core/command_index_expander.py.
# When present it is used automatically; the curated 34-command artifact above stays
# byte-for-byte intact and remains the fallback on deployments that have not built
# the expanded index yet.
FULL_COMMAND_INDEX_FILE = KNOWLEDGE_DIR / "command_index_full.json"
COMMAND_DEFS_FILE = KNOWLEDGE_DIR / "command_definitions_all_44.json"
MULTIMODULE_AUDIT_FILE = KNOWLEDGE_DIR / "crossref_audit_multimodule.json"
FULL_AUDIT_FILE = PROJECT_ROOT / "crossref_audit_full.json"
CANONICAL_TAXONOMY_FILE = KNOWLEDGE_DIR / "canonical_taxonomy.json"

# Known Aliases mapped to Canonical Commands with Provenance
KNOWN_ALIASES: Dict[str, Dict[str, Any]] = {
    "add_black_box": {
        "alias": "add_black_box",
        "canonical_command": "add_black_boxes",
        "relationship": "singular_inflection",
        "source_evidence": {
            "file": "knowledge/txt/tshell_ref (1).txt",
            "page": 127,
            "section": "Command Reference",
            "notes": "Corpus authoritative manual entry is add_black_boxes; add_black_box is singular inflection in lab transcripts.",
            "verified": True,
        },
    },
    "read_cell_lib": {
        "alias": "read_cell_lib",
        "canonical_command": "read_cell_library",
        "relationship": "command_abbreviation",
        "source_evidence": {
            "file": "knowledge/txt/tshell_ref (1).txt",
            "page": 1650,
            "section": "Command Reference",
            "notes": "Abbreviated form of read_cell_library in ATPG lab transcripts and course slides.",
            "verified": True,
        },
    },
    "set_system": {
        "alias": "set_system",
        "canonical_command": "set_system_mode",
        "relationship": "command_abbreviation",
        "source_evidence": {
            "file": "knowledge/txt/tshell_ref (1).txt",
            "page": 3052,
            "section": "Command Reference",
            "notes": "Abbreviated command prefix for set_system_mode (e.g., 'set_system mode setup') in lab transcripts.",
            "verified": True,
        },
    },
    "set_scan": {
        "alias": "set_scan",
        "canonical_command": "set_scan_insertion_options",
        "relationship": "command_abbreviation",
        "source_evidence": {
            "file": "knowledge/txt/tshell_ref (1).txt",
            "page": 3000,
            "section": "Command Reference",
            "notes": "Abbreviated command prefix for set_scan_insertion_options in lab transcripts.",
            "verified": True,
        },
    },
}

# Unverified Candidate Commands (legacy candidates in KNOWN_COMMANDS with 0 corpus definitions)
UNVERIFIED_CANDIDATES: Dict[str, Dict[str, Any]] = {
    "build_model": {
        "command": "build_model",
        "status": "UNVERIFIED_IN_CORPUS",
        "definitions_found": 0,
        "source_evidence": None,
        "note": "Legacy candidate in KNOWN_COMMANDS allowlist with 0 definitions in Siemens manuals and 0 occurrences in audited workbooks.",
    },
    "report_memory_bists": {
        "command": "report_memory_bists",
        "status": "UNVERIFIED_IN_CORPUS",
        "definitions_found": 0,
        "source_evidence": None,
        "note": "Legacy candidate in KNOWN_COMMANDS allowlist with 0 definitions in Siemens manuals and 0 occurrences in audited workbooks.",
    },
    "run_drc": {
        "command": "run_drc",
        "status": "UNVERIFIED_IN_CORPUS",
        "definitions_found": 0,
        "source_evidence": None,
        "note": "Legacy candidate in KNOWN_COMMANDS allowlist with 0 definitions in Siemens manuals and 0 occurrences in audited workbooks.",
    },
    "set_transition_hold": {
        "command": "set_transition_hold",
        "status": "UNVERIFIED_IN_CORPUS",
        "definitions_found": 0,
        "source_evidence": None,
        "note": "Legacy candidate in KNOWN_COMMANDS allowlist with 0 definitions in Siemens manuals and 0 occurrences in audited workbooks.",
    },
    "verify_scan_chains": {
        "command": "verify_scan_chains",
        "status": "UNVERIFIED_IN_CORPUS",
        "definitions_found": 0,
        "source_evidence": None,
        "note": "Legacy candidate in KNOWN_COMMANDS allowlist with 0 definitions in Siemens manuals and 0 occurrences in audited workbooks.",
    },
    "write_tcl_library": {
        "command": "write_tcl_library",
        "status": "UNVERIFIED_IN_CORPUS",
        "definitions_found": 0,
        "source_evidence": None,
        "note": "Legacy candidate in KNOWN_COMMANDS allowlist with 0 definitions in Siemens manuals and 0 occurrences in audited workbooks.",
    },
}

# Primary Module & Flow Step classification for Canonical Commands
FLOW_MAPPINGS: Dict[str, Dict[str, str]] = {
    "add_black_boxes": {"primary_module": "ATPG", "flow_step": "Design & Blackbox Configuration"},
    "add_cell_models": {"primary_module": "SCAN", "flow_step": "Test Logic Library Setup"},
    "add_clocks": {"primary_module": "ATPG", "flow_step": "Clock Definition & Timing Setup"},
    "add_faults": {"primary_module": "ATPG", "flow_step": "Fault Model & Target Setup"},
    "add_scan_chains": {"primary_module": "SCAN", "flow_step": "Scan Chain Definition"},
    "add_scan_groups": {"primary_module": "SCAN", "flow_step": "Scan Grouping Specification"},
    "add_scan_mode": {"primary_module": "SCAN", "flow_step": "Scan Mode Configuration"},
    "analyze_scan_chains": {"primary_module": "SCAN", "flow_step": "Scan Chain Synthesis & Analysis"},
    "check_design_rules": {"primary_module": "SCAN", "flow_step": "Design Rule Checking (DRC)"},
    "create_patterns": {"primary_module": "ATPG", "flow_step": "Test Pattern Generation & Compression"},
    "delete_design": {"primary_module": "ATPG", "flow_step": "Design Management & Cleanup"},
    "insert_test_logic": {"primary_module": "SCAN", "flow_step": "Test Logic Insertion"},
    "read_cell_library": {"primary_module": "SCAN", "flow_step": "Cell Library Ingestion"},
    "read_patterns": {"primary_module": "ATPG", "flow_step": "Pattern Ingestion & Verification"},
    "read_verilog": {"primary_module": "SCAN", "flow_step": "Netlist Ingestion"},
    "report_scan_cells": {"primary_module": "SCAN", "flow_step": "Scan Cell Inspection & Reporting"},
    "report_scan_chains": {"primary_module": "SCAN", "flow_step": "Scan Chain Reporting"},
    "report_scan_elements": {"primary_module": "SCAN", "flow_step": "Scan Element Reporting"},
    "report_statistics": {"primary_module": "ATPG", "flow_step": "Test Coverage & Statistics Reporting"},
    "report_test_logic": {"primary_module": "SCAN", "flow_step": "Test Logic Inspection"},
    "set_atpg_limits": {"primary_module": "ATPG", "flow_step": "ATPG Effort & Limit Configuration"},
    "set_context": {"primary_module": "TSHELL_LINUX", "flow_step": "Tool Context Initialization"},
    "set_current_design": {"primary_module": "TSHELL_LINUX", "flow_step": "Current Root Design Selection"},
    "set_edt_options": {"primary_module": "EDT", "flow_step": "EDT Compression Configuration"},
    "set_fault_type": {"primary_module": "ATPG", "flow_step": "Fault Model Selection"},
    "set_insertion_options": {"primary_module": "SCAN", "flow_step": "Scan Insertion Options"},
    "set_output_masks": {"primary_module": "EDT", "flow_step": "EDT Channel Masking Setup"},
    "set_scan_insertion_options": {"primary_module": "SCAN", "flow_step": "Scan Synthesis Options"},
    "set_scan_signals": {"primary_module": "SCAN", "flow_step": "Scan Signal Pin Assignment"},
    "set_system_mode": {"primary_module": "TSHELL_LINUX", "flow_step": "Tool Mode Transition (Setup/Analysis)"},
    "set_test_logic": {"primary_module": "SCAN", "flow_step": "Test Point & Logic Options"},
    "write_atpg_setup": {"primary_module": "SCAN", "flow_step": "ATPG Setup Handoff Export"},
    "write_design": {"primary_module": "SCAN", "flow_step": "Scan-Inserted Netlist Export"},
    "write_patterns": {"primary_module": "ATPG", "flow_step": "Pattern File Output & Export"},
}


def _extract_definition_metadata(raw_text: str, command_name: str) -> Dict[str, Any]:
    """Parses context, mode, purpose, and syntax from verified raw manual text."""
    # Context
    m_ctx = re.search(r"Context:\s*([^\n]+)", raw_text)
    context = m_ctx.group(1).strip() if m_ctx else "all contexts"

    # Mode
    m_mode = re.search(r"Mode:\s*([^\n]+)", raw_text)
    mode = m_mode.group(1).strip() if m_mode else "all modes"

    # Purpose / Description (text between Mode and Usage)
    purpose = ""
    if m_mode:
        post_mode = raw_text[m_mode.end():]
        m_usage = re.search(r"\nUsage\b", post_mode)
        if m_usage:
            desc_block = post_mode[:m_usage.start()].strip()
            # Remove Prerequisites block if present
            desc_clean = re.sub(r"Prerequisites:.*?(?=\n[A-Z]|\Z)", "", desc_block, flags=re.DOTALL).strip()
            lines = [l.strip() for l in desc_clean.splitlines() if l.strip()]
            if lines:
                purpose = " ".join(lines)
    if not purpose:
        raise ValueError(f"Could not extract authoritative purpose for command '{command_name}' from manual text")

    # Syntax (Usage block)
    syntax = ""
    m_usage_all = re.search(r"\nUsage\s*\n", raw_text)
    if m_usage_all:
        post_usage = raw_text[m_usage_all.end():]
        m_term = re.search(
            r"\n(?:Description|Arguments|Examples|Related Topics)\b", post_usage
        )
        usage_block = (
            post_usage[:m_term.start()].strip() if m_term else post_usage[:400].strip()
        )
        clean_lines = [l.rstrip() for l in usage_block.splitlines() if l.strip()]
        syntax = "\n".join(clean_lines)
    if not syntax:
        raise ValueError(f"Could not extract authoritative syntax for command '{command_name}' from manual text")

    return {
        "context": context,
        "mode": mode,
        "purpose": purpose,
        "syntax": syntax,
    }


def canonical_domain_for_course(course_name):
    """Canonical taxonomy domain for a course / module key.

    The canonical domain vocabulary is the "domains" list of
    knowledge/canonical_taxonomy.json (ATPG, SCAN, EDT, TSHELL_LINUX), the
    same vocabulary knowledge/command_index.json stores in its "module" and
    "modules" fields.  A course is not always a domain: the Tessent Shell
    course is the TSHELL_LINUX domain and the ATPG Advanced course belongs
    to the ATPG domain.
    """
    name = str(course_name or "").strip().upper()
    if name == "TSHELL":
        return "TSHELL_LINUX"
    if name == "ATPG_ADV":
        return "ATPG"
    return name


def _extract_workbook_usages(
    multimodule_path: Path, full_audit_path: Path
) -> Dict[str, Dict[str, Any]]:
    """Gathers workbook references and module co-occurrences across audit baselines."""
    usages: Dict[str, Dict[str, Any]] = {}

    if multimodule_path.exists():
        with open(multimodule_path, "r", encoding="utf-8") as f:
            multi_data = json.load(f)
        for mod_name, mod_info in multi_data.get("modules", {}).items():
            for item in mod_info.get("items", []):
                for cmd in item.get("extracted_commands", []):
                    entry = usages.setdefault(
                        cmd, {"modules": set(), "workbook_references": []}
                    )
                    # Normalize module domain name
                    domain = canonical_domain_for_course(mod_name)
                    entry["modules"].add(domain)
                    entry["workbook_references"].append(
                        {
                            "module": mod_name,
                            "item_id": item.get("item_id"),
                            "source_reference": item.get("source_reference"),
                        }
                    )

    if full_audit_path.exists():
        with open(full_audit_path, "r", encoding="utf-8") as f:
            full_data = json.load(f)
        for cmd, info in full_data.get("known_commands_results", {}).items():
            if info.get("used_in_count", 0) > 0:
                entry = usages.setdefault(
                    cmd, {"modules": set(), "workbook_references": []}
                )
                entry["modules"].add("ATPG")

    return usages


def build_command_index(output_file: Optional[Path] = None) -> Dict[str, Any]:
    """Generates the canonical command index knowledge/command_index.json deterministically.
    
    Returns:
        The generated index dictionary conforming to §8.3 metadata schema.
    """
    if output_file is None:
        output_file = COMMAND_INDEX_FILE

    # Lazy-import definition retriever to keep index queries lightweight
    from core.rag_builder import retrieve_command_definition

    with open(COMMAND_DEFS_FILE, "r", encoding="utf-8") as f:
        cmd_defs = json.load(f)

    workbook_usages = _extract_workbook_usages(MULTIMODULE_AUDIT_FILE, FULL_AUDIT_FILE)

    verified_cmds = sorted(
        [c for c, v in cmd_defs["results"].items() if v["definitions_found"] > 0]
    )

    commands_dict: Dict[str, Any] = {}

    for cmd in verified_cmds:
        retrieval = retrieve_command_definition(cmd)
        if not retrieval:
            raise RuntimeError(f"Failed to retrieve verified definition for command '{cmd}'")

        primary_chunk = retrieval[0]
        source_file = primary_chunk.get("source", "knowledge/txt/tshell_ref (1).txt")
        if not source_file.startswith("knowledge/"):
            source_file = f"knowledge/txt/{source_file}"
        page_num = primary_chunk.get("page", 1)
        raw_text = primary_chunk.get("text", "")

        parsed = _extract_definition_metadata(raw_text, cmd)
        flow_info = FLOW_MAPPINGS.get(
            cmd, {"primary_module": "ATPG", "flow_step": "General Flow"}
        )

        # Module domain sets
        wb_info = workbook_usages.get(cmd, {"modules": set(), "workbook_references": []})
        applicable_modules = set(wb_info["modules"])
        applicable_modules.add(flow_info["primary_module"])

        # Also map from Context field if relevant
        ctx_lower = parsed["context"].lower()
        if "edt" in ctx_lower:
            applicable_modules.add("EDT")
        if "scan" in ctx_lower:
            applicable_modules.add("SCAN")
        if "patterns" in ctx_lower:
            applicable_modules.add("ATPG")
        if "all contexts" in ctx_lower:
            applicable_modules.add("TSHELL_LINUX")

        # Invert KNOWN_ALIASES to discover aliases for this canonical command
        aliases_for_cmd = sorted(
            [
                alias_name
                for alias_name, ainfo in KNOWN_ALIASES.items()
                if ainfo["canonical_command"] == cmd
            ]
        )

        record = {
            "command": cmd,
            "canonical_name": cmd,
            "aliases": aliases_for_cmd,
            "module": flow_info["primary_module"],
            "modules": sorted(list(applicable_modules)),
            "flow_step": flow_info["flow_step"],
            "content_type": "command_reference",
            "syntax": parsed["syntax"],
            "purpose": parsed["purpose"],
            "mode": parsed["mode"],
            "context": parsed["context"],
            "difficulty": "introductory" if flow_info["primary_module"] in ["TSHELL_LINUX", "SCAN"] else "intermediate",
            "source_evidence": {
                "file": source_file,
                "page": page_num,
                "section": "Command Reference",
                "verified": True,
            },
            "workbook_references": sorted(
                wb_info["workbook_references"],
                key=lambda x: (x.get("module", ""), x.get("item_id", "")),
            ),
        }
        commands_dict[cmd] = record

    alias_map = {
        alias_name: info["canonical_command"]
        for alias_name, info in sorted(KNOWN_ALIASES.items())
    }

    index_payload = {
        "index_version": "1.0.0",
        "approved_plan_reference": "PHASE_1_PLAN_APPROVED.md §5.D, §8.2, §8.3; PHASE_2_PREFLIGHT_REPORT.md Stage 2.2",
        "generated_at": "2026-10-16T15:00:00Z",  # Deterministic timestamp
        "total_canonical_commands": len(commands_dict),
        "total_aliases": len(KNOWN_ALIASES),
        "total_unverified_candidates": len(UNVERIFIED_CANDIDATES),
        "commands": {k: commands_dict[k] for k in sorted(commands_dict.keys())},
        "aliases": {k: KNOWN_ALIASES[k] for k in sorted(KNOWN_ALIASES.keys())},
        "alias_map": alias_map,
        "unverified_candidates": {
            k: UNVERIFIED_CANDIDATES[k] for k in sorted(UNVERIFIED_CANDIDATES.keys())
        },
    }

    # Validate index constraints
    _validate_command_index(index_payload)

    # Deterministic write
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(index_payload, f, indent=2, sort_keys=True)

    return index_payload


def _validate_command_index(index_payload: Dict[str, Any]) -> None:
    """Enforces §8.3 schema integrity and Stage 2.2 acceptance criteria."""
    commands = index_payload.get("commands", {})
    aliases = index_payload.get("aliases", {})
    alias_map = index_payload.get("alias_map", {})

    if len(commands) != 34:
        raise ValueError(f"Expected 34 verified commands, found {len(commands)}")

    # Check unique identifiers and non-null required fields
    seen_cmds = set()
    for name, rec in commands.items():
        if name in seen_cmds:
            raise ValueError(f"Duplicate canonical command identifier: {name}")
        seen_cmds.add(name)

        # Plan §8.3 & Stage 2.2 mandatory non-null fields
        for field in ["command", "canonical_name", "module", "syntax", "purpose"]:
            if not rec.get(field):
                raise ValueError(f"Command '{name}' is missing mandatory field '{field}'")

        evidence = rec.get("source_evidence")
        if not evidence or not evidence.get("file") or not evidence.get("page"):
            raise ValueError(f"Command '{name}' missing valid source_evidence (file, page)")

        if not isinstance(evidence.get("page"), int) or evidence["page"] <= 0:
            raise ValueError(f"Command '{name}' has invalid page number: {evidence.get('page')}")

    # Check aliases
    for alias_name, alias_rec in aliases.items():
        target = alias_rec.get("canonical_command")
        if not target or target not in commands:
            raise ValueError(f"Alias '{alias_name}' points to non-existent target '{target}'")
        if alias_name in commands:
            raise ValueError(f"Alias collision: '{alias_name}' is also a canonical command")
        if alias_map.get(alias_name) != target:
            raise ValueError(f"alias_map mismatch for alias '{alias_name}'")


# In-memory index cache for instant queries (<1ms)
_CACHED_INDEX: Optional[Dict[str, Any]] = None


def load_command_index(index_path: Optional[Path] = None, force_reload: bool = False) -> Dict[str, Any]:
    """Loads and caches the command index from disk.

    Preference order: an explicit ``index_path``, then the expanded WP-2 index
    (``knowledge/command_index_full.json``) when it has been built, then the
    curated 34-command index.
    """
    global _CACHED_INDEX
    if index_path is not None:
        # An explicit path is a one-off read (tests and tooling validate the
        # curated artifact this way). It must NOT replace the shared cache, or a
        # later default load/search would silently use the smaller index.
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if _CACHED_INDEX is not None and not force_reload:
        return _CACHED_INDEX

    if FULL_COMMAND_INDEX_FILE.exists():
        path = FULL_COMMAND_INDEX_FILE
    else:
        path = COMMAND_INDEX_FILE
    if not path.exists():
        # Auto-generate if missing
        return build_command_index(path)

    with open(path, "r", encoding="utf-8") as f:
        _CACHED_INDEX = json.load(f)
    return _CACHED_INDEX


def get_command(name: str) -> Optional[Dict[str, Any]]:
    """Looks up a command by exact name or alias (case-insensitive).
    
    Returns the canonical command record, augmented with alias resolution metadata if applicable.
    """
    index = load_command_index()
    norm = name.strip().lower()

    # Exact canonical match
    if norm in index.get("commands", {}):
        return index["commands"][norm]

    # Alias match
    if norm in index.get("alias_map", {}):
        canonical = index["alias_map"][norm]
        rec = dict(index["commands"][canonical])
        rec["resolved_from_alias"] = norm
        rec["alias_metadata"] = index["aliases"].get(norm)
        return rec

    return None


if __name__ == "__main__":
    print(f"Building canonical command index at: {COMMAND_INDEX_FILE}")
    idx = build_command_index()
    print(
        f"SUCCESS: Indexed {idx['total_canonical_commands']} canonical commands, "
        f"{idx['total_aliases']} aliases, {idx['total_unverified_candidates']} unverified candidates."
    )
