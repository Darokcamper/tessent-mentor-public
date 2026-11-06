"""Expand the canonical command index from the real Tessent Shell corpus (WP-2).

The historical ``knowledge/command_index.json`` holds 34 curated commands with
hand-audited module / flow-step / workbook metadata. The Shell Reference Manual
documents far more entries (the "``cmdname`` followed by ``Context:``" pattern
matches ~930 of them), so the Commands Explorer could only ever show 3 of the 32
``write_*`` commands even though its search logic was correct.

This module derives every documented entry straight from the cached page text and
writes an ADDITIVE artifact (``knowledge/command_index_full.json``). The original
file is never rewritten.

Grounding rules enforced here:

* Nothing is invented. Command name, context, mode, description, syntax and
  examples are copied from the manual text; a field the manual does not contain is
  emitted as ``None`` with an explicit ``*_unavailable`` flag.
* ``module`` / ``modules`` are derived only from the manual's own ``Context:``
  field and are flagged ``module_inferred``; curated records keep their audited
  values untouched.
* ``flow_step`` and ``difficulty`` are never guessed for new entries; they stay
  ``None`` with an ``*_unavailable`` flag so the UI can say "not recorded in the
  available sources" instead of showing an invented stage.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
TXT_DIR = KNOWLEDGE_DIR / "txt"
CURATED_INDEX_FILE = KNOWLEDGE_DIR / "command_index.json"
FULL_INDEX_FILE = KNOWLEDGE_DIR / "command_index_full.json"

INDEX_VERSION = "2.0.0"

# Same entry shape the retrieval fast path uses (core/rag_builder.py).
_ENTRY_RE = re.compile(r"\n([a-z][a-z0-9_]{2,60})\s*\n\s*Context:", re.IGNORECASE)
_PAGE_RE = re.compile(r"--- PAGE (\d+) ---")

_CONTEXT_RE = re.compile(r"Context:\s*(.+)", re.IGNORECASE)
_MODE_RE = re.compile(r"Mode:\s*(.+)", re.IGNORECASE)

_SECTION_ENDS = ("arguments", "options", "example", "examples", "related", "note", "returns", "see also")
_USAGE_START = re.compile(r"^\s*(usage|syntax)\s*$", re.IGNORECASE)
_EXAMPLE_START = re.compile(r"^\s*(example|examples)\s*:?\s*$", re.IGNORECASE)

_ARG_RE = re.compile(r"(?<![\w-])(-[A-Za-z][\w-]*|\{[^}\n]{1,80}\})")

_PURPOSE_LIMIT = 600
_SYNTAX_LIMIT = 2000
_EXAMPLE_LIMIT = 1500

# Only the four canonical domains are ever assigned.
_DOMAIN_HINTS = (
    ("EDT", re.compile(r"\bedt\b", re.IGNORECASE)),
    ("SCAN", re.compile(r"\bscan\b", re.IGNORECASE)),
    ("ATPG", re.compile(r"\b(atpg|patterns)\b", re.IGNORECASE)),
)


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r", "")).strip()


def _page_for_offset(page_spans: List[tuple], offset: int) -> Optional[int]:
    page = None
    for start, number in page_spans:
        if start <= offset:
            page = number
        else:
            break
    return page


def _section_after(lines: List[str], start: int, limit: int) -> str:
    """Collect lines after *start*, stopping at the next known section heading."""
    collected: List[str] = []
    total = 0
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            if collected:
                collected.append("")
            continue
        lowered = stripped.lower().rstrip(":").strip()
        if lowered in _SECTION_ENDS or re.match(r"^[a-z][a-z0-9_]{2,60}$", stripped):
            break
        collected.append(stripped)
        total += len(stripped) + 1
        if total >= limit:
            break
    return _clean("\n".join(collected))[:limit]


def _derive_modules(context: str) -> List[str]:
    """Domains evidenced by the manual's own Context field (never guessed)."""
    modules: List[str] = []
    for domain, pattern in _DOMAIN_HINTS:
        if pattern.search(context or ""):
            modules.append(domain)
    return modules


def parse_entry(command: str, block: str, source_file: str, page: Optional[int]) -> Dict[str, Any]:
    """Build one index record from a single manual entry block."""
    lines = block.split("\n")
    context = ""
    mode = ""
    for line in lines[:6]:
        match = _CONTEXT_RE.search(line)
        if match and not context:
            context = _clean(match.group(1))
        match = _MODE_RE.search(line)
        if match and not mode:
            mode = _clean(match.group(1))

    # Description = first prose AFTER the Mode line (or after the Context line when
    # there is no Mode line). Starting after the header prevents a wrapped Context
    # continuation such as "-scan_retargeting, patterns -scan_diagnosis" from being
    # mistaken for the command's purpose.
    purpose_lines: List[str] = []
    usage_index = None
    example_index = None
    mode_index = None
    for idx, line in enumerate(lines[:8]):
        if _MODE_RE.search(line):
            mode_index = idx
            break
    prose_start = (mode_index + 1) if mode_index is not None else 1
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if usage_index is None and _USAGE_START.match(stripped):
            usage_index = idx
            continue
        if example_index is None and _EXAMPLE_START.match(stripped):
            example_index = idx
            continue
        if idx < prose_start:
            continue
        if usage_index is None and example_index is None:
            if not stripped or _CONTEXT_RE.search(stripped) or _MODE_RE.search(stripped) or stripped.lower() == command:
                continue
            purpose_lines.append(stripped)

    purpose = _clean(" ".join(purpose_lines))[:_PURPOSE_LIMIT]
    syntax = _section_after(lines, usage_index + 1, _SYNTAX_LIMIT) if usage_index is not None else ""
    examples = _section_after(lines, example_index + 1, _EXAMPLE_LIMIT) if example_index is not None else ""
    if examples and examples.lower().rstrip(":").strip() in _SECTION_ENDS:
        examples = ""

    arguments: List[str] = []
    for match in _ARG_RE.finditer(syntax or ""):
        token = _clean(match.group(1))
        if token and token not in arguments:
            arguments.append(token)
        if len(arguments) >= 80:
            break

    modules = _derive_modules(context)
    return {
        "command": command,
        "canonical_name": command,
        "purpose": purpose or None,
        "purpose_unavailable": not bool(purpose),
        "syntax": syntax or None,
        "syntax_unavailable": not bool(syntax),
        "arguments": arguments,
        "arguments_unavailable": not bool(arguments),
        "arguments_note": (
            "Switch and brace-placeholder tokens captured verbatim from the manual's "
            "Usage/Syntax block."
            if arguments
            else "The manual entry does not list arguments or options."
        ),
        "examples": examples or None,
        "examples_unavailable": not bool(examples),
        "context": context,
        "mode": mode,
        "module": modules[0] if modules else None,
        "modules": modules,
        "module_inferred": True,
        "module_evidence": context,
        "flow_step": None,
        "flow_step_unavailable": True,
        "difficulty": None,
        "difficulty_unavailable": True,
        "content_type": "command_reference",
        "source_evidence": {
            "file": source_file,
            "page": page,
            "section": "Command Reference",
            "verified": True,
        },
        "workbook_references": [],
    }


def scan_corpus(txt_dir: Path = TXT_DIR) -> Dict[str, Dict[str, Any]]:
    """Every documented command entry in the cached corpus; first definition wins."""
    found: Dict[str, Dict[str, Any]] = {}
    if not txt_dir.exists():
        return found

    # Reference manual first so its full entries win over short mentions elsewhere.
    files = sorted(
        txt_dir.glob("*.txt"),
        key=lambda p: (0 if "tshell_ref" in p.name.lower() else 1, p.name),
    )
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        page_spans = [(m.end(), int(m.group(1))) for m in _PAGE_RE.finditer(text)]
        for match in _ENTRY_RE.finditer(text):
            command = match.group(1).lower()
            if command in found:
                continue
            next_match = _ENTRY_RE.search(text, match.end())
            end = next_match.start() if next_match else min(len(text), match.start() + 12000)
            page = _page_for_offset(page_spans, match.start())
            found[command] = parse_entry(command, text[match.start():end], path.name, page)
    return found


def _merge_curated(curated_record: Dict[str, Any], discovered_record: Dict[str, Any]) -> Dict[str, Any]:
    """Curated (audited) metadata wins; only missing fields are filled from the corpus.

    Field-level availability flags are added so the UI can distinguish "no example
    recorded in the sources" from "field missing because the record predates the
    schema".
    """
    merged = dict(curated_record)
    if not merged.get("examples"):
        merged["examples"] = discovered_record.get("examples")
    if not merged.get("arguments"):
        merged["arguments"] = list(discovered_record.get("arguments") or [])
    if not merged.get("context"):
        merged["context"] = discovered_record.get("context", "")
    merged["module_inferred"] = False
    merged["purpose_unavailable"] = not bool(merged.get("purpose"))
    merged["syntax_unavailable"] = not bool(merged.get("syntax"))
    merged["arguments_unavailable"] = not bool(merged.get("arguments"))
    merged["examples_unavailable"] = not bool(merged.get("examples"))
    merged["flow_step_unavailable"] = not bool(merged.get("flow_step"))
    merged["difficulty_unavailable"] = not bool(merged.get("difficulty"))
    merged.setdefault(
        "arguments_note",
        "Switch and brace-placeholder tokens captured verbatim from the manual's Usage/Syntax block."
        if merged.get("arguments")
        else "The manual entry does not list arguments or options.",
    )
    return merged


def build_full_index(
    output_file: Path = FULL_INDEX_FILE,
    txt_dir: Path = TXT_DIR,
    curated_file: Path = CURATED_INDEX_FILE,
) -> Dict[str, Any]:
    """Merge curated metadata over the corpus-derived entries and write the artifact."""
    curated_payload: Dict[str, Any] = {}
    if curated_file.exists():
        curated_payload = json.loads(curated_file.read_text(encoding="utf-8"))
    curated: Dict[str, Any] = curated_payload.get("commands", {})

    discovered = scan_corpus(txt_dir)

    commands: Dict[str, Dict[str, Any]] = {}
    curated_preserved = 0
    for name, record in discovered.items():
        if name in curated:
            commands[name] = _merge_curated(curated[name], record)
            curated_preserved += 1
        else:
            commands[name] = record

    # A curated command whose manual entry the scanner did not reproduce is still
    # verified data; keep it rather than silently dropping it.
    curated_only = 0
    for name, record in curated.items():
        if name not in commands:
            commands[name] = _merge_curated(record, {})
            curated_only += 1

    index = {
        "index_version": INDEX_VERSION,
        "generated_from": "knowledge/txt ('cmdname' + 'Context:' entries)",
        "approved_plan_reference": curated_payload.get("approved_plan_reference"),
        "supersedes_scope": (
            "Additive superset of knowledge/command_index.json; the original artifact is "
            "preserved byte-for-byte for baseline comparison."
        ),
        "total_canonical_commands": len(commands),
        "curated_records_preserved": curated_preserved,
        "curated_records_without_corpus_entry": curated_only,
        "commands": dict(sorted(commands.items())),
        "aliases": curated_payload.get("aliases", {}),
        "alias_map": curated_payload.get("alias_map", {}),
        "unverified_candidates": curated_payload.get("unverified_candidates", {}),
    }
    output_file.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def validate_full_index(index: Dict[str, Any], minimum_commands: int = 800) -> List[str]:
    """Return a list of problems; empty means the artifact is usable."""
    problems: List[str] = []
    commands = index.get("commands", {})
    if len(commands) < minimum_commands:
        problems.append(f"only {len(commands)} commands indexed (expected >= {minimum_commands})")

    unverified = index.get("unverified_candidates", {})
    for name, record in commands.items():
        evidence = record.get("source_evidence") or {}
        if not evidence.get("file") or not evidence.get("page"):
            problems.append(f"{name}: missing source_evidence file/page")
        if not record.get("purpose") and not record.get("purpose_unavailable"):
            problems.append(f"{name}: purpose is empty without an availability flag")
        if not record.get("syntax") and not record.get("syntax_unavailable"):
            problems.append(f"{name}: syntax is empty without an availability flag")
        if not record.get("examples") and not record.get("examples_unavailable"):
            problems.append(f"{name}: examples are empty without an availability flag")
        if name in unverified:
            problems.append(f"{name}: unverified candidate leaked into canonical commands")
    return problems


def main() -> int:
    index = build_full_index()
    problems = validate_full_index(index)
    write_commands = sum(1 for name in index["commands"] if name.startswith("write_"))
    print(f"indexed commands      : {index['total_canonical_commands']}")
    print(f"write_* commands      : {write_commands}")
    print(f"curated preserved     : {index['curated_records_preserved']}")
    print(f"curated without entry : {index['curated_records_without_corpus_entry']}")
    if problems:
        print("validation problems:")
        for problem in problems[:20]:
            print("  -", problem)
        return 1
    print("validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
