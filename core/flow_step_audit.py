"""WP-4: audit the curated `flow_step` metadata against the manual page it cites.

`flow_step` labels ("Clock Definition & Timing Setup", "ATPG Setup Handoff Export", …) are
**editorial taxonomy**, not manual quotations. This module measures how much of each label is
actually supported by the page the command cites, and classifies every mapping:

* ``documented``  — the page text contains most of the label's distinctive words;
* ``editorial``   — the page supports the command but not the flow label (most cases);
* ``unsupported`` — the page could not be read at all;
* ``unavailable`` — the command has no flow_step.

Nothing is relabelled here: an incorrect label cannot be identified by keyword overlap alone,
so the audit reports and the UI labels the metadata as curated.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TXT_DIR = PROJECT_ROOT / "knowledge" / "txt"

_STOP = frozenset("""
and the for with from into its that this step export setup configuration selection
ingestion analysis generation reporting insertion specification verification definition
""".split())
_WORD = re.compile(r"[a-z0-9_]+")
_PAGE_SPLIT = re.compile(r"---\s*PAGE\s+(\d+)\s*---", re.IGNORECASE)


def _tokens(text: str) -> List[str]:
    return [t for t in _WORD.findall(str(text or "").lower())
            if len(t) >= 4 and t not in _STOP]


def load_page_text(file_name: str, page: int, cache: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Best-effort page text from the cached corpus text files."""
    if cache is not None and file_name in cache:
        raw = cache[file_name]
    else:
        path = TXT_DIR / Path(str(file_name)).name
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
        if cache is not None:
            cache[file_name] = raw
    marks = list(_PAGE_SPLIT.finditer(raw))
    if not marks:
        return None
    for index, match in enumerate(marks):
        if int(match.group(1)) == int(page):
            end = marks[index + 1].start() if index + 1 < len(marks) else len(raw)
            return raw[match.end():end]
    return None


def classify(command: str, entry: Dict[str, Any], cache: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    flow_step = entry.get("flow_step")
    if not flow_step:
        return {"command": command, "flow_step": None, "verdict": "unavailable",
                "support": 0.0, "label_tokens": [], "matched": []}

    evidence = entry.get("source_evidence") or {}
    page_text = load_page_text(evidence.get("file", ""), int(evidence.get("page") or 0), cache)
    if page_text is None:
        return {"command": command, "flow_step": flow_step, "verdict": "unsupported",
                "support": 0.0, "label_tokens": _tokens(flow_step), "matched": []}

    label_tokens = _tokens(flow_step)
    page_tokens = set(_tokens(page_text))
    matched = [token for token in label_tokens if token in page_tokens]
    support = round(len(matched) / len(label_tokens), 3) if label_tokens else 0.0
    if support >= 0.6:
        verdict = "documented"
    elif support > 0.0:
        verdict = "editorial"
    else:
        verdict = "editorial"
    return {"command": command, "flow_step": flow_step, "verdict": verdict,
            "support": support, "label_tokens": label_tokens, "matched": matched}


def audit(index: Optional[Dict[str, Any]] = None,
          limit: Optional[int] = None) -> Dict[str, Any]:
    if index is None:
        from core.command_indexer import load_command_index

        index = load_command_index().get("commands", {})
    cache: Dict[str, str] = {}
    rows = []
    for command in sorted(index):
        rows.append(classify(command, index[command], cache))
        if limit and len(rows) >= limit:
            break
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    documented = [row for row in rows if row["verdict"] == "documented"]
    editorial = [row for row in rows if row["verdict"] == "editorial"]
    return {
        "work_package": "WP-4 flow_step audit",
        "commands_scanned": len(rows),
        "verdicts": counts,
        "documented": documented,
        "editorial": editorial,
        "unsupported": [row for row in rows if row["verdict"] == "unsupported"],
        "note": ("flow_step is curated taxonomy. 'documented' means the cited page contains the "
                 "label's own vocabulary; 'editorial' means it does not — the label is a human "
                 "classification and is presented as such in the UI."),
    }
