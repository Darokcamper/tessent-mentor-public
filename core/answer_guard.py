"""Nothing reaches the user with a citation the knowledge base cannot prove.

Root cause this closes (2026-11-07). A GLS question was answered with
``[Source: GLS_Simulation_Types, Page 1]``, ``[Source: Scan_DRC_part_1, Page 1]`` and
``[Source: DFT_Overview, Page 1]``. None of those three files exists in ``knowledge/txt``;
the verifier said ``PAGE NOT FOUND`` for all three. Three contributing causes:

* the agent prompt itself recommended a source name that does not exist
  (``source_filter`` example ``"Scan_DRC_part_2"``), which the model copied and
  incremented into ``Scan_DRC_part_1``;
* neither agent funnel ran the verifier that already existed in
  :mod:`core.citation_check`, so an invented pointer looked exactly like a checked one;
* the GLS knowledge needed to answer at all is not in the corpus, so nothing could
  have been grounded even with a perfect prompt.

The rule enforced here is one sentence: **a citation is shown only when the cited page
exists and carries the words the sentence is built on.** Everything else is deleted from
the answer, and an answer left with no citation says plainly that it is not grounded.

The claim text itself is never rewritten - only the pointer is judged.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.citation_check import claim_span, corpus_available, verify_citation

# Bracketed form the prompts ask for, plus the "Source Document: file (Page N)" form the
# retrieval tool prints into observations, which models echo back as if it were a citation.
_CITATION_PATTERNS = (
    re.compile(r"\[Source:\s*(?P<source>[^\[\],]+?)\s*,\s*Page\s*(?P<page>\d{1,5})\s*\]"),
    re.compile(r"Source(?:\s+Document)?:\s*(?P<source>[^\[\](),\n]+?)\s*[,(]\s*"
               r"Page\s*(?P<page>\d{1,5})\s*\)?"),
)

UNGROUNDED_NOTE = (
    "> \u26a0\ufe0f **Not grounded in the Tessent knowledge base.** No verified page supports "
    "these statements, so no page is cited. Treat this as general engineering background and "
    "check it against the manuals before relying on it."
)

UNVERIFIABLE_NOTE = (
    "> \u26a0\ufe0f **Citations withheld.** The knowledge base could not be checked on this "
    "host, so no page is cited. Treat this as general engineering background."
)



def _find_citations(text: str) -> List[Dict[str, Any]]:
    """Every citation in *text*, in order, with its span and matched groups."""
    found: List[Dict[str, Any]] = []
    for pattern in _CITATION_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            if any(other["span"][0] <= start < other["span"][1] for other in found):
                continue  # already captured by a more specific pattern
            found.append({
                "span": (start, end),
                "source": match.group("source").strip().strip('"').strip("'"),
                "page": int(match.group("page")),
                "raw": match.group(0),
            })
    return sorted(found, key=lambda item: item["span"][0])


def _tidy(text: str) -> str:
    """Remove the holes a deleted citation leaves behind."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def strip_unsupported_citations(text: str, root: Optional[Path] = None) -> Dict[str, Any]:
    """Return *text* with every unprovable citation removed.

    ``kept`` lists the citations that survived (``verified``, or ``page_shifted`` rewritten
    to the page that really prints the words). ``removed`` lists what was deleted and why.
    ``grounded`` is True only when at least one citation survived.
    """
    body = str(text or "")
    found = _find_citations(body)
    if not found:
        return {"text": body, "kept": [], "removed": [], "grounded": False, "note": ""}

    checking = corpus_available(root)
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    replacements: List[tuple] = []

    for item in found:
        claim = claim_span(body, item["span"][0])
        check = verify_citation(item["source"], item["page"], claim, root=root)
        status = check.get("status")
        if status in ("verified", "page_shifted"):
            page = check.get("correct_page") or item["page"]
            kept.append({"source": item["source"], "page": page, "status": status})
            if status == "page_shifted":
                replacements.append((item["span"],
                                     f"[Source: {item['source']}, Page {page}]"))
            continue
        reason = ("page not found in the knowledge base"
                  if check.get("page_exists") is False
                  else check.get("reason") or "could not be verified")
        removed.append({"source": item["source"], "page": item["page"], "reason": reason})
        replacements.append((item["span"], ""))

    guarded = body
    for (start, end), replacement in sorted(replacements, key=lambda pair: pair[0][0],
                                            reverse=True):
        guarded = guarded[:start] + replacement + guarded[end:]
    guarded = _tidy(guarded)

    note = ""
    if not kept:
        note = UNGROUNDED_NOTE if checking else UNVERIFIABLE_NOTE
        guarded = f"{guarded}\n\n{note}".strip() if guarded else note

    return {"text": guarded, "kept": kept, "removed": removed,
            "grounded": bool(kept), "note": note}


def guard_or_strip(text: str, root: Optional[Path] = None) -> str:
    """Text safe to display. Never fails open.

    If the verifier itself raises, every citation marker is removed rather than trusted -
    an outage in the checker must not become an outage in the guarantee.
    """
    try:
        return strip_unsupported_citations(text, root=root)["text"]
    except Exception:  # pragma: no cover - defensive
        return _tidy(re.sub(r"\[Source:[^\]]*\]", "", str(text or "")))


def guard_line(result: Dict[str, Any]) -> str:
    """One short line for the UI, so the reader knows what happened to the citations."""
    kept = len(result.get("kept") or [])
    removed = len(result.get("removed") or [])
    if not kept and not removed:
        return ""
    if not kept:
        return f"0 citations verified \u00b7 {removed} removed"
    if not removed:
        return f"{kept} citation(s) verified against the cited page"
    return f"{kept} citation(s) verified \u00b7 {removed} removed as unprovable"

    return text.strip()
