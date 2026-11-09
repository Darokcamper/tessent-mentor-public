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

OUTSIDE_SOURCE_NOTE = (
    "> \u2139\ufe0f **Outside the verified sources.** This answer is general engineering "
    "knowledge, not a statement taken from the Tessent manuals, so no manual page is cited "
    "for it. Confirm anything you plan to act on against the manuals."
)

OUTSIDE_SOURCE_SEARCHED_NOTE = (
    "> \u2139\ufe0f **Outside the verified sources.** The knowledge base was searched and holds "
    "no page that supports this answer, so no manual page is cited. Treat it as general "
    "engineering knowledge and confirm it against the manuals."
)

UNVERIFIABLE_NOTE = (
    "> \u26a0\ufe0f **Citations withheld, sources not checked.** The knowledge base is not "
    "available on this host, so nothing could be verified and no page is cited. Treat this as "
    "general engineering knowledge."
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

    return {"text": guarded, "kept": kept, "removed": removed,
            "grounded": bool(kept), "note": ""}


# An answer that already refuses has stated its status; labelling it again would just say
# "not from the sources" twice. Kept here (not imported from agents/) so core stays below
# the agent layer.
_ABSTENTION_MARKERS = (
    "do not have this information",
    "do not have the exact answer",
    "not in my verified source",
    "does not contain enough detail",
    "retrieved manual context does not contain",
)


def is_abstention(text: str) -> bool:
    """True when the answer already tells the reader it is not in the sources."""
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in _ABSTENTION_MARKERS)


def guard_answer(text: str, *, retrieved: bool = False,
                 root: Optional[Path] = None) -> Dict[str, Any]:
    """Everything safe to show, plus a verdict - for *every* answer, cited or not.

    A verified citation means the answer is grounded and needs no label. Anything else is
    labelled out loud, because an answer a reader cannot tell apart from a grounded one is
    the failure this module exists to prevent. ``retrieved`` records whether the knowledge
    base was actually searched, which changes what the label can honestly claim.
    """
    try:
        result = strip_unsupported_citations(text, root=root)
    except Exception:  # pragma: no cover - defensive
        result = {"text": _tidy(re.sub(r"\[Source:[^\]]*\]", "", str(text or ""))),
                  "kept": [], "removed": [], "grounded": False, "note": ""}

    if result["kept"]:
        return result

    if is_abstention(result["text"]):
        # The answer refuses to answer; that is already the plainest possible statement
        # that it is not sourced, so no second label is added.
        return result

    if not corpus_available(root):
        note = UNVERIFIABLE_NOTE
    elif retrieved:
        note = OUTSIDE_SOURCE_SEARCHED_NOTE
    else:
        note = OUTSIDE_SOURCE_NOTE

    body = result["text"] or ""
    result["text"] = f"{body}\n\n{note}".strip() if body else note
    result["note"] = note
    result["grounded"] = False
    return result


def guard_or_strip(text: str, *, retrieved: bool = False,
                   root: Optional[Path] = None) -> str:
    """Text safe to display. Never fails open.

    If the verifier itself raises, every citation marker is removed and the answer is
    labelled, rather than trusted - an outage in the checker must not become an outage in
    the guarantee.
    """
    try:
        return guard_answer(text, retrieved=retrieved, root=root)["text"]
    except Exception:  # pragma: no cover - defensive
        stripped = _tidy(re.sub(r"\[Source:[^\]]*\]", "", str(text or "")))
        return f"{stripped}\n\n{OUTSIDE_SOURCE_NOTE}".strip()


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
