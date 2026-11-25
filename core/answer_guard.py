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

from core.citation_check import claim_terms, corpus_available, verify_citation

# Bracketed form the prompts ask for, plus the "Source Document: file (Page N)" form the
# retrieval tool prints into observations, which models echo back as if it were a citation.
_CITATION_PATTERNS = (
    re.compile(r"\[Source:\s*(?P<source>[^\[\],]+?)\s*,\s*Page\s*(?P<page>\d{1,5})\s*\]"),
    re.compile(r"Source(?:\s+Document)?:\s*(?P<source>[^\[\](),\n]+?)\s*[,(]\s*"
               r"Page\s*(?P<page>\d{1,5})\s*\)?"),
)

# Reading the user's own attachment is allowed - it is a source they supplied - but it must
# never masquerade as a manual page, because there is no page to verify. The attachment prompt
# asks for this marker, and lines carrying it are kept and disclosed instead of being deleted.
ATTACHMENT_MARKER_RE = re.compile(
    r"\[(?:Image|Attachment|Attached\s+image)\s*[:;\-\u2014]?\s*[^\[\]]{0,160}\]",
    re.IGNORECASE,
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

ATTACHMENT_NOTE = (
    "> \u2139\ufe0f **From your attachment only.** These lines were read from the file or image "
    "you attached, not from the Tessent manuals, so no manual page is cited for them."
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


def _drop_empty_sections(text: str) -> str:
    """Remove a heading left with nothing under it once its sentences were dropped.

    Without this the answer keeps a confident "### Why STA is indispensable" with no content,
    which reads like a rendering bug and, worse, like an answer that was simply cut short.
    """
    lines = str(text or "").split("\n")
    kept: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not _HEADING_RE.match(line):
            kept.append(line)
            index += 1
            continue
        end = index + 1
        body = []
        while end < len(lines) and not _HEADING_RE.match(lines[end]):
            body.append(lines[end])
            end += 1
        if any(piece.strip() for piece in body):
            kept.append(line)
            kept.extend(body)          # the section's own lines, not just its title
        index = end
    return "\n".join(kept)


_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_TABLE_DELIM_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")


def _split_sentences(line: str) -> List[str]:
    """Sentences within one line, keeping a bullet's marker and its citation attached.

    "… accepts Udfm. [Source: file, Page 2690]" must stay one unit: split before the bracket
    and the claim loses its citation while the citation becomes a sentence with no claim - both
    halves then look ungrounded, and a correctly cited sentence gets thrown away.
    """
    merged: List[str] = []
    for piece in _SENTENCE_BREAK.split(line):
        if not piece.strip():
            continue
        orphan_citation = (not claim_terms(_claim_text(piece))
                           and bool(_find_citations(piece)))
        if merged and orphan_citation:
            merged[-1] = f"{merged[-1].rstrip()} {piece.strip()}"
            continue
        merged.append(piece)
    return merged


_BULLET_RE = re.compile(r"^\s*(?:[-*+\u2022]|\d+[.)])\s+")


def _units(text: str) -> List[Dict[str, Any]]:
    """Split an answer into the units that can each be checked on their own.

    A paragraph is not one claim: in the answer that motivated this, an invented sentence about
    "process, voltage and temperature corners" sat between two correctly cited bullets and was
    never checked by anything, because only citations were verified. Headings, table rules and
    blank lines are structural and carry no claim.

    Hard-wrapped lines are joined first. Treating each printed line as a sentence split
    "ATPG is like checking if" / "all the robot's parts are connected" into two claims, and
    would also separate a citation from the sentence it belongs to.
    """
    units: List[Dict[str, Any]] = []
    buffer: List[str] = []

    def flush() -> None:
        if not buffer:
            return
        paragraph = " ".join(piece.strip() for piece in buffer)
        buffer.clear()
        for piece in _split_sentences(paragraph):
            units.append({"kind": "text", "text": piece})

    for line in str(text or "").split("\n"):
        stripped = line.strip()
        if not stripped:
            flush()
            units.append({"kind": "blank", "text": ""})
        elif (_HEADING_RE.match(line) or _TABLE_DELIM_RE.match(stripped)
              or stripped in {"---", "***", "___"}):
            flush()
            units.append({"kind": "structural", "text": line})
        elif _BULLET_RE.match(line) or stripped.startswith("|"):
            flush()
            buffer.append(stripped)
        else:
            buffer.append(stripped)
    flush()
    return units


def _claim_text(unit: str) -> str:
    """The unit with its own citation markers removed, so file names are not scored as words."""
    text = unit
    for pattern in _CITATION_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def strip_unsupported_citations(text: str, root: Optional[Path] = None) -> Dict[str, Any]:
    """Keep only what the corpus can prove, sentence by sentence.

    ``kept`` lists the citations that survived (``verified``, or ``page_shifted`` rewritten to
    the page that really prints the words). ``removed`` lists citations that failed.
    ``dropped`` lists whole sentences that were deleted, with the reason - an uncited sentence
    that asserts something is deleted too, because a citation on the sentence next to it says
    nothing about it. ``grounded`` is True only when at least one citation survived.
    """
    body = str(text or "")
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    kept_attachment: List[str] = []
    kept_lines: List[str] = []

    for unit in _units(body):
        kind, raw = unit["kind"], unit["text"]
        if kind in ("blank", "structural"):
            kept_lines.append(raw)
            continue

        citations = _find_citations(raw)
        if not citations:
            if is_abstention(raw):
                kept_lines.append(raw)     # an honest "this is not in the sources"
            elif ATTACHMENT_MARKER_RE.search(raw):
                # The user's own attachment: allowed, disclosed, never counted as a manual page.
                kept_attachment.append(raw.strip()[:200])
                kept_lines.append(raw)
            else:
                dropped.append({"reason": "no citation", "text": raw.strip()[:200]})
            continue

        rewritten = raw
        failed = False
        unit_kept: List[str] = []
        unit_removed: List[str] = []
        for item in citations:
            check = verify_citation(item["source"], item["page"], _claim_text(raw), root=root)
            status = check.get("status")
            if status in ("verified", "page_shifted"):
                page = check.get("correct_page") or item["page"]
                kept.append({"source": item["source"], "page": page, "status": status})
                unit_kept.append(item["raw"])
                if status == "page_shifted":
                    rewritten = rewritten.replace(
                        item["raw"], f"[Source: {item['source']}, Page {page}]")
                continue
            failed = True
            reason = ("page not found in the knowledge base"
                      if check.get("page_exists") is False
                      else check.get("reason") or "could not be verified")
            removed.append({"source": item["source"], "page": item["page"], "reason": reason})
            unit_removed.append(reason)

        if failed:
            # One unprovable pointer discredits the sentence it is attached to: the sentence
            # either is not on that page or the page does not say it. Neither is safe to show.
            dropped.append({"reason": unit_removed[0], "text": raw.strip()[:200]})
            continue
        kept_lines.append(rewritten)

    guarded = _tidy("\n".join(kept_lines))
    guarded = _drop_empty_sections(guarded)
    content = [line for line in guarded.split("\n")
               if line.strip() and not _HEADING_RE.match(line) and line.strip() not in {"---", "***"}]

    return {"text": guarded, "kept": kept, "removed": removed, "dropped": dropped,
            "kept_attachment": kept_attachment,
            "kept_content_units": len(content), "grounded": bool(kept), "note": ""}


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


NOTHING_GROUNDED_LINE = (
    "> \U0001f6ab **Not in the verified sources.** No page in the knowledge base supports an "
    "answer to this question, so none is given. A plausible answer without a source is worse "
    "than this line."
)

NOTHING_GROUNDED_SEARCHED_LINE = (
    "> \U0001f6ab **Not in the verified sources.** The knowledge base was searched and holds no "
    "page that supports an answer to this question, so none is given. A plausible answer "
    "without a source is worse than this line."
)


def _footer(result: Dict[str, Any]) -> str:
    """One line stating what was verified and what was thrown away.

    Silence about removed sentences is what let the original defect look like a normal answer:
    four correct bullets and an invented paragraph read the same as four correct bullets.
    """
    kept = len(result.get("kept") or [])
    dropped = len(result.get("dropped") or [])
    attached = len(result.get("kept_attachment") or [])
    if not dropped and not attached:
        return ""
    pieces = []
    if kept:
        pages = len({(item.get("source"), item.get("page")) for item in result.get("kept") or []})
        pieces.append(f"{kept} sentence citation(s) verified against {pages} page(s)")
    if attached:
        pieces.append(f"{attached} line(s) read from your attachment, not from the manuals")
    if dropped:
        pieces.append(f"{dropped} sentence(s) removed: no page in the sources supports them")
    return "> \U0001f512 " + " \u00b7 ".join(pieces) + "."


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
                  "kept": [], "removed": [], "dropped": [], "kept_content_units": 0,
                  "grounded": False, "note": ""}

    footer = _footer(result)

    if result["kept"]:
        if footer:
            result["text"] = f"{result['text']}\n\n{footer}".strip()
        return result

    if is_abstention(result["text"]):
        # The answer refuses to answer; that is already the plainest possible statement
        # that it is not sourced, so no second label is added.
        if footer:
            result["text"] = f"{result['text']}\n\n{footer}".strip()
        return result

    if not corpus_available(root):
        note = UNVERIFIABLE_NOTE
    elif result.get("kept_attachment") and not result["kept"]:
        note = ATTACHMENT_NOTE
    elif retrieved:
        note = OUTSIDE_SOURCE_SEARCHED_NOTE
    else:
        note = OUTSIDE_SOURCE_NOTE

    # Headings alone are not an answer. When everything was stripped, say so once, plainly,
    # instead of showing the skeleton of an answer that no longer exists.
    body = result["text"] if result.get("kept_content_units") else ""
    if not body:
        pieces = [NOTHING_GROUNDED_SEARCHED_LINE if retrieved else NOTHING_GROUNDED_LINE]
    else:
        pieces = [body]
    # The "outside the verified sources" label explains unverified prose that is still shown.
    # When nothing is shown there is no prose to label, and the line above already says it.
    if not corpus_available(root):
        pieces.append(UNVERIFIABLE_NOTE)
    elif body:
        pieces.append(note)
    if footer:
        pieces.append(footer)
    result["text"] = "\n\n".join(piece for piece in pieces if piece).strip()
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
    dropped = len(result.get("dropped") or [])
    if not kept and not removed and not dropped:
        return ""
    if not kept:
        return f"0 citations verified \u00b7 {removed} removed \u00b7 {dropped} sentence(s) dropped"
    if not removed and not dropped:
        return f"{kept} citation(s) verified against the cited page"
    return (f"{kept} citation(s) verified \u00b7 {removed} removed as unprovable "
            f"\u00b7 {dropped} sentence(s) dropped")
