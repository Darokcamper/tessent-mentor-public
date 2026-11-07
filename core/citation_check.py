"""Verification of ``[Source: <file>, Page <N>]`` citations against the corpus text.

A generated explanation may only cite pages the knowledge base actually contains. Even
when retrieval is honest, the model can attach the wrong page number to a sentence (in
production review a quoted bullet list sitting on page 34 was cited as page 33). Page-level
traceability is the product's main claim, so every citation is checked against the cached
page text before it is shown:

* ``verified``     - the page exists and carries the words the sentence is built on;
* ``page_shifted`` - the cited page does not, but a neighbouring page does (off-by-N), so
                     the citation is re-pointed at the page that really holds the text;
* ``unverified``   - the page does not exist, or nothing on any nearby page matches. The
                     citation is left alone but reported, never silently trusted.

The check is lexical and explainable on purpose: it is a pointer check ("does this page
contain these words"), not a claim of semantic entailment, and it never rewrites the
model's technical statements - only the page number of a proven mismatch.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIRNAME = "txt"

PAGE_MARKER_RE = re.compile(r"--- PAGE (\d+) ---")
CITATION_RE = re.compile(
    r"\[Source:\s*(?P<source>[^\[\],]+?)\s*,\s*Page\s*(?P<page>\d{1,5})\s*\]",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z0-9_]+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# A citation is "supported" when enough of the sentence's distinctive words are actually
# printed on the page. Two independent conditions keep single common words from passing.
MIN_OVERLAP = 0.34
MIN_MATCHED_TERMS = 2
MAX_CLAIM_CHARS = 700
MAX_SHIFT = 3                     # how far to look for the page that really holds the text
SHIFT_GAIN = 0.2                  # a shift must be clearly better than the cited page

_STOPWORDS = frozenset({
    "about", "above", "after", "again", "against", "allow", "allows", "also", "although",
    "always", "another", "because", "before", "being", "below", "between", "both", "cannot",
    "could", "described", "does", "doing", "each", "either", "every", "explain", "explained",
    "follows", "from", "further", "given", "have", "having", "here", "into", "itself",
    "manual", "many", "more", "most", "must", "only", "other", "over", "page", "same",
    "should", "show", "shows", "slide", "some", "source", "such", "than", "that", "their",
    "them", "then", "there", "these", "they", "this", "those", "through", "under", "until",
    "very", "were", "what", "when", "where", "whether", "which", "while", "will", "with",
    "within", "without", "would", "your",
})


def corpus_dir(root: Optional[Path] = None) -> Path:
    """Folder holding the cached corpus text (``knowledge/txt`` by default)."""
    return (Path(root) if root is not None else PROJECT_ROOT) / "knowledge" / CORPUS_DIRNAME


def corpus_available(root: Optional[Path] = None) -> bool:
    """False when the knowledge bundle is not restored on this host.

    Without the corpus nothing can be verified, and that is a different statement from "the
    page is missing". Reporting the two the same way would accuse honest citations of being
    fabricated on a cold container.
    """
    try:
        return corpus_dir(root).is_dir()
    except OSError:
        return False


def _normalise(text: str) -> str:
    """Lowercase and drop everything but letters and digits.

    The corpus text comes from PDF extraction, so words are frequently run together
    ("Individualports maybe slowduring0->1transition"). Removing separators from both
    sides makes a term search work on such text.
    """
    return _NON_ALNUM_RE.sub("", str(text or "").lower())


def claim_terms(claim: str, limit: int = 25) -> List[str]:
    """Distinctive lowercase terms of one claim, longest first."""
    seen: List[str] = []
    for token in _WORD_RE.findall(str(claim or "").lower()):
        if len(token) < 5 or token in _STOPWORDS or token.isdigit():
            continue
        if token not in seen:
            seen.append(token)
    seen.sort(key=len, reverse=True)
    return seen[:limit]


def _overlap(terms: List[str], haystack: str) -> Tuple[float, List[str]]:
    if not terms:
        return 0.0, []
    matched = [term for term in terms if term in haystack]
    return len(matched) / len(terms), matched


def read_page(source: str, page: int, root: Optional[Path] = None) -> Optional[str]:
    """Text of one page of one corpus file, or ``None`` when it is not there."""
    if not source:
        return None
    path = corpus_dir(root) / Path(str(source)).name
    try:
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    markers = list(PAGE_MARKER_RE.finditer(raw))
    if not markers:
        return None
    for position, marker in enumerate(markers):
        if int(marker.group(1)) != int(page):
            continue
        start = marker.end()
        end = markers[position + 1].start() if position + 1 < len(markers) else len(raw)
        return raw[start:end]
    return None


def claim_span(text: str, end: int) -> str:
    """The sentence or bullet that a citation at offset *end* is attached to.

    Several boundaries are tried, longest surviving span first, and one is only accepted
    when what remains still carries at least two distinctive terms. A plain "cut at the
    last full stop" can leave nothing at all when the sentence itself ends in a period.
    """
    head = str(text or "")[:max(0, end)][-MAX_CLAIM_CHARS:]
    boundaries = {head.rfind(sep) for sep in ("\n", ". ", "; ", ": ", "* ")}
    for cut in sorted((b for b in boundaries if b >= 0), reverse=True):
        candidate = head[cut + 1:].strip()
        if len(claim_terms(candidate)) >= 2:
            return candidate
    return head.strip()


def verify_citation(source: str, page: int, claim: str,
                    root: Optional[Path] = None) -> Dict[str, Any]:
    """Check one citation against the corpus. Never raises."""
    source = str(source or "").strip()
    try:
        page = int(page)
    except (TypeError, ValueError):
        return {"source": source, "page": page, "status": "unverified",
                "reason": "page number is not an integer", "matched": [], "overlap": 0.0,
                "correct_page": None, "terms": 0}

    terms = claim_terms(claim)
    if not corpus_available(root):
        return {"source": source, "page": page, "status": "unverified",
                "reason": "the knowledge corpus is not available on this host", "matched": [],
                "overlap": 0.0, "correct_page": None, "terms": len(terms),
                "page_exists": None}
    page_text = read_page(source, page, root=root)
    if page_text is None:
        return {"source": source, "page": page, "status": "unverified",
                "reason": "page not found in the corpus file", "matched": [],
                "overlap": 0.0, "correct_page": None, "terms": len(terms),
                "page_exists": False}

    overlap, matched = _overlap(terms, _normalise(page_text))
    best = {"page": page, "overlap": overlap, "matched": matched}

    if overlap >= MIN_OVERLAP and len(matched) >= MIN_MATCHED_TERMS:
        return {"source": source, "page": page, "status": "verified", "reason": "",
                "matched": matched, "overlap": round(overlap, 3), "correct_page": page,
                "terms": len(terms), "page_exists": True}

    # Off-by-N: look nearby for the page that actually prints this sentence.
    for distance in range(1, MAX_SHIFT + 1):
        for candidate in (page + distance, page - distance):
            if candidate < 1:
                continue
            neighbour = read_page(source, candidate, root=root)
            if neighbour is None:
                continue
            n_overlap, n_matched = _overlap(terms, _normalise(neighbour))
            if n_overlap > best["overlap"]:
                best = {"page": candidate, "overlap": n_overlap, "matched": n_matched}

    if (best["page"] != page and best["overlap"] >= MIN_OVERLAP
            and len(best["matched"]) >= MIN_MATCHED_TERMS
            and best["overlap"] - overlap >= SHIFT_GAIN):
        return {"source": source, "page": page, "status": "page_shifted",
                "reason": f"the quoted text is on page {best['page']}",
                "matched": best["matched"], "overlap": round(overlap, 3),
                "correct_page": best["page"], "correct_overlap": round(best["overlap"], 3),
                "terms": len(terms), "page_exists": True}

    return {"source": source, "page": page, "status": "unverified",
            "reason": ("the page exists but does not carry the quoted wording"
                       if terms else "no distinctive wording to check"),
            "matched": matched, "overlap": round(overlap, 3), "correct_page": None,
            "terms": len(terms), "page_exists": True}


def verify_text(text: str, root: Optional[Path] = None) -> Dict[str, Any]:
    """Verify every citation in *text*; returns a report plus page-corrected text."""
    body = str(text or "")
    checks: List[Dict[str, Any]] = []
    corrected = body
    matches = list(CITATION_RE.finditer(body))
    # Walk matches in reverse so earlier offsets stay valid while rewriting.
    for match in reversed(matches):
        source = match.group("source").strip()
        page = int(match.group("page"))
        check = verify_citation(source, page, claim_span(body, match.start()), root=root)
        check["citation"] = body[match.start():match.end()]
        checks.append(check)
        if check["status"] == "page_shifted" and check.get("correct_page"):
            replacement = f"[Source: {source}, Page {check['correct_page']}]"
            corrected = corrected[:match.start()] + replacement + corrected[match.end():]

    checks.reverse()
    counts = {"verified": 0, "page_shifted": 0, "unverified": 0}
    for check in checks:
        counts[check["status"]] = counts.get(check["status"], 0) + 1

    distinct = {(c["source"], c["page"]) for c in checks}
    missing = [c for c in checks if c.get("page_exists") is False]
    return {
        "citations": _dedupe(checks),
        "counts": counts,
        "total": len(checks),
        "distinct_pages": len(distinct),
        "pages_missing": len(missing),
        "corpus_available": corpus_available(root),
        "corrected_text": corrected,
        "changed": corrected != body,
        "all_verified": bool(checks) and counts["unverified"] == 0 and counts["page_shifted"] == 0,
    }


def status_label(check: Dict[str, Any]) -> str:
    """Plain-English status for a handout appendix row."""
    status = str(check.get("status") or "")
    if status == "verified":
        return "verified against the page"
    if status == "page_shifted":
        return f"page corrected to {check.get('correct_page')}"
    if check.get("page_exists") is None:
        return "not checked - knowledge corpus unavailable"
    if not check.get("page_exists", True):
        return "PAGE NOT FOUND - do not rely on this"
    return "page exists, wording not matched"


def summary_line(report: Dict[str, Any]) -> str:
    """One-line, human-readable outcome for a UI caption or a handout footer."""
    total = int(report.get("total") or 0)
    if not total:
        return "No citations to check on this slide."
    if report.get("corpus_available") is False:
        return (f"{total} citation(s) not checked: the knowledge corpus is not restored on "
                "this host, so page numbers cannot be confirmed.")
    counts = report.get("counts") or {}
    verified = int(counts.get("verified") or 0)
    shifted = int(counts.get("page_shifted") or 0)
    unverified = int(counts.get("unverified") or 0)
    if not shifted and not unverified:
        return f"All {total} citation(s) verified against the cited page."
    parts = [f"{verified} of {total} citation(s) verified against the cited page"]
    if shifted:
        parts.append(f"{shifted} page number(s) corrected to the page that holds the text")
    if unverified:
        parts.append(f"{unverified} could not be confirmed and must not be relied on")
    return "; ".join(parts) + "."


def _dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop repeats of the same (source, page, status) triple, keeping order."""
    seen = set()
    out = []
    for item in items:
        key = (item.get("source"), item.get("page"), item.get("status"))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
