"""Correlate every kind of source in the knowledge base, not just manual pages.

The corpus is not "the manuals plus some other stuff". It holds, for the same topic, several
independent expressions of it:

    atpg_video_slides__1_dft_basic_concepts.txt   what was on screen in the video (a frame)
    atpg_video_subs__1_dft_basic_concepts.txt     what the trainer said about it
    atpg_gd.txt / tshell_ref (1).txt              the exact command reference, with page numbers
    Tessent_ATPG_Core_Topics_lab_wkbk_..._s.txt   the exercise that practises it
    atpg_video_quiz__9_level_1_exam_...txt        how the course examines it

This module names those kinds, groups files that cover the same topic, and finds the pages in
each kind that talk about one slide - so an explanation can combine all of them instead of
treating a manual page as the only authority and everything else as unverified.

Two lookups matter for a deck:

* :func:`find_pages` - which pages, in which kind of source, cover this slide's wording;
* :func:`source_kind` / :func:`source_label` - how to name a citation honestly, so a transcript
  quote is never passed off as a manual page and a manual page is never called unverified.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.citation_check import PAGE_MARKER_RE, corpus_dir, read_page

_VIDEO_SLIDE_RE = re.compile(r"video_slides?__\d+_(?P<topic>.+)$", re.IGNORECASE)
_VIDEO_SUBS_RE = re.compile(r"video_subs__\d+_(?P<topic>.+)$", re.IGNORECASE)
_VIDEO_QUIZ_RE = re.compile(r"video_quiz__\d+", re.IGNORECASE)
_LAB_RE = re.compile(r"lab_wkb|lab_wkbk|lab_manual|workbook", re.IGNORECASE)
_MANUAL_RE = re.compile(r"(_ref|_user|_gd\b|user_manual|manual)", re.IGNORECASE)

KIND_LABELS = {
    "manual": "manual page",
    "video_slide": "video slide (frame on screen)",
    "video_transcript": "video transcript (spoken)",
    "lab_workbook": "lab workbook page",
    "course_exam": "course exam page",
    "reference": "knowledge-base page",
}

KIND_ROLES = {
    "manual": "exact commands, options, arguments and definitions",
    "video_slide": "what the course actually showed on screen",
    "video_transcript": "the trainer's spoken explanation and the reasoning behind it",
    "lab_workbook": "the hands-on exercise steps and expected results",
    "course_exam": "how the topic is examined, i.e. what is graded",
    "reference": "supporting reference text",
}

_WORD_RE = re.compile(r"[a-z0-9_]+")
_HEAD_CHARS = 1400          # normalized characters kept per page for heading matching
_MIN_PAGE_SCORE = 4         # distinctive terms that must appear on a page to link it
_MIN_RELATIVE = 0.25        # ...and that is also a quarter of the slide's own terms

_STOPWORDS = frozenset({
    "about", "after", "again", "also", "although", "another", "because", "before", "being",
    "between", "both", "cannot", "could", "does", "doing", "each", "either", "every", "from",
    "have", "having", "into", "more", "most", "must", "only", "other", "over", "same", "should",
    "some", "such", "than", "that", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "under", "until", "very", "were", "what", "when", "where", "which",
    "while", "will", "with", "within", "without", "would", "your", "slide", "slides", "video",
    "tessent",
})

# Module topics contain the flow; the order is not significant for matching.
_TOPIC_WORDS = re.compile(r"[a-z0-9]+")


def source_kind(filename: str) -> str:
    """Which kind of source a corpus file is. Never raises."""
    name = str(filename or "")
    if _VIDEO_SLIDE_RE.search(Path(name).stem):
        return "video_slide"
    if _VIDEO_SUBS_RE.search(Path(name).stem):
        return "video_transcript"
    if _VIDEO_QUIZ_RE.search(name):
        return "course_exam"
    if _LAB_RE.search(name):
        return "lab_workbook"
    if _MANUAL_RE.search(name):
        return "manual"
    return "reference"


def source_label(filename: str) -> str:
    """Human label for a citation, e.g. ``video transcript (spoken)``."""
    return KIND_LABELS.get(source_kind(filename), KIND_LABELS["reference"])


def source_role(filename: str) -> str:
    """What this kind of source is good for, for the synthesis prompt."""
    return KIND_ROLES.get(source_kind(filename), KIND_ROLES["reference"])


def module_topic(filename: str) -> str:
    """Module slug of a course file, e.g. ``dft_basic_concepts``; empty when not a module."""
    stem = Path(str(filename or "")).stem
    for pattern in (_VIDEO_SLIDE_RE, _VIDEO_SUBS_RE):
        match = pattern.search(stem)
        if match:
            return match.group("topic").strip().lower()
    return ""


def corpus_files(root: Optional[Path] = None) -> List[str]:
    """Names of the corpus text files on this host (empty when the bundle is absent)."""
    try:
        return sorted(path.name for path in corpus_dir(root).glob("*.txt"))
    except OSError:
        return []


def topics_of(names: Iterable[str]) -> Dict[str, List[str]]:
    """Group corpus file names by module topic."""
    grouped: Dict[str, List[str]] = {}
    for name in names:
        topic = module_topic(name)
        if topic:
            grouped.setdefault(topic, []).append(name)
    return grouped


def terms_of(text: str, limit: int = 24) -> List[str]:
    """Distinctive lowercase terms of a piece of text (what a page must share)."""
    seen: List[str] = []
    for token in _WORD_RE.findall(str(text or "").lower()):
        if len(token) < 5 or token in _STOPWORDS or token.isdigit():
            continue
        if token not in seen:
            seen.append(token)
    return seen[:limit]


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


# Page index: {file name: [(page number, normalised head of the page)]}. Built once per
# process because the corpus is ~18 MB and a deck run asks the same question 100+ times.
_INDEX_CACHE: Dict[str, Any] = {"key": None, "index": {}}


def _cache_key(directory: Path) -> Tuple[str, int, float]:
    try:
        files = list(directory.glob("*.txt"))
        newest = max((path.stat().st_mtime for path in files), default=0.0)
        return (str(directory), len(files), round(newest, 3))
    except OSError:
        return (str(directory), 0, 0.0)


def page_index(root: Optional[Path] = None,
               force: bool = False) -> Dict[str, List[Tuple[int, str]]]:
    """Normalised head of every page of every corpus file, cached per process."""
    directory = corpus_dir(root)
    key = _cache_key(directory)
    if not force and _INDEX_CACHE["key"] == key:
        return _INDEX_CACHE["index"]

    index: Dict[str, List[Tuple[int, str]]] = {}
    for name in corpus_files(root):
        try:
            raw = (directory / name).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        markers = list(PAGE_MARKER_RE.finditer(raw))
        pages: List[Tuple[int, str]] = []
        for position, marker in enumerate(markers):
            start = marker.end()
            end = markers[position + 1].start() if position + 1 < len(markers) else len(raw)
            pages.append((int(marker.group(1)), _normalise(raw[start:end])[:_HEAD_CHARS]))
        index[name] = pages
    _INDEX_CACHE["key"] = key
    _INDEX_CACHE["index"] = index
    return index


def matching_topics(text: str, root: Optional[Path] = None, limit: int = 2) -> List[str]:
    """Module topics whose name best matches the text."""
    terms = set(terms_of(text, limit=40))
    scored: List[Tuple[int, str]] = []
    for topic in topics_of(corpus_files(root)):
        words = {word for word in _TOPIC_WORDS.findall(topic) if len(word) >= 4}
        if not words:
            continue
        score = len(words & terms)
        if score:
            scored.append((score, topic))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [topic for _score, topic in scored[:limit]]


def find_pages(text: str, root: Optional[Path] = None, kinds: Optional[Iterable[str]] = None,
               names: Optional[Iterable[str]] = None, limit_per_kind: int = 1,
               limit: int = 4) -> List[Dict[str, Any]]:
    """Pages that cover *text*, chosen across kinds rather than by raw similarity.

    The point is coverage of *evidence types*: for one slide this usually returns the manual
    page that states the rule, the video slide that showed it, the transcript that explained
    it and the lab page that practises it - which is what "combine all sources" means.
    """
    terms = terms_of(text)
    if len(terms) < 2:
        return []
    # Two conditions keep incidental word matches out: an absolute number of shared terms and
    # a share of the slide's own vocabulary. Off-corpus material (a deck about quantum
    # annealing) shares only stray words and must not come back with "sources".
    threshold = max(_MIN_PAGE_SCORE, int(round(len(terms) * _MIN_RELATIVE)))
    wanted = {str(kind) for kind in kinds} if kinds else None
    candidates = {str(name) for name in names} if names else None
    index = page_index(root)

    def _scan(names_to_scan, kinds_to_scan):
        found: Dict[str, List[Dict[str, Any]]] = {}
        for name in names_to_scan:
            kind = source_kind(name)
            if kinds_to_scan is not None and kind not in kinds_to_scan:
                continue
            for page, head in index.get(name, []):
                score = sum(1 for term in terms if term in head)
                if score < threshold:
                    continue
                found.setdefault(kind, []).append(
                    {"source": name, "page": page, "kind": kind, "score": score})
        return found

    scoped_kinds = {"video_slide", "video_transcript"}
    # 1. Choose the video module first: the frame that shows this subject decides which
    #    transcript belongs with it, so "At-Speed / OCC" never borrows a patterns transcript.
    video_names = [name for name in index if source_kind(name) == "video_slide"
                   and (candidates is None or name in candidates)]
    video_hits: List[Dict[str, Any]] = []
    if wanted is None or "video_slide" in wanted:
        video_hits = _scan(video_names, {"video_slide"}).get("video_slide", [])
    module = ""
    if video_hits:
        best_video = max(video_hits, key=lambda item: (item["score"], item["source"]))
        module = module_topic(best_video["source"])

    scan_names = []
    for name in index:
        if candidates is not None and name not in candidates:
            continue
        kind = source_kind(name)
        if wanted is not None and kind not in wanted:
            continue
        if module and kind in scoped_kinds and module_topic(name) != module:
            continue
        scan_names.append(name)

    best_by_kind = _scan(scan_names, wanted)
    if module and "video_transcript" in best_by_kind and not best_by_kind["video_transcript"]:
        del best_by_kind["video_transcript"]

    results: List[Dict[str, Any]] = []
    for kind in sorted(best_by_kind):
        ranked = sorted(best_by_kind[kind],
                        key=lambda item: (-item["score"], item["source"], item["page"]))
        results.extend(ranked[:limit_per_kind])
    results.sort(key=lambda item: -item["score"])
    return results[:limit]


def evidence_from_pages(pages: Iterable[Dict[str, Any]], root: Optional[Path] = None,
                        origin: str = "correlated") -> List[Dict[str, Any]]:
    """Turn correlated page references into evidence entries the explainer can cite."""
    out: List[Dict[str, Any]] = []
    for item in pages:
        page_text = read_page(item["source"], item["page"], root=root)
        if not page_text or not page_text.strip():
            continue
        out.append({
            "source": item["source"],
            "page": item["page"],
            "text": page_text.strip(),
            "citation": f"[Source: {item['source']}, Page {item['page']}]",
            "score": item.get("score"),
            "kind": item.get("kind") or source_kind(item["source"]),
            "label": source_label(item["source"]),
            "role": source_role(item["source"]),
            "origin": origin,
        })
    return out


def kind_summary(evidence: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """How many independent kinds of source back one slide."""
    counts: Dict[str, int] = {}
    for item in evidence:
        kind = item.get("kind") or source_kind(item.get("source") or "")
        counts[kind] = counts.get(kind, 0) + 1
    present = [KIND_LABELS.get(kind, kind) for kind in sorted(counts)]
    return {
        "counts": counts,
        "kinds": sorted(counts),
        "labels": present,
        "independent_kinds": len(counts),
        "line": (f"{len(counts)} kind(s) of source: " + ", ".join(
            f"{KIND_LABELS.get(kind, kind)} x{counts[kind]}" for kind in sorted(counts))
            if counts else "no sources"),
    }

