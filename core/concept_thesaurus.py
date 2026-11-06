"""Concept → command thesaurus for beginner-phrased Tessent questions (WP-E).

The WP-8 measurement showed that a question written in user language ("how do I stop the
tool looking inside hard macros") retrieves the right manual page only rarely, while the
manual-worded form retrieves it almost always. The thesaurus closes part of that gap
without inventing knowledge:

* every concept maps only onto commands that exist in the verified command index, and
  onto vocabulary that already occurs in the corpus;
* ``validate()`` drops unknown commands and reports them, so the file cannot drift into
  asserting commands the manual does not document;
* the layer is **opt-in** and additive — the caller decides whether the suggestion is
  prepended, appended or ignored. Nothing here changes what counts as evidence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# WP-1 (Stage B): prefer the measured v2 thesaurus when it is present. The original v1 file
# stays on disk untouched and remains the fallback, exactly like the taxonomy v2 precedent.
_THESAURUS_V2 = PROJECT_ROOT / "knowledge" / "concept_thesaurus_v2.json"
THESAURUS_PATH = _THESAURUS_V2 if _THESAURUS_V2.exists() else PROJECT_ROOT / "knowledge" / "concept_thesaurus.json"

_CACHE: Dict[str, Any] = {}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def load_thesaurus(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and cache the thesaurus (default: knowledge/concept_thesaurus.json)."""
    key = str(path or THESAURUS_PATH)
    if key in _CACHE:
        return _CACHE[key]
    target = Path(key)
    if not target.exists():
        payload: Dict[str, Any] = {"concepts": [], "problems": [f"missing: {target}"], "version": None}
    else:
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload.setdefault("concepts", [])
    _CACHE[key] = payload
    return payload


def validate(path: Optional[Path] = None, command_names: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Check every mapped command against the active index; report unknown entries."""
    payload = load_thesaurus(path)
    if command_names is None:
        from core.command_indexer import load_command_index

        command_names = load_command_index().get("commands", {}).keys()
    known = set(command_names)

    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, str]] = []
    command_free: List[Dict[str, str]] = []
    for concept in payload["concepts"]:
        declared = list(concept.get("commands", []))
        commands = [name for name in declared if name in known]
        for name in declared:
            if name not in known:
                dropped.append({"concept": concept.get("id", "?"), "command": name,
                                "reason": "not in command index"})
        if not declared and concept.get("command_free"):
            # Deliberately concept-only: the manual documents it in prose, so suggesting a
            # command would add a *wrong* dictionary block (measured regression in WP-E).
            command_free.append({"concept": concept.get("id", "?"),
                                 "reason": concept.get("command_free_reason", "")})
            kept.append({**concept, "commands": []})
            continue
        if not commands:
            dropped.append({"concept": concept.get("id", "?"), "command": "",
                            "reason": "no verified command remains"})
            continue
        kept.append({**concept, "commands": commands})
    return {
        "version": payload.get("version"),
        "concepts": kept,
        "dropped": dropped,
        "command_free": command_free,
        "concept_count": len(kept),
        "command_count": len({c for concept in kept for c in concept["commands"]}),
    }


def match_concepts(question: str, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Concepts whose surface forms occur in the question, in order of appearance."""
    text = _normalise(question)
    if not text:
        return []
    matches = []
    for concept in load_thesaurus(path)["concepts"]:
        best: Optional[Tuple[int, str]] = None
        for form in concept.get("surface_forms", []):
            needle = _normalise(form)
            if not needle:
                continue
            position = text.find(needle)
            if position >= 0 and (best is None or position < best[0]):
                best = (position, form)
        if best is not None:
            matches.append({"id": concept.get("id"), "label": concept.get("label"),
                            "matched_form": best[1], "position": best[0],
                            "commands": list(concept.get("commands", [])),
                            "corpus_terms": list(concept.get("corpus_terms", []))})
    return sorted(matches, key=lambda item: item["position"])


def suggest_commands(question: str, path: Optional[Path] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Verified command names implied by the question, plus the concepts that imply them."""
    matches = match_concepts(question, path)
    ordered: List[str] = []
    for match in matches:
        for command in match["commands"]:
            if command not in ordered:
                ordered.append(command)
    return ordered, matches


def augment_question(question: str, path: Optional[Path] = None) -> Tuple[str, List[str]]:
    """Append corpus vocabulary for the matched concepts (additive, never replaces)."""
    _, matches = suggest_commands(question, path)
    extra: List[str] = []
    for match in matches:
        for term in match["corpus_terms"]:
            if term not in extra:
                extra.append(term)
    if not extra:
        return question, []
    return f"{question} {' '.join(extra)}", extra


def describe_matches(matches: Sequence[Dict[str, Any]]) -> str:
    """One-line, citation-free explanation of why extra commands were suggested."""
    if not matches:
        return "No concept matched; retrieval is unchanged."
    return "; ".join(f"'{m['matched_form']}' → {m['label']} ({', '.join(m['commands'])})" for m in matches)
