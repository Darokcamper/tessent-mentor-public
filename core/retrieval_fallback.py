"""Low-confidence retrieval fallback (WP-1 Stage B).

When an ordinary-language question matches no concept and the retrieved evidence is weak,
the honest response is still "I do not have this in my sources" — but the user deserves a
way forward. This module produces **suggestions only**:

* closest matching commands (by concept, by command-name token overlap, by alias);
* each suggestion carries the command's verified manual page;
* nothing here is an answer, a definition, or a claim: the UI must label the block as
  suggestions and keep the abstention text authoritative.

Design rules (fixed before implementation):
* never invent a command — every candidate must exist in the command index;
* never fabricate a page — the citation comes from the index's verified evidence;
* never claim the suggestion answers the question;
* keep the existing abstention wording untouched.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

_STOPWORDS = frozenset("""
a an and are as at be by can could do does for from how i if in into is it its me my of on or
please should so that the their them then there these this to tool use used using want was we
what when where which who why will with would you your
""".split())

_WORD_RE = re.compile(r"[a-z0-9_]+")
SUGGESTION_HEADER = "Closest matching commands (suggestions only — not answers)"


def _tokens(text: str) -> List[str]:
    return [t for t in _WORD_RE.findall(str(text or "").lower())
            if len(t) >= 3 and t not in _STOPWORDS]


def _command_tokens(name: str) -> List[str]:
    return [part for part in str(name).lower().split("_") if part]


def suggest_commands_for_question(
    question: str,
    *,
    limit: int = 5,
    concept_commands: Optional[Iterable[str]] = None,
    command_names: Optional[Iterable[str]] = None,
    index: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Candidate commands for an unmatched question, each with its verified page.

    Ranking (deterministic, no LLM):
      1. commands implied by a matched concept          (weight 3 per concept command)
      2. commands whose name tokens appear in the question (weight 2 per token)
      3. commands whose purpose text shares rare question tokens (weight 1)
    """
    if index is None:
        from core.command_indexer import load_command_index

        index = load_command_index().get("commands", {})
    names = list(command_names) if command_names is not None else list(index)
    question_tokens = set(_tokens(question))
    if not question_tokens and not concept_commands:
        return []

    scores: Dict[str, float] = {}
    reasons: Dict[str, List[str]] = {}

    for command in concept_commands or []:
        if command in index:
            scores[command] = scores.get(command, 0) + 3
            reasons.setdefault(command, []).append("related concept in your question")

    for name in names:
        if name not in index:
            continue
        parts = set(_command_tokens(name))
        overlap = parts & question_tokens
        if overlap:
            scores[name] = scores.get(name, 0) + 2 * len(overlap)
            reasons.setdefault(name, []).append("name matches: " + ", ".join(sorted(overlap)))

    for name, entry in index.items():
        if name in scores and scores[name] >= 3:
            continue
        purpose = str(entry.get("purpose") or "")
        purpose_tokens = set(_tokens(purpose))
        shared = question_tokens & purpose_tokens
        if len(shared) >= 2:
            scores[name] = scores.get(name, 0) + len(shared)
            reasons.setdefault(name, []).append("purpose mentions: " + ", ".join(sorted(shared)[:4]))

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    suggestions = []
    for name, score in ranked:
        evidence = (index.get(name) or {}).get("source_evidence") or {}
        suggestions.append({
            "command": name,
            "reason": "; ".join(reasons.get(name, [])[:2]),
            "score": round(score, 2),
            "page": evidence.get("page"),
            "file": evidence.get("file"),
            "citation": (f"[Source: {evidence.get('file')}, Page {evidence.get('page')}]"
                         if evidence.get("file") and evidence.get("page") else None),
            "purpose": str((index.get(name) or {}).get("purpose") or "")[:220] or None,
            "is_answer": False,
        })
    return suggestions


def is_low_confidence(question: str, evidence: Optional[Iterable[Any]] = None) -> bool:
    """True when there is nothing to build an answer on apart from generic chunks."""
    from core.concept_thesaurus import match_concepts

    try:
        if match_concepts(question):
            return False
    except Exception:
        pass
    chunks = list(evidence or [])
    if not chunks:
        return True
    best = 0.0
    for chunk in chunks:
        try:
            best = max(best, float(chunk.get("score") or 0.0))
        except Exception:
            continue
    return best < 0.62


def render_suggestions_markdown(suggestions: List[Dict[str, Any]]) -> Optional[str]:
    """Markdown block for the UI. Explicitly not an answer."""
    if not suggestions:
        return None
    lines = [
        f"**{SUGGESTION_HEADER}**",
        "",
        "Your question did not match the manual wording closely enough for a grounded answer. "
        "These commands look related — check them in the Commands Explorer, or rephrase with "
        "the exact term you need.",
        "",
    ]
    for item in suggestions:
        citation = f" — {item['citation']}" if item.get("citation") else " — no page recorded"
        lines.append(f"- `{item['command']}`{citation}")
        if item.get("reason"):
            lines.append(f"  - why: {item['reason']}")
    lines += ["", "_Suggestions are pointers, not answers. No claim is made about them._"]
    return "\n".join(lines)
