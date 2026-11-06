"""Query understanding for naturally worded Tessent questions (WP-8).

The layer is deliberately conservative and evidence-neutral:

* The **original** question is always the first query; rewrites are *additive*.
* Synonym/abbreviation expansion only maps domain vocabulary onto terms that already
  appear in the corpus (no new claims, no invented commands).
* Command names are detected against the command index vocabulary, never guessed.
* Multi-query results are merged by agreement, so an unhelpful rewrite can only add
  lower-ranked candidates.

Nothing here changes what counts as evidence: citations still come from the retrieved
manual chunks, and an empty retrieval still produces an explicit abstention.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Domain vocabulary → corpus terms. Keys are normalised (lowercase, spaces collapsed).
SYNONYMS: Dict[str, str] = {
    "memory bist": "mbist memory test",
    "memory built in self test": "mbist memory test",
    "built in self test": "bist mbist",
    "sram test": "memory test mbist",
    "compression": "edt testkompress compression",
    "decompressor": "edt decompressor",
    "compactor": "edt compactor",
    "boundary scan": "jtag boundary scan",
    "tap controller": "jtag tap controller",
    "core wrapper": "wrapper ieee 1500",
    "on chip clock": "occ clock control",
    "at speed": "occ transition path_delay",
    "gate level simulation": "gls simulation sdf",
    "static timing": "sta setup hold timing",
    "design rule check": "design rule checking drc",
    "pattern generation": "pattern generation atpg",
    "test patterns": "test patterns pattern",
    "netlist": "verilog netlist",
    "cell library": "cell library library",
    "scan chain": "scan chain chains",
    "test point": "test point test logic",
    "fault model": "fault type fault model",
    "redundant fault": "redundant faults redundancy",
    "clock skew": "clock skew lockup latch",
}

ABBREVIATIONS: Dict[str, str] = {
    "mbist": "memory built-in self test",
    "edt": "embedded deterministic test compression",
    "occ": "on-chip clock controller",
    "ijtag": "ieee 1687 instrument connectivity",
    "jtag": "ieee 1149.1 boundary scan",
    "bsr": "boundary scan register",
    "tap": "test access port controller",
    "drc": "design rule checking",
    "udfm": "user defined fault model",
    "tsdb": "tessent shell database",
    "gls": "gate level simulation",
    "sta": "static timing analysis",
    "atpg": "automatic test pattern generation",
    "ccd": "clock control definition",
}

_FOLLOW_UP_MARKERS = (
    "it", "this", "that", "they", "those", "these", "its", "their",
    "what about", "how about", "why", "and its", "the second", "the first", "the last",
)

# A query that already contains one of these is specific enough on its own.
_SPECIFIC_MARKERS = ("syntax", "usage", "options", "arguments", "default", "example")

_COMMAND_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b")
_INDEX_VOCABULARY: Optional[frozenset] = None


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def expansion_enabled(environ: Optional[Dict[str, str]] = None) -> bool:
    """Whether vocabulary expansion is active (default OFF).

    Measured on ``evaluation/paraphrase_set_v1`` (16 records):
      baseline  literal 0.875 / paraphrase 0.0  (hit@1)
      expand    literal 0.875 / paraphrase 0.0  -> neutral
    Expansion is therefore opt-in: it changes query strings without a measured gain.
    """
    env = os.environ if environ is None else environ
    return str(env.get("VM_ENABLE_QUERY_EXPANSION", "")).strip().lower() in ("1", "true", "yes", "on")


def build_retrieval_query(
    question: str,
    history: Optional[Iterable[Dict[str, Any]]] = None,
    environ: Optional[Dict[str, str]] = None,
) -> str:
    """Query used for a single retrieval call: referent retention + optional expansion."""
    query = retain_referent(question, history)
    if expansion_enabled(environ):
        query, _expansions = expand_terms(query)
    return query


def _vocabulary() -> frozenset:
    """Command vocabulary from the active index (cached)."""
    global _INDEX_VOCABULARY
    if _INDEX_VOCABULARY is None:
        try:
            from core.command_indexer import load_command_index

            index = load_command_index()
            names = set(index.get("commands", {}))
            names |= set(index.get("alias_map", {}))
            _INDEX_VOCABULARY = frozenset(names)
        except Exception:
            _INDEX_VOCABULARY = frozenset()
    return _INDEX_VOCABULARY


def detect_commands(text: str) -> List[str]:
    """Canonical command names that literally appear in *text* (underscored form)."""
    vocabulary = _vocabulary()
    found: List[str] = []
    for token in _COMMAND_TOKEN_RE.findall(_normalise(text)):
        if token in vocabulary and token not in found:
            found.append(token)
    return found


def is_follow_up(text: str) -> bool:
    """Short or pronoun-led question that needs the previous turn for context."""
    normalised = _normalise(text)
    if not normalised:
        return False
    if len(normalised.split()) <= 5:
        return True
    return any(marker in normalised for marker in _FOLLOW_UP_MARKERS)


def retain_referent(question: str, history: Optional[Iterable[Dict[str, Any]]] = None) -> str:
    """Prefix a follow-up question with the previous user question (context retention)."""
    if not history:
        return question
    previous = ""
    for message in reversed(list(history)):
        if message.get("role") == "user":
            previous = str(message.get("content") or "").split("\n\n[📎 Attached:")[0].strip()
            break
    if not previous or not is_follow_up(question):
        return question
    if _normalise(previous) in _normalise(question):
        return question
    return f"{previous} {question}"


def expand_terms(text: str) -> Tuple[str, List[str]]:
    """Add corpus vocabulary for recognised domain terms. Returns (text, expansions)."""
    normalised = _normalise(text)
    additions: List[str] = []
    for phrase, expansion in SYNONYMS.items():
        if phrase in normalised and expansion not in normalised:
            additions.append(expansion)
    for abbreviation, expansion in ABBREVIATIONS.items():
        if re.search(rf"\b{re.escape(abbreviation)}\b", normalised) and expansion not in normalised:
            additions.append(expansion)
    if not additions:
        return text, []
    return f"{text} {' '.join(additions)}", additions


def rewrite_query(question: str, history: Optional[Iterable[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Structured rewrite record; the original question is always preserved."""
    original = str(question or "").strip()
    resolved = retain_referent(original, history)
    commands = detect_commands(resolved)
    expanded, expansions = expand_terms(resolved)
    return {
        "original": original,
        "referent_resolved": resolved,
        "expanded": expanded,
        "commands": commands,
        "expansions": expansions,
        "is_follow_up": is_follow_up(original),
    }


def multi_queries(question: str, history: Optional[Iterable[Dict[str, Any]]] = None) -> List[str]:
    """At most three ordered queries: original, referent-resolved, expanded."""
    rewritten = rewrite_query(question, history)
    queries = [rewritten["original"]]
    if rewritten["referent_resolved"] and rewritten["referent_resolved"] not in queries:
        queries.append(rewritten["referent_resolved"])
    if rewritten["expanded"] and rewritten["expanded"] not in queries:
        queries.append(rewritten["expanded"])
    # A very specific query (syntax/usage/arguments) does not need vocabulary padding.
    if len(queries) == 3 and any(marker in _normalise(question) for marker in _SPECIFIC_MARKERS):
        queries = queries[:2]
    return queries[:3]


def merge_results(result_lists: List[List[Dict[str, Any]]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Merge retrieval lists by agreement, then by weighted score (deterministic)."""
    merged: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    for rank, results in enumerate(result_lists):
        weight = 1.0 / (rank + 1)
        for position, item in enumerate(results or []):
            key = (str(item.get("source")), item.get("page"))
            entry = merged.get(key)
            if entry is None:
                entry = dict(item)
                entry["_agreement"] = 0
                entry["_weighted"] = 0.0
                entry["_best_rank"] = position
                merged[key] = entry
            entry["_agreement"] += 1
            entry["_weighted"] += weight * float(item.get("score") or 0.0) + weight
            entry["_best_rank"] = min(entry["_best_rank"], position)

    ordered = sorted(
        merged.values(),
        key=lambda e: (-e["_agreement"], -e["_weighted"], e["_best_rank"], str(e.get("source")), e.get("page") or 0),
    )
    out = []
    for entry in ordered[:top_k]:
        clean = {k: v for k, v in entry.items() if not k.startswith("_")}
        clean["score"] = round(entry["_weighted"] / max(1, entry["_agreement"]), 6)
        out.append(clean)
    return out


def retrieve_with_understanding(
    question: str,
    history: Optional[Iterable[Dict[str, Any]]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Retrieve with multi-query expansion and agreement-based merging."""
    from core.rag_builder import retrieve

    queries = multi_queries(question, history)
    if len(queries) == 1:
        return retrieve(queries[0], top_k=top_k)
    lists = []
    for query in queries:
        try:
            lists.append(retrieve(query, top_k=top_k))
        except Exception:
            lists.append([])
    return merge_results(lists, top_k=top_k)


def format_context(results: List[Dict[str, Any]]) -> str:
    """Render retrieved chunks in the format the agents' prompts already expect."""
    if not results:
        return ""
    parts = [
        f"Source Document: {item.get('source')} (Page {item.get('page')})\nContent: {item.get('text')}"
        for item in results
    ]
    return "\n\n---\n\n".join(parts)
