"""Ranking helpers that need no ML runtime, so they can be tested without loading torch.

`core.rag_builder` imports sentence-transformers and faiss at module import, which costs ~30 s;
keeping pure ranking logic here means its unit tests run in milliseconds.
"""
from __future__ import annotations

import math
import os
import re

# On by default. Measured on this corpus: one file took 9 of 10 retrieval slots for
# "ATPG difficult interview questions" and pushed the folder that answers it to rank 8;
# a single manual (tshell_ref) is 47% of the 30,788 chunks. Set VM_RAG_MAX_PER_SOURCE=0 to
# get the old winner-takes-all behaviour back.
DEFAULT_MAX_PER_SOURCE = 2

# How hard a rare query term pulls its own pages up the ranking. Measured: at 0 the query
# "flatmodel" did not return Flatmodel.txt at all, although that file contains the word nine
# times and only three other files mention it once each.
DEFAULT_LEXICAL_WEIGHT = 0.45

# A page that *defines* a rare term can score far below an unrelated transcript: measured
# cosine for "flatmodel" was 0.31 on the note that defines it and 0.85 on a lesson transcript
# that never prints the word. With the corpus's +0.25/+0.15 source boosts on top, any ceiling
# below ~1.0 mathematically cannot promote the defining page, so the boost tops out at 1.5.
MAX_LEXICAL_BOOST = 1.5

# A term found in at most this many chunks is precise enough that those chunks must be
# retrieved even when cosine similarity misses them completely (measured: Flatmodel.txt sits
# at rank 101, outside the 64-chunk candidate pool, so a reranking boost alone could never
# reach it). Terms in hundreds of chunks are left to the semantic ranking.
RARE_DF_MAX = 5

TOKEN_RE = re.compile(r"[a-z][a-z0-9_]{2,}")
STOPWORDS = frozenset("""
the and for with what how does that this from into are can you your when which while about
have has not but all its use using used one two any may also more than then them they there
their would could should will shall was were been being our out over such only some very
""".split())

# token -> (document frequency, chunk positions), filled by the first query that asks for the
# token. One process serves one corpus, so this is safe; tools and tests pass their own cache
# (see rare_term_scan) rather than clearing this one.
_SCAN_CACHE: dict = {}


def query_tokens(query: str) -> list:
    """Distinct meaningful tokens, in order. Short words and stopwords carry no signal."""
    seen, tokens = set(), []
    for token in TOKEN_RE.findall(str(query or "").lower()):
        if token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def rare_term_scan(token: str, texts, max_df: int = RARE_DF_MAX, cache: dict = None) -> tuple:
    """``(document frequency, chunk positions)`` for *token*, in one pass over the corpus.

    Positions are only collected while the term still looks rare, so a common word costs a
    count and no memory. Matching is case-insensitive: the note that defines "flatmodel"
    starts the sentence with a capital F.

    *cache* keys on the token alone, which assumes one corpus per process - true for the app.
    Tests and tools pass a fresh dict so their fixtures cannot leak into each other.
    """
    cache = _SCAN_CACHE if cache is None else cache
    cached = cache.get(token)
    if cached is not None:
        return cached
    count, positions = 0, []
    for position, text in enumerate(texts):
        if token in str(text).lower():
            count += 1
            if count <= max_df:
                positions.append(position)
    result = (count, tuple(positions) if count <= max_df else ())
    cache[token] = result
    return result


def document_frequency(token: str, texts, cache: dict = None) -> int:
    """How many chunks contain *token*."""
    return rare_term_scan(token, texts, cache=cache)[0]


def rare_term_hits(tokens, texts, max_df: int = RARE_DF_MAX, cache: dict = None) -> dict:
    """token -> chunk positions, for every token that appears in at most *max_df* chunks."""
    texts = list(texts)
    hits = {}
    for token in tokens:
        count, positions = rare_term_scan(token, texts, max_df=max_df, cache=cache)
        if positions:
            hits[token] = list(positions)
    return hits


def rarity_weights(tokens, texts, total_chunks: int = 0, cache: dict = None) -> dict:
    """Map token -> 0..1 rarity, where 1 means it appears in almost no chunk.

    Uses a smoothed inverse document frequency normalised by the corpus size, so a term in
    4 of 30,788 chunks scores near 1.0 and a term in thousands scores near 0. Common technical
    words therefore cannot drown a precise one.
    """
    texts = list(texts)
    total = total_chunks or len(texts) or 1
    ceiling = math.log(1 + total)
    weights = {}
    for token in tokens:
        df = document_frequency(token, texts, cache=cache)
        weights[token] = round(math.log(1 + total / (1 + df)) / ceiling, 4) if ceiling else 0.0
    return weights


def lexical_boost(tokens, weights: dict, chunk_text: str,
                  weight: float = DEFAULT_LEXICAL_WEIGHT,
                  ceiling: float = MAX_LEXICAL_BOOST) -> float:
    """Reward chunks that contain the query's rare terms, more so when they repeat them.

    A page that prints "flatmodel" nine times is *about* flatmodel; a slide that mentions it
    once in passing is not. The repeat factor (``1 + log(tf)``) is what separates the two
    without letting a long transcript win merely by length.
    """
    if not tokens or weight <= 0:
        return 0.0
    haystack = str(chunk_text or "").lower()
    score = 0.0
    for token in tokens:
        occurrences = haystack.count(token)
        if occurrences:
            score += weights.get(token, 0.0) * (1.0 + math.log(occurrences))
    return min(ceiling, score * weight)


def lexical_weight_from_env(default: float = DEFAULT_LEXICAL_WEIGHT) -> float:
    raw = os.environ.get("VM_RAG_LEXICAL_WEIGHT", "")
    if not raw.strip():
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def max_per_source_from_env(default: int = DEFAULT_MAX_PER_SOURCE) -> int:
    """Read the per-source cap, falling back to the default on junk rather than crashing."""
    raw = os.environ.get("VM_RAG_MAX_PER_SOURCE", "")
    if not raw.strip():
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def diversify(ranked: list, top_k: int, cap: int) -> list:
    """Limit how many chunks one source may contribute, backfilling from the rest.

    A single 5,000-page manual holds 47% of this corpus's 30,788 chunks, so without a cap one
    file can fill every retrieval slot and hide the other 236 sources. Measured before this
    existed: for "ATPG difficult interview questions", one file took 7 of the top 12 slots and
    the folder that answers the question was pushed to rank 8.

    Order within the result still follows score; the cap only decides which chunks get a slot.
    """
    if cap <= 0:
        return ranked[:top_k]
    picked, overflow, seen = [], [], {}
    for score, chunk in ranked:
        source = chunk.get("source")
        used = seen.get(source, 0)
        if used < cap:
            seen[source] = used + 1
            picked.append((score, chunk))
        else:
            overflow.append((score, chunk))
        if len(picked) == top_k:
            break
    if len(picked) < top_k:
        picked.extend(overflow[:top_k - len(picked)])
    return picked
