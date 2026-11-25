"""Wiring test: a rare query term must reach retrieve() even when cosine ranks it far down.

Guards the measured production defect end to end without loading the real model or index:
the note that defines "flatmodel" scores 0.3121 cosine and sits at rank 101 of 30,788, i.e.
outside the 64-chunk candidate pool. Two things have to be true for it to be returned -
its chunks must be added to the pool, and the rerank must be allowed to outrank a merely
similar transcript. Both are asserted separately, so a future change that keeps one and
drops the other fails here.

The fakes are shaped like the real objects: ``index.search`` returns the top-k cosines,
``index.reconstruct`` returns a stored vector, and vectors are L2-normalised at build time.
Every call here also proves the general path does not depend on an import that only the
``source_filter`` branch performs (that was an UnboundLocalError for ``np``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.rag_builder as rb

RARE_SOURCE = "Flatmodel.txt"
# Nine occurrences, exactly like knowledge/txt/Flatmodel.txt.
NOTE_TEXT = ("How to open the failing instance without rerunning ATPG --> using flatmodel. "
             "Flatmodel: design data and settings are stored. read_flat_model flatmodel. "
             "write_flat_model flatmodel -replace. " + "flatmodel " * 5)
TRANSCRIPT_TEXT = ("the atpg flow and the scan chain are explained for the lab session in "
                   "this long lesson transcript, with the clock and the pattern setup")

NOTE_INDEX = 40          # the note sits far outside the 8-chunk pool the fake returns
POOL_SIZE = 8
TOTAL_CHUNKS = 48


class FakeModel:
    def encode(self, texts):
        return np.array([[1.0, 0.0, 0.0]], dtype="float32")


class FakeIndex:
    """Eight transcript chunks at high cosine; the note is only reachable by reconstruction."""

    def search(self, query_vector, k):
        indices = [i for i in range(POOL_SIZE) if i != NOTE_INDEX][:min(k, POOL_SIZE)]
        scores = [0.85 - 0.01 * rank for rank in range(len(indices))]
        return (np.array([scores], dtype="float32"), np.array([indices]))

    def reconstruct(self, idx):
        if idx == NOTE_INDEX:
            return np.array([0.3121, 0.9, 0.9], dtype="float32")
        return np.array([0.85, 0.1, 0.1], dtype="float32")


def _metadata():
    rows = [{"source": f"atpg_video_subs__{rank}_lesson.txt", "page": rank + 1,
             "text": TRANSCRIPT_TEXT}
            for rank in range(TOTAL_CHUNKS)]
    rows[NOTE_INDEX] = {"source": RARE_SOURCE, "page": 1, "text": NOTE_TEXT}
    return rows


@pytest.fixture()
def wired(monkeypatch):
    """Point retrieve() at the fakes; monkeypatch restores every attribute afterwards."""
    monkeypatch.setattr(rb, "load_rag", lambda: True)
    monkeypatch.setattr(rb, "_thesaurus_enabled", lambda: False)
    monkeypatch.setattr(rb, "_loaded_model", FakeModel())
    monkeypatch.setattr(rb, "_loaded_index", FakeIndex())
    monkeypatch.setattr(rb, "_loaded_metadata", _metadata())
    monkeypatch.delenv("VM_RAG_MAX_PER_SOURCE", raising=False)
    monkeypatch.setenv("VM_RAG_LEXICAL_WEIGHT", "0.45")
    return rb


class TestRareTermIsReachable:
    def test_note_outside_the_pool_is_returned_and_ranks_first(self, wired):
        results = wired.retrieve("flatmodel", top_k=3)
        assert [row["source"] for row in results][0] == RARE_SOURCE

    def test_without_the_boost_the_note_loses_to_the_transcripts(self, wired, monkeypatch):
        """The pool alone is not enough: cosine plus source boosts still bury it."""
        monkeypatch.setenv("VM_RAG_LEXICAL_WEIGHT", "0")
        results = wired.retrieve("flatmodel", top_k=3)
        assert RARE_SOURCE not in [row["source"] for row in results]

    def test_a_query_with_no_rare_terms_is_ranked_by_cosine_alone(self, wired):
        results = wired.retrieve("scan chain lesson", top_k=3)
        assert len(results) == 3
        assert results[0]["source"].startswith("atpg_video_subs__")
        assert all(row["source"] != RARE_SOURCE for row in results)

    def test_scores_stay_descending_without_a_source_filter(self, wired):
        results = wired.retrieve("flatmodel", top_k=4)
        scores = [row["score"] for row in results]
        assert scores == sorted(scores, reverse=True)
