"""Rare-term (lexical) boost: a precise query word must beat a merely similar transcript.

The defect, measured on this corpus: the query "flatmodel" returned no page of Flatmodel.txt
at all, although that file prints the word nine times and only three other files mention it
once each. The note that defines the term scores 0.3121 cosine and sits at rank 101 of 30,788
chunks, while an unrelated lesson transcript scores 0.85 - so it is outside the 64-chunk
candidate pool *and* it cannot win the rerank. Both halves of that are exercised here.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.rag_ranking import (DEFAULT_LEXICAL_WEIGHT, MAX_LEXICAL_BOOST, RARE_DF_MAX,
                              document_frequency, lexical_boost, lexical_weight_from_env,
                              lowercase_corpus, query_tokens, rare_term_hits, rare_term_scan,
                              rarity_weights)

# A miniature corpus with the same shape as the real one: one chunk repeats a rare term,
# a couple mention it once, and the rest are generic lessons about common words.
FILLER = [f"lesson {index} about the scan chain, the clock and the pattern" for index in range(30)]
CORPUS = [
    "How to open the failing instance without rerunning ATPG --> using flatmodel. "
    "Flatmodel: design data and settings are stored. read_flat_model flatmodel. flatmodel flatmodel",
    "the atpg flow shows the flatmodel option once in a slide",
    "design rules overview mentions Flatmodel in passing",
] + FILLER


class TestQueryTokens:
    def test_stopwords_short_words_and_duplicates_are_dropped(self):
        assert query_tokens("How do I open a failing instance with Flatmodel?") == [
            "open", "failing", "instance", "flatmodel"]

    def test_empty_and_none_are_empty(self):
        assert query_tokens("") == []
        assert query_tokens(None) == []

    def test_repeats_of_the_same_word_are_counted_once(self):
        assert query_tokens("fault fault FAULT launch") == ["fault", "launch"]


class TestDocumentFrequency:
    def test_counts_every_chunk_that_contains_the_term(self):
        assert document_frequency("flatmodel", CORPUS, cache={}) == 3

    def test_matching_ignores_case(self):
        # "Flatmodel" at the start of a sentence is the same term.
        assert document_frequency("flatmodel", ["Flatmodel: design data"], cache={}) == 1

    def test_term_absent_everywhere_is_zero(self):
        assert document_frequency("lockup", CORPUS, cache={}) == 0

    def test_scan_reports_positions_for_a_rare_term(self):
        count, positions = rare_term_scan("flatmodel", CORPUS, cache={})
        assert count == 3 and positions == (0, 1, 2)

    def test_scan_drops_positions_once_the_term_is_not_rare(self):
        count, positions = rare_term_scan("scan", CORPUS, max_df=RARE_DF_MAX, cache={})
        assert count > RARE_DF_MAX
        assert positions == ()

    def test_a_supplied_cache_keeps_corpora_apart(self):
        # Two different corpora must not share one answer through the default cache.
        first, second = {}, {}
        assert document_frequency("flatmodel", ["flatmodel"], cache=first) == 1
        assert document_frequency("flatmodel", ["nothing here"], cache=second) == 0


class TestPreLoweredCorpus:
    """Lowering once is worth 8x: measured 0.24 s per term scan against 0.03 s on this cache."""

    def test_lowercase_corpus_lowercases_every_chunk(self):
        assert lowercase_corpus(["Flatmodel", None]) == ["flatmodel", "none"]

    def test_the_pre_lowered_path_gives_the_same_answer(self):
        plain = document_frequency("flatmodel", CORPUS, cache={})
        pre_lowered = document_frequency("flatmodel", lowercase_corpus(CORPUS), cache={},
                                        lower=False)
        assert plain == pre_lowered == 3

    def test_the_pre_lowered_path_trusts_the_caller(self):
        # Documented contract: lower=False means "these are already lower-case".
        assert document_frequency("flatmodel", ["Flatmodel: design data"], cache={},
                                  lower=False) == 0



class TestRarityWeights:
    def test_a_rare_term_outweighs_a_common_one(self):
        weights = rarity_weights(["flatmodel", "scan"], CORPUS, cache={})
        assert weights["flatmodel"] > weights["scan"]

    def test_a_term_in_four_of_thirty_thousand_chunks_is_near_the_top_of_the_scale(self):
        # The real index: 30,788 chunks, and "flatmodel" occurs in 4 of them.
        weights = rarity_weights(["flatmodel"], CORPUS, total_chunks=30788, cache={})
        assert weights["flatmodel"] > 0.8

    def test_weights_are_bounded(self):
        weights = rarity_weights(["flatmodel", "scan", "absent"], CORPUS, cache={})
        assert all(0.0 <= value <= 1.0 for value in weights.values())

    def test_empty_token_list_gives_no_weights(self):
        assert rarity_weights([], CORPUS, cache={}) == {}

    def test_corpus_size_can_be_supplied_for_a_large_index(self):
        small = rarity_weights(["flatmodel"], CORPUS, total_chunks=33, cache={})
        large = rarity_weights(["flatmodel"], CORPUS, total_chunks=30788, cache={})
        assert large["flatmodel"] > small["flatmodel"]



class TestRareTermHits:
    def test_only_rare_terms_are_returned(self):
        hits = rare_term_hits(["flatmodel", "scan"], CORPUS, cache={})
        assert hits == {"flatmodel": [0, 1, 2]}

    def test_a_query_without_rare_terms_adds_nothing(self):
        assert rare_term_hits(["scan", "clock"], CORPUS, cache={}) == {}


class TestCeilingIsHighEnoughToMatter:
    """The measured gap the ceiling has to close, so it cannot be quietly lowered again."""

    def test_ceiling_exceeds_the_measured_cosine_gap(self):
        # defining page 0.3121 + 0.15 source boost vs transcript 0.85 + 0.25 + 0.15
        gap = (0.85 + 0.25 + 0.15) - (0.3121 + 0.15)
        assert MAX_LEXICAL_BOOST > gap

    def test_the_measured_scores_put_the_defining_page_first(self):
        weights = rarity_weights(["flatmodel"], CORPUS, total_chunks=30788, cache={})
        defining = 0.3121 + 0.15 + lexical_boost(["flatmodel"], weights, "flatmodel " * 9)
        transcript = 0.85 + 0.25 + 0.15
        assert defining > transcript


class TestDefaultIsOn:
    def test_default_weight_is_the_measured_one(self):
        assert DEFAULT_LEXICAL_WEIGHT == 0.45

    def test_unset_env_uses_the_default(self, monkeypatch):
        monkeypatch.delenv("VM_RAG_LEXICAL_WEIGHT", raising=False)
        assert lexical_weight_from_env() == 0.45

    def test_zero_disables_it(self, monkeypatch):
        monkeypatch.setenv("VM_RAG_LEXICAL_WEIGHT", "0")
        assert lexical_weight_from_env() == 0.0

    def test_explicit_value_is_honoured(self, monkeypatch):
        monkeypatch.setenv("VM_RAG_LEXICAL_WEIGHT", "0.9")
        assert lexical_weight_from_env() == 0.9

    def test_junk_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("VM_RAG_LEXICAL_WEIGHT", "lots")
        assert lexical_weight_from_env() == 0.45

    def test_negative_is_clamped_to_zero(self, monkeypatch):
        monkeypatch.setenv("VM_RAG_LEXICAL_WEIGHT", "-2")
        assert lexical_weight_from_env() == 0.0


class TestLexicalBoost:
    def _weights(self, *tokens):
        return {token: 1.0 for token in tokens}

    def test_no_tokens_means_no_boost(self):
        assert lexical_boost([], self._weights("flatmodel"), "flatmodel") == 0.0

    def test_zero_weight_switches_the_boost_off(self):
        assert lexical_boost(["flatmodel"], self._weights("flatmodel"), "flatmodel",
                             weight=0.0) == 0.0

    def test_a_chunk_without_the_term_gets_nothing(self):
        assert lexical_boost(["flatmodel"], self._weights("flatmodel"),
                             "the atpg flow slide") == 0.0

    def test_a_repeated_term_beats_a_passing_mention(self):
        note = "flatmodel " * 9
        mention = "the slide mentions flatmodel once"
        assert (lexical_boost(["flatmodel"], self._weights("flatmodel"), note)
                > lexical_boost(["flatmodel"], self._weights("flatmodel"), mention))

    def test_matching_ignores_case(self):
        assert lexical_boost(["flatmodel"], self._weights("flatmodel"),
                             "Flatmodel: design data") > 0.0

    def test_boost_is_capped(self):
        many = "flatmodel " * 500
        assert lexical_boost(["flatmodel"], self._weights("flatmodel"), many) == MAX_LEXICAL_BOOST

    def test_repeat_factor_is_logarithmic(self):
        score = lexical_boost(["flatmodel"], {"flatmodel": 0.5}, "flatmodel " * 4, weight=1.0,
                              ceiling=99.0)
        assert score == 0.5 * (1 + math.log(4))
