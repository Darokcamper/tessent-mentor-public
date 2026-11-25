"""`core/citation_check.py` - page-level verification of generated citations.

The production defect behind these tests: a slide explanation quoted the bullets printed on
page 34 of a corpus file and cited page 33. Page-level traceability is what the tool sells,
so a citation is now checked against the page it names, and an off-by-N is corrected to the
page that actually carries the wording. Everything here runs offline on a synthetic corpus.
"""

from __future__ import annotations

from pathlib import Path

from core import citation_check as cc

PAGE_33 = "ManufacturingDefects\nDefects are physical problems that occur in silicon\n"
PAGE_34 = ("Manufacturing Defects (Cont.)\nHard tie to power or ground:\n"
           "Port is always stuck-at 1 or stuck-at 0\nShorts, or bridges between two connections:\n"
           "Data in one path adversely affects data in a different path\n"
           "Capacitive or resistive pads:\nIndividualports maybe slowduring0->1transition\n")
PAGE_35 = "Getting Help - Within Tessent Tool\nType help followed by the command name\n"


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    target = root / "knowledge" / "txt"
    target.mkdir(parents=True)
    (target / "slides.txt").write_text(
        f"--- PAGE 33 --- \n{PAGE_33}\n--- PAGE 34 --- \n{PAGE_34}\n--- PAGE 35 --- \n{PAGE_35}\n",
        encoding="utf-8",
    )
    return root


BULLETS_CLAIM = ("Hard tie to power or ground makes a port stuck-at 1 or stuck-at 0, shorts or "
                 "bridges corrupt data in another path, capacitive pads make transitions slow.")


class TestPageLookup:
    def test_reads_the_requested_page_only(self, tmp_path):
        root = _corpus(tmp_path)
        text = cc.read_page("slides.txt", 34, root=root)
        assert "Hard tie to power or ground" in text
        assert "Getting Help" not in text

    def test_missing_page_is_none(self, tmp_path):
        assert cc.read_page("slides.txt", 9999, root=_corpus(tmp_path)) is None

    def test_missing_file_is_none(self, tmp_path):
        assert cc.read_page("absent.txt", 1, root=_corpus(tmp_path)) is None

    def test_file_without_markers_is_none(self, tmp_path):
        root = _corpus(tmp_path)
        (root / "knowledge" / "txt" / "flat.txt").write_text("no page markers here", encoding="utf-8")
        assert cc.read_page("flat.txt", 1, root=root) is None


class TestVerification:
    def test_correct_page_is_verified(self, tmp_path):
        result = cc.verify_citation("slides.txt", 34, BULLETS_CLAIM, root=_corpus(tmp_path))
        assert result["status"] == "verified"
        assert result["correct_page"] == 34
        assert len(result["matched"]) >= 2

    def test_off_by_one_is_detected_and_corrected(self, tmp_path):
        """The exact production case: page 33 cited for text printed on page 34."""
        result = cc.verify_citation("slides.txt", 33, BULLETS_CLAIM, root=_corpus(tmp_path))
        assert result["status"] == "page_shifted"
        assert result["correct_page"] == 34
        assert "34" in result["reason"]

    def test_nonexistent_page_is_unverified(self, tmp_path):
        result = cc.verify_citation("slides.txt", 9999, BULLETS_CLAIM, root=_corpus(tmp_path))
        assert result["status"] == "unverified"
        assert "not found" in result["reason"]

    def test_unrelated_page_is_not_silently_accepted(self, tmp_path):
        result = cc.verify_citation("slides.txt", 35, "scan chain reordering for clock domains",
                                    root=_corpus(tmp_path))
        assert result["status"] == "unverified"
        assert result["correct_page"] is None

    def test_bad_page_type_never_raises(self, tmp_path):
        result = cc.verify_citation("slides.txt", "not-a-number", "x", root=_corpus(tmp_path))
        assert result["status"] == "unverified"


class TestVerifyText:
    def _explanation(self):
        return (
            "The slide lists manufacturing defects.\n"
            f"- Hard tie, bridges and pads: {BULLETS_CLAIM} [Source: slides.txt, Page 33]\n"
            "- Inside the Tessent tool you type help followed by the command name "
            "[Source: slides.txt, Page 35]\n"
        )

    def test_report_counts_and_corrected_text(self, tmp_path):
        report = cc.verify_text(self._explanation(), root=_corpus(tmp_path))
        assert report["total"] == 2
        assert report["counts"]["page_shifted"] == 1
        assert report["counts"]["verified"] == 1
        assert "Page 34" in report["corrected_text"]
        assert "Page 33" not in report["corrected_text"]
        assert report["changed"] is True
        assert report["all_verified"] is False

    def test_claims_are_never_rewritten(self, tmp_path):
        report = cc.verify_text(self._explanation(), root=_corpus(tmp_path))
        assert BULLETS_CLAIM in report["corrected_text"]

    def test_clean_text_is_not_changed(self, tmp_path):
        text = f"Pads and bridges: {BULLETS_CLAIM} [Source: slides.txt, Page 34]"
        report = cc.verify_text(text, root=_corpus(tmp_path))
        assert report["changed"] is False
        assert report["all_verified"] is True

    def test_text_without_citations(self, tmp_path):
        report = cc.verify_text("No citations at all.", root=_corpus(tmp_path))
        assert report["total"] == 0
        assert report["all_verified"] is False


class TestSummaries:
    def test_all_verified_line(self, tmp_path):
        text = f"{BULLETS_CLAIM} [Source: slides.txt, Page 34]"
        line = cc.summary_line(cc.verify_text(text, root=_corpus(tmp_path)))
        assert "All 1 citation(s) verified" in line

    def test_mixed_line_mentions_both(self, tmp_path):
        text = (f"{BULLETS_CLAIM} [Source: slides.txt, Page 33] "
                "[Source: slides.txt, Page 9999]")
        line = cc.summary_line(cc.verify_text(text, root=_corpus(tmp_path)))
        assert "corrected" in line and "not be relied on" in line

    def test_empty_line(self):
        assert cc.summary_line({"total": 0}) == "No citations to check on this slide."


class TestClaimExtraction:
    def test_claim_is_the_sentence_before_the_citation(self):
        text = "First sentence here.\nSecond sentence carries the claim [Source: f.txt, Page 3]"
        span = cc.claim_span(text, text.index("[Source"))
        assert span == "Second sentence carries the claim"

    def test_terms_skip_short_words_and_stopwords(self):
        terms = cc.claim_terms("The slide shows the port and a stuck fault model")
        assert "model" in terms
        assert "slide" not in terms
        assert all(len(term) >= 5 for term in terms)


class TestMissingCorpusIsNotAnAccusation:
    """On a cold container the bundle is absent; citations must not read as fabricated."""

    def _empty_root(self, tmp_path):
        root = tmp_path / "empty_root"
        root.mkdir()
        return root

    def test_corpus_availability_is_reported(self, tmp_path):
        assert cc.corpus_available(_corpus(tmp_path)) is True
        assert cc.corpus_available(self._empty_root(tmp_path)) is False

    def test_status_is_not_checked_rather_than_missing(self, tmp_path):
        result = cc.verify_citation("slides.txt", 34, BULLETS_CLAIM,
                                    root=self._empty_root(tmp_path))
        assert result["status"] == "unverified"
        assert result["page_exists"] is None
        assert "corpus is not available" in result["reason"]

    def test_report_says_so_once(self, tmp_path):
        text = f"{BULLETS_CLAIM} [Source: slides.txt, Page 34]"
        report = cc.verify_text(text, root=self._empty_root(tmp_path))
        assert report["corpus_available"] is False
        assert report["pages_missing"] == 0
        assert "not checked" in cc.summary_line(report)


class TestMalformedCitations:
    """A model can emit a citation bracket that is not a citation as far as the checker knows.

    Production review caught "[; tshell_ref (1).txt, Page 495]": the label is junk, so
    CITATION_RE never matched it, the page was never verified, and the malformed bracket was
    shown to the user. Normalising happens before checking, and only brackets that carry a
    corpus file name and a page number are touched.
    """

    def test_junk_label_is_replaced_with_the_canonical_form(self):
        text, rewrites = cc.normalise_citations("see [; slides.txt, Page 34] for the list")
        assert text == "see [Source: slides.txt, Page 34] for the list"
        assert rewrites == 1

    def test_missing_label_is_added(self):
        text, rewrites = cc.normalise_citations("[slides.txt, Page 34]")
        assert text == "[Source: slides.txt, Page 34]"
        assert rewrites == 1

    def test_junk_separator_inside_the_source_label(self):
        text, _ = cc.normalise_citations("[Source; slides.txt; Page 34]")
        assert text == "[Source: slides.txt, Page 34]"

    def test_file_name_with_parentheses_and_spaces_survives(self):
        text, _ = cc.normalise_citations("[; tshell_ref (1).txt, Page 495]")
        assert text == "[Source: tshell_ref (1).txt, Page 495]"

    def test_canonical_citation_is_left_alone(self):
        text, rewrites = cc.normalise_citations("[Source: slides.txt, Page 34]")
        assert text == "[Source: slides.txt, Page 34]"
        assert rewrites == 0

    def test_prose_that_only_mentions_a_page_is_not_rewritten(self):
        text, rewrites = cc.normalise_citations("[see the lab notes, page 5] and [3]")
        assert rewrites == 0
        assert text == "[see the lab notes, page 5] and [3]"

    def test_a_malformed_citation_is_now_verified_instead_of_ignored(self, tmp_path):
        report = cc.verify_text(f"{BULLETS_CLAIM} [; slides.txt, Page 34]",
                                root=_corpus(tmp_path))
        assert report["total"] == 1
        assert report["citations_normalised"] == 1
        assert report["counts"]["verified"] == 1
        assert "[Source: slides.txt, Page 34]" in report["corrected_text"]

    def test_normalisation_alone_still_counts_as_a_change(self, tmp_path):
        # The bracket is malformed but the page does not carry the claim: the text still
        # changes (the bracket is repaired) even though no page number is corrected.
        report = cc.verify_text("unrelated wording [; slides.txt, Page 35]",
                                root=_corpus(tmp_path))
        assert report["citations_normalised"] == 1
        assert report["changed"] is True
        assert report["counts"]["unverified"] == 1

        assert "PAGE NOT FOUND" not in cc.status_label(report["citations"][0])
