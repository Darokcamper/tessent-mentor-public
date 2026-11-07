"""Export a fresher assessment as a handout: Markdown for review, .docx for the SPOC.

The document is assembled from an :func:`core.assess_mode.assess_deck` result and nothing
else, so what the reviewer reads is exactly what the pipeline produced - the same numbers,
the same per-slide notes, the same gaps. Two things are added on purpose:

* a **coverage statement** and a **provenance appendix**, because the value of the sheet is
  that every technical statement can be traced to a manual file and page the reviewer can
  open, and that the slides which were *not* covered say so;
* a **legend** separating manual citations from unverified picture descriptions, so nobody
  mistakes a described figure for evidence.

Nothing is written to disk: both exporters return bytes/str for ``st.download_button``.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.citation_check import status_label

_HOW_TO_READ = (
    "Every slide entry lists what a fresher must be able to state, the mistake they usually "
    "make, and one question with the expected answer. Technical statements carry a manual "
    "citation of the form [Source: <file>, Page <N>]; each one is checked against the page "
    "that holds the text before it is printed here. A slide whose topic the knowledge base "
    "does not cover says so explicitly instead of guessing."
)


def _citations_of(item: Dict[str, Any]) -> List[str]:
    refs = [str(ref) for ref in (item.get("manual_refs") or []) if str(ref).strip()]
    if refs:
        return refs
    return [str(citation) for citation in (item.get("citations") or [])]


def _verification_of(item: Dict[str, Any]) -> str:
    report = item.get("citation_check") or {}
    counts = report.get("counts") or {}
    total = int(report.get("total") or 0)
    if not total:
        return "no citation to check"
    if report.get("corpus_available") is False:
        return f"{total} citation(s) not checked - knowledge corpus unavailable on this host"
    verified = int(counts.get("verified") or 0)
    shifted = int(counts.get("page_shifted") or 0)
    unverified = int(counts.get("unverified") or 0)
    missing = int(report.get("pages_missing") or 0)
    if missing:
        return f"{missing} of {total} cited page(s) NOT FOUND - do not rely on those"
    if verified + shifted == total:
        if shifted:
            return f"all {total} page(s) confirmed, {shifted} number(s) corrected"
        return f"all {total} citation(s) verified"
    return (f"{verified + shifted}/{total} verified; "
            f"{unverified} cite a real page whose wording differs (paraphrase)")


def build_document(result: Dict[str, Any], *, title: str = "Fresher assessment",
                   prepared_for: str = "", deck_label: str = "") -> Dict[str, Any]:
    """Normalise an assessment result into the structure both exporters consume."""
    coverage = result.get("coverage") or {}
    assessments = list(result.get("assessments") or [])
    return {
        "title": str(title or "Fresher assessment").strip(),
        "prepared_for": str(prepared_for or "").strip(),
        "deck_label": str(deck_label or "").strip(),
        "generated": datetime.now().strftime("%Y-%m-%d"),
        "how_to_read": _HOW_TO_READ,
        "coverage": coverage,
        "flows": coverage.get("flows") or [],
        "slides": assessments,
        "quiz": (result.get("quiz") or {}).get("questions") or [],
        "gaps": coverage.get("slides_with_gaps") or [],
    }


def to_markdown(document: Dict[str, Any]) -> str:
    """Plain Markdown version (easy to paste into a mail or a ticket)."""
    lines: List[str] = [f"# {document['title']}", ""]
    meta = [f"Generated {document['generated']}"]
    if document.get("deck_label"):
        meta.append(f"Deck: {document['deck_label']}")
    if document.get("prepared_for"):
        meta.append(f"Prepared for: {document['prepared_for']}")
    lines.append(" - ".join(meta))
    lines.append("")

    coverage = document.get("coverage") or {}
    lines.append("## What this sheet covers")
    lines.append("")
    lines.append(f"- {coverage.get('statement', 'coverage not recorded')}")
    for flow in document.get("flows") or []:
        if flow.get("slides"):
            numbers = ", ".join(str(number) for number in flow["slides"])
            lines.append(f"- {flow['name']}: slides {numbers}")
    lines.append("")
    lines.append(f"**How to read it.** {document['how_to_read']}")
    lines.append("")

    lines.append("## Slide-by-slide assessment")
    for item in document["slides"]:
        lines.append("")
        lines.append(f"### {item.get('label') or ('Slide ' + str(item.get('slide_number')))}")
        must_know = item.get("must_know") or []
        if must_know:
            lines.append("")
            lines.append("**Must be able to state**")
            for point in must_know:
                lines.append(f"- {point}")
        if item.get("misconception"):
            lines.append("")
            lines.append(f"**Typical mistake:** {item['misconception']}")
        if item.get("check_question"):
            lines.append("")
            lines.append(f"**Check question:** {item['check_question']}")
        if item.get("expected_answer"):
            lines.append(f"**Expected answer:** {item['expected_answer']}")
        if item.get("not_covered"):
            lines.append("")
            lines.append(f"**Not covered by the knowledge base:** {item['not_covered']}")
        if item.get("parse_error"):
            lines.append("")
            lines.append(f"**Note:** {item['parse_error']} — raw reply kept in the app only.")
        refs = _citations_of(item)
        if refs:
            lines.append("")
            lines.append(f"**Manual pages:** {'; '.join(refs)} ({_verification_of(item)})")

    quiz = document.get("quiz") or []
    if quiz:
        lines.append("")
        lines.append("## End-of-session quiz")
        for position, entry in enumerate(quiz, start=1):
            lines.append("")
            lines.append(f"{position}. {entry['question']}")
        lines.append("")
        lines.append("## Answer key")
        lines.append("")
        for position, entry in enumerate(quiz, start=1):
            slide = entry.get("slide")
            where = f" (slide {slide})" if slide else ""
            lines.append(f"{position}. {entry['answer']}{where}")

    lines.append("")
    lines.append("## Provenance appendix")
    lines.append("")
    lines.append("| Slide | Manual file | Page | Verification |")
    lines.append("| --- | --- | --- | --- |")
    for item in document["slides"]:
        report = item.get("citation_check") or {}
        for citation in report.get("citations") or []:
            lines.append(
                f"| {item.get('slide_number')} | {citation.get('source')} | "
                f"{citation.get('page')} | {status_label(citation)} |"
            )
    lines.append("")

    gaps = document.get("gaps") or []
    if gaps:
        lines.append("## Slides with an explicit knowledge-base gap")
        lines.append("")
        for gap in gaps:
            lines.append(f"- Slide {gap.get('slide')} ({gap.get('title') or 'untitled'}): "
                         f"{gap.get('reason')}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def to_docx(document: Dict[str, Any]) -> bytes:
    """Word handout: cover page, per-slide blocks, quiz, answer key, provenance appendix."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, RGBColor

    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    def para(text: str = "", *, bold: bool = False, italic: bool = False,
             size: Optional[int] = None, space_after: int = 4):
        paragraph = doc.add_paragraph()
        run = paragraph.add_run(text)
        run.bold = bold
        run.italic = italic
        if size:
            run.font.size = Pt(size)
        paragraph.paragraph_format.space_after = Pt(space_after)
        return paragraph

    def label_value(label: str, value: str):
        paragraph = doc.add_paragraph()
        heading = paragraph.add_run(f"{label} ")
        heading.bold = True
        paragraph.add_run(value)
        paragraph.paragraph_format.space_after = Pt(2)
        return paragraph

    # ── Cover ────────────────────────────────────────────────────────────────
    title = doc.add_heading(document["title"], level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    meta = doc.add_paragraph()
    meta_run = meta.add_run(f"Generated {document['generated']}")
    meta_run.italic = True
    meta_run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
    if document.get("deck_label"):
        label_value("Deck:", document["deck_label"])
    if document.get("prepared_for"):
        label_value("Prepared for:", document["prepared_for"])

    doc.add_heading("What this sheet covers", level=1)
    coverage = document.get("coverage") or {}
    para(coverage.get("statement", "coverage not recorded"))
    for flow in document.get("flows") or []:
        if flow.get("slides"):
            numbers = ", ".join(str(number) for number in flow["slides"])
            doc.add_paragraph(f"{flow['name']}: slides {numbers}", style="List Bullet")
    para(document["how_to_read"], italic=True)

    # ── Per-slide assessment ─────────────────────────────────────────────────
    doc.add_page_break()
    doc.add_heading("Slide-by-slide assessment", level=1)
    for item in document["slides"]:
        doc.add_heading(item.get("label") or f"Slide {item.get('slide_number')}", level=2)
        must_know = item.get("must_know") or []
        if must_know:
            para("Must be able to state", bold=True)
            for point in must_know:
                doc.add_paragraph(point, style="List Bullet")
        if item.get("misconception"):
            label_value("Typical mistake:", item["misconception"])
        if item.get("check_question"):
            label_value("Check question:", item["check_question"])
        if item.get("expected_answer"):
            label_value("Expected answer:", item["expected_answer"])
        if item.get("not_covered"):
            label_value("Not covered by the knowledge base:", item["not_covered"])
        refs = _citations_of(item)
        if refs:
            label_value("Manual pages:", f"{'; '.join(refs)} ({_verification_of(item)})")

    # ── Quiz and key ─────────────────────────────────────────────────────────
    quiz = document.get("quiz") or []
    if quiz:
        doc.add_page_break()
        doc.add_heading("End-of-session quiz", level=1)
        for position, entry in enumerate(quiz, start=1):
            doc.add_paragraph(f"{position}. {entry['question']}")
        doc.add_heading("Answer key", level=1)
        for position, entry in enumerate(quiz, start=1):
            slide = entry.get("slide")
            where = f" (slide {slide})" if slide else ""
            label_value(f"{position}.", f"{entry['answer']}{where}")

    # ── Provenance appendix ──────────────────────────────────────────────────
    rows = []
    for item in document["slides"]:
        report = item.get("citation_check") or {}
        for citation in report.get("citations") or []:
            rows.append((str(item.get("slide_number")), str(citation.get("source")),
                         str(citation.get("page")), str(citation.get("status"))))
    if rows:
        doc.add_page_break()
        doc.add_heading("Provenance appendix", level=1)
        para("Every technical statement in this sheet points at one of these manual pages. "
             "Each citation is checked against the cited page before printing: 'verified' "
             "means the page carries the quoted wording; 'page_shifted' means the number was "
             "corrected to the page that does; 'unverified' means the page could not confirm "
             "it and the statement should not be relied on.", italic=True)
        table = doc.add_table(rows=1, cols=4)
        table.style = "Light Grid Accent 1"
        header = table.rows[0].cells
        for position, name in enumerate(("Slide", "Manual file", "Page", "Verification")):
            header[position].text = name
        for row in rows:
            cells = table.add_row().cells
            for position, value in enumerate(row):
                cells[position].text = value

    gaps = document.get("gaps") or []
    if gaps:
        doc.add_heading("Slides with an explicit knowledge-base gap", level=1)
        for gap in gaps:
            doc.add_paragraph(f"Slide {gap.get('slide')} ({gap.get('title') or 'untitled'}): "
                              f"{gap.get('reason')}", style="List Bullet")

    para()
    footer = para("Generated by the Tessent Mentor AI deck explainer. The uploaded "
                  "presentation is processed in memory only; no slide content, file name or "
                  "path is written to the session log.", size=8, italic=True)
    footer.runs[0].font.color.rgb = RGBColor(0x70, 0x70, 0x70)

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
