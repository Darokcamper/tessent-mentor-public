"""PPT Explainer core (WP-10).

Slide extraction, per-slide explanation and whole-deck sequencing for uploaded
``.pptx`` presentations, built on the project's existing pieces:

* extraction      - ``python-pptx`` (already a project dependency)
* source grounding - ``core.rag_builder.retrieve``
* generation      - the shared ``core.llm`` router (capability ``batch_report``)

Privacy (approved policy, see ``docs/enhancements/WP10_PPT_PRIVACY_ANALYSIS.md``):
this module never writes logs, never persists the upload, and never includes a file
name or path in a prompt. Callers log counts only via ``log_ppt_explainer``.

Grounding rules: the model may only cite the reference context this module retrieved;
the prompt separates SLIDE CONTENT from REFERENCE CONTEXT and requires an explicit
"not covered by the knowledge base" statement when evidence is insufficient.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

SUPPORTED_SUFFIXES = (".pptx",)

_MAX_SLIDE_CHARS = 6000
_MAX_EVIDENCE_CHARS = 9000
_MAX_TABLE_ROWS = 60
_MAX_IMAGES_PER_SLIDE = 4


class PptExplainerError(RuntimeError):
    """Raised when a presentation cannot be read at all."""


@dataclass
class Slide:
    number: int
    title: str = ""
    text: str = ""
    tables: List[List[List[str]]] = field(default_factory=list)
    notes: str = ""
    picture_count: int = 0
    pictures: List[Dict[str, Any]] = field(default_factory=list)  # in-memory only
    warnings: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.title.strip() or self.text.strip() or self.tables or self.notes.strip())

    def retrieval_text(self) -> str:
        """Text used to look for supporting evidence (title first, then body)."""
        parts = [self.title, self.text]
        for table in self.tables:
            for row in table[:_MAX_TABLE_ROWS]:
                parts.append(" | ".join(cell for cell in row if cell))
        return " ".join(part.strip() for part in parts if part and part.strip())[:2000]

    def display_label(self) -> str:
        title = self.title.strip() or "(untitled slide)"
        return f"Slide {self.number}: {title}"


@dataclass
class Deck:
    slides: List[Slide] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    slide_size: Optional[tuple] = None

    @property
    def slide_count(self) -> int:
        return len(self.slides)


def _shape_text(shape) -> List[str]:
    texts: List[str] = []
    try:
        if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
            for paragraph in shape.text_frame.paragraphs:
                line = "".join(run.text for run in paragraph.runs).strip()
                if line:
                    texts.append(line)
    except Exception:
        pass
    return texts


def _shape_table(shape) -> Optional[List[List[str]]]:
    try:
        if getattr(shape, "has_table", False) and shape.has_table:
            rows = []
            for row in shape.table.rows:
                cells = [str(cell.text).strip() for cell in row.cells]
                if any(cells):
                    rows.append(cells)
            return rows or None
    except Exception:
        pass
    return None


def _slide_notes(slide) -> str:
    try:
        frame = slide.notes_slide.notes_text_frame
        if frame is None:
            return ""
        return "\n".join(p.text.strip() for p in frame.paragraphs if p.text.strip())
    except Exception:
        return ""


def extract_deck(file_bytes: bytes, filename: str = "") -> Deck:
    """Read a .pptx into ordered slides. Raises PptExplainerError on unreadable input."""
    lower = str(filename or "").lower()
    if lower.endswith(".ppt"):
        raise PptExplainerError(
            "Legacy binary .ppt presentations are not supported. Save the deck as .pptx "
            "and upload it again."
        )
    if lower and not lower.endswith(".pptx"):
        suffix = lower.rsplit(".", 1)[-1] or "unknown"
        raise PptExplainerError(
            f"Unsupported file type '{suffix}'. Only .pptx presentations can be explained."
        )
    if not file_bytes:
        raise PptExplainerError("The uploaded file is empty.")

    try:
        from pptx import Presentation
    except ImportError as error:  # pragma: no cover - dependency is pinned
        raise PptExplainerError(f"python-pptx is not available: {error}") from error

    try:
        presentation = Presentation(io.BytesIO(file_bytes))
    except Exception as error:
        raise PptExplainerError(
            "The presentation could not be opened; the file may be corrupt or not a "
            f"PowerPoint package ({type(error).__name__})."
        ) from error

    deck = Deck()
    try:
        deck.slide_size = (presentation.slide_width, presentation.slide_height)
    except Exception:
        deck.slide_size = None

    for index, slide in enumerate(presentation.slides, start=1):
        record = Slide(number=index)
        try:
            for shape in slide.shapes:
                for line in _shape_text(shape):
                    record.text = f"{record.text}\n{line}" if record.text else line
                table = _shape_table(shape)
                if table:
                    record.tables.append(table)
                try:
                    if hasattr(shape, "image"):
                        record.picture_count += 1
                        # F1: keep the picture in memory only, tagged with its slide, so a
                        # vision model can describe it later. Nothing is written to disk.
                        picture = getattr(shape, "image", None)
                        blob = getattr(picture, "blob", None)
                        if blob:
                            record.pictures.append({
                                "slide_number": index,
                                "name": str(getattr(picture, "filename", "") or f"picture-{record.picture_count}"),
                                "mime": str(getattr(picture, "content_type", "") or "image/png"),
                                "bytes": bytes(blob),
                            })
                except Exception:
                    pass
        except Exception as error:
            record.warnings.append(
                f"Some content on slide {index} could not be read ({type(error).__name__})."
            )

        record.notes = _slide_notes(slide)

        if record.text:
            record.title = record.text.splitlines()[0].strip()[:160]

        if record.is_empty:
            record.warnings.append(f"Slide {index} contains no extractable text, table or notes.")
        if record.picture_count:
            record.warnings.append(
                f"Slide {index} contains {record.picture_count} picture(s). They are described "
                "from the slide image only when a vision model is reachable, and are never "
                "treated as manual evidence."
            )
        if len(record.text) > _MAX_SLIDE_CHARS:
            record.text = record.text[:_MAX_SLIDE_CHARS]
            record.warnings.append(f"Slide {index} text was truncated for prompting.")

        deck.slides.append(record)
        deck.warnings.extend(record.warnings)

    return deck


# ── Evidence and explanation ────────────────────────────────────────────────

NOT_COVERED = (
    "The available knowledge base does not contain sufficient supporting information "
    "for this part of the slide."
)

# ── WP-R3: evidence *relevance*, not mere retrieval presence ─────────────────
# Retrieval always returns its top-k chunks, so "chunks > 0" said nothing about whether
# they had anything to do with the slide: during live validation a quantum-annealing
# slide was reported as `grounded` with six retrieved chunks. Coverage is now decided by
# how much of the slide's own vocabulary actually appears in the retrieved text.
_COVERAGE_OVERLAP_THRESHOLD = 0.20
_COVERAGE_STOPWORDS = frozenset({
    "about", "above", "after", "again", "against", "also", "among", "another", "because",
    "been", "before", "being", "below", "between", "both", "cannot", "could", "does",
    "doing", "during", "each", "from", "further", "have", "having", "here", "into",
    "itself", "more", "most", "must", "only", "other", "over", "same", "should", "some",
    "such", "than", "that", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "under", "until", "very", "were", "what", "when", "where", "which",
    "while", "will", "with", "would", "your", "slide", "shows", "using", "used", "command",
})
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _slide_terms(text: str) -> List[str]:
    """Distinctive lowercase terms of the slide text (len >= 4, no stopwords)."""
    terms: List[str] = []
    for token in _TOKEN_RE.findall(str(text or "").lower()):
        if len(token) < 4 or token in _COVERAGE_STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def evidence_relevance(slide: Slide, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """How much of the slide's own vocabulary the retrieved evidence actually covers.

    ``overlap`` is the share of the slide's distinctive terms found in the retrieved
    text. It is deliberately simple and explainable — a lexical signal, not a claim of
    semantic entailment — and it never influences abstention, only the coverage label.
    """
    terms = _slide_terms(slide.retrieval_text())
    haystack = " ".join(str(item.get("text") or "") for item in evidence).lower()
    matched = [term for term in terms if term in haystack]
    overlap = (len(matched) / len(terms)) if terms else 0.0
    best_score = 0.0
    for item in evidence:
        try:
            best_score = max(best_score, float(item.get("score") or 0.0))
        except (TypeError, ValueError):
            continue
    return {
        "slide_terms": len(terms),
        "matched_terms": len(matched),
        "overlap": round(overlap, 3),
        "matched": matched[:12],
        "best_retrieval_score": best_score,
        "threshold": _COVERAGE_OVERLAP_THRESHOLD,
        "label": ("grounded" if evidence and overlap >= _COVERAGE_OVERLAP_THRESHOLD
                  else ("no_evidence" if not evidence else "weak_evidence")),
    }


def _default_retrieve(query: str, top_k: int = 6):
    from core.rag_builder import retrieve

    return retrieve(query, top_k=top_k)


def _citation(result: Dict[str, Any]) -> str:
    return f"[Source: {result.get('source')}, Page {result.get('page')}]"


# ── F1: slide-image interpretation ──────────────────────────────────────────
# A slide picture is *the user's own material*, exactly like slide text: it may be
# described, but it is never manual evidence. Every observation therefore carries
# verified=False and is handed to the explanation model in a clearly separated block.

IMAGE_PROMPT = """You are looking at one picture taken from a technical presentation
slide about Siemens Tessent / DFT.

Describe only what is literally visible:
- diagram blocks and their labels, signal names, arrows and their direction;
- waveform names, axes and the transitions actually drawn;
- command lines or code exactly as printed (copy them character for character);
- tables and their column headers.

Hard rules:
1. Never guess a label you cannot read. Write "label unreadable" instead.
2. Never add a command, option, signal or architectural detail that is not visible.
3. Never explain what the picture is *for* and never cite manual pages: that belongs to
   the separate evidence step.
4. If the picture is decorative or unreadable, reply exactly: NO_TECHNICAL_CONTENT

Slide context (may be empty): {context}
"""

NO_VISUAL_CONTENT = "NO_TECHNICAL_CONTENT"


def _default_vision(image_bytes: bytes, context: str = "", mime: str = "image/png") -> str:
    """Default vision call. Imported lazily so offline use never touches a provider."""
    from agents.diagram_vision import describe_diagram

    return describe_diagram(image_bytes, context=context, mime=mime)


def interpret_slide_images(
    slide: Slide,
    *,
    vision_fn: Optional[Callable] = None,
    max_images: int = _MAX_IMAGES_PER_SLIDE,
) -> Dict[str, Any]:
    """Describe the slide's own pictures. Returns unverified observations only.

    Never raises: an unreachable vision model degrades to ``available=False`` so the
    text explanation can still proceed. No image is written to disk, logged or returned
    in the result — only the description text.
    """
    pictures = slide.pictures[:max_images]
    result: Dict[str, Any] = {
        "available": bool(pictures),
        "analysed": 0,
        "skipped": max(0, len(slide.pictures) - len(pictures)),
        "observations": [],
        "error": None,
        "kind": "visual_observation",
        "verified": False,
    }
    if not pictures:
        return result

    describe = vision_fn or _default_vision
    context = " ".join(part for part in (slide.title, slide.text[:400]) if part).strip()
    for picture in pictures:
        try:
            description = str(describe(picture["bytes"], context=context, mime=picture["mime"]) or "").strip()
        except Exception as error:  # keep the failure type only; never echo the image
            result["error"] = type(error).__name__
            continue
        result["analysed"] += 1
        if not description or NO_VISUAL_CONTENT in description.upper():
            continue
        result["observations"].append({
            "picture": picture["name"],
            "description": description,
            "verified": False,
            "kind": "visual_observation",
        })
    if result["error"] and not result["observations"]:
        result["available"] = False
    return result


def _observations_block(images: Dict[str, Any]) -> Optional[str]:
    observations = images.get("observations") or []
    if not observations:
        return None
    lines = [
        "UNVERIFIED VISUAL OBSERVATIONS (from the slide's own pictures; these are NOT manual",
        "evidence, carry no page numbers, and must not be used as the source of any technical",
        "claim):",
    ]
    for item in observations:
        lines.append(f"- picture '{item['picture']}': {item['description']}")
    if images.get("skipped"):
        lines.append(f"- ({images['skipped']} further picture(s) were not analysed.)")
    return "\n".join(lines)


def gather_evidence(slide: Slide, retrieve_fn: Optional[Callable] = None, top_k: int = 6) -> List[Dict[str, Any]]:
    """Knowledge-base chunks that may support this slide. Never invents a citation."""
    query = slide.retrieval_text().strip()
    if not query:
        return []
    retrieve_fn = retrieve_fn or _default_retrieve
    try:
        results = retrieve_fn(query, top_k=top_k) or []
    except Exception:
        return []

    evidence = []
    for item in results:
        text = str(item.get("text") or "")
        source = item.get("source")
        page = item.get("page")
        if not text or not source or page is None:
            continue
        evidence.append(
            {
                "source": source,
                "page": page,
                "text": text,
                "citation": _citation(item),
                "score": item.get("score"),
            }
        )
    return evidence


def _evidence_block(evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return "<NO REFERENCE CONTEXT WAS RETRIEVED>"
    parts = []
    used = 0
    for item in evidence:
        chunk = f"Source Document: {item['source']} (Page {item['page']})\nContent: {item['text']}"
        if used + len(chunk) > _MAX_EVIDENCE_CHARS:
            break
        parts.append(chunk)
        used += len(chunk)
    return "\n\n---\n\n".join(parts) if parts else "<NO REFERENCE CONTEXT WAS RETRIEVED>"


SLIDE_PROMPT = """You are a senior Siemens Tessent DFT trainer explaining ONE slide of a
technical presentation to an engineer who is learning the Tessent scan/ATPG flow.

GROUNDING RULES (MANDATORY):
1. "SLIDE CONTENT" below is the user's own presentation. Treat it as the subject, never as
   a source of Tessent facts.
2. "REFERENCE CONTEXT" is the only permitted source for technical claims, commands,
   options and page numbers. Cite every such claim as [Source: <file>, Page <N>] using a
   file and page that literally appear in the reference context.
3. Copy the source file name EXACTLY as it is printed in the reference context, including
   its extension (for example [Source: tshell_ref (1).txt, Page 1427]). Never shorten the
   name, never drop the extension, never invent a name that is not in the context.
4. Never invent a citation, page number, command, option or manual value. If the reference
   context does not support a claim, write: "{not_covered}"
5. If the slide's subject is not covered by the reference context at all, still describe
   what the slide shows and label the topic as not covered by the knowledge base.
6. Explain every abbreviation and command that appears on the slide.

SLIDE CONTENT (slide {number} of {total}):
{slide_body}

REFERENCE CONTEXT:
{evidence}

Write the explanation with these headings:
### Slide {number}: <subject>
**What this slide shows** - describe the slide content in context.
**Technical explanation** - define the concepts, commands and flow position, with citations.
**Tessent flow position** - where this belongs in the scan/ATPG flow (state if the slide or
the sources do not establish it).
**Not covered by the knowledge base** - list anything you could not verify, or write "None".
"""


def build_slide_prompt(deck: Deck, slide: Slide, evidence: List[Dict[str, Any]],
                       images: Optional[Dict[str, Any]] = None) -> str:
    body = []
    if slide.title:
        body.append(f"Title: {slide.title}")
    if slide.text:
        body.append(slide.text)
    for table in slide.tables:
        body.append("\n".join(" | ".join(cell for cell in row if cell) for row in table[:_MAX_TABLE_ROWS]))
    if slide.notes:
        body.append(f"Speaker notes: {slide.notes}")
    observations = _observations_block(images or {})
    if observations:
        body.append(observations)
    elif slide.picture_count:
        body.append(f"(This slide contains {slide.picture_count} picture(s); no image analysis was performed.)")
    if not body:
        body.append("(This slide contains no extractable text, table or notes.)")

    return SLIDE_PROMPT.format(
        not_covered=NOT_COVERED,
        number=slide.number,
        total=deck.slide_count,
        slide_body="\n".join(body)[:_MAX_SLIDE_CHARS],
        evidence=_evidence_block(evidence),
    )


def _default_llm(prompt: str) -> str:
    from core.llm import llm

    response = llm.invoke(prompt, capability="batch_report")
    return str(getattr(response, "content", response)).strip()


def explain_slide(
    deck: Deck,
    index: int,
    *,
    retrieve_fn: Optional[Callable] = None,
    llm_fn: Optional[Callable] = None,
    vision_fn: Optional[Callable] = None,
    interpret_images: bool = True,
    generate: bool = True,
    top_k: int = 6,
) -> Dict[str, Any]:
    """Explain one slide (1-based *index*).

    ``generate=False`` returns the extracted slide plus its evidence without touching the
    model, so extraction and grounding stay usable when the model is unavailable.
    ``interpret_images=False`` (or an unreachable vision model) leaves the pictures
    counted but undescribed, and never blocks the text explanation.
    """
    if not deck.slides:
        raise PptExplainerError("The presentation contains no slides.")
    if not 1 <= int(index) <= deck.slide_count:
        raise PptExplainerError(f"Slide {index} is out of range (1-{deck.slide_count}).")

    slide = deck.slides[int(index) - 1]
    evidence = gather_evidence(slide, retrieve_fn=retrieve_fn, top_k=top_k)
    relevance = evidence_relevance(slide, evidence)
    images = (interpret_slide_images(slide, vision_fn=vision_fn)
              if interpret_images and slide.pictures else
              {"available": bool(slide.pictures), "analysed": 0,
               "skipped": len(slide.pictures), "observations": [], "error": None,
               "kind": "visual_observation", "verified": False})

    result: Dict[str, Any] = {
        "slide_number": slide.number,
        "title": slide.title,
        "label": slide.display_label(),
        "slide_text": slide.text,
        "notes": slide.notes,
        "tables": slide.tables,
        "picture_count": slide.picture_count,
        "images": images,
        "warnings": list(slide.warnings),
        "empty": slide.is_empty,
        "evidence": evidence,
        "citations": [item["citation"] for item in evidence],
        "relevance": relevance,
        "coverage": relevance["label"],
        "explanation": None,
        "model_error": None,
    }

    if not generate:
        result["coverage"] = "not_requested"
        return result

    prompt = build_slide_prompt(deck, slide, evidence, images)
    try:
        explanation = (llm_fn or _default_llm)(prompt)
    except Exception as error:
        # The failure type only: never echo the prompt or the slide text back out.
        result["model_error"] = type(error).__name__
        result["explanation"] = (
            "Slide content was extracted successfully, but the explanation model is "
            f"currently unavailable ({type(error).__name__}). The extracted slide and any "
            "retrieved references are shown above/below; retry when the model is reachable."
        )
        return result

    result["explanation"] = str(explanation or "").strip() or NOT_COVERED
    return result


DECK_SUMMARY_PROMPT = """You are a senior Siemens Tessent DFT trainer. Below are the
per-slide explanations of one presentation (already grounded in the reference context).

Write a short deck-level synthesis with these headings:
### Presentation overview
### Learning path through the deck
### Gaps and unverified content

Rules: do not introduce new technical claims, commands or citations that are not already
present in the per-slide explanations. If a section would need evidence that is absent,
say: "{not_covered}"
"""


def synthesise_deck(deck: Deck, slide_results: List[Dict[str, Any]], llm_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """Deck-level synthesis from the per-slide explanations (no new claims)."""
    explained = [r for r in slide_results if r.get("explanation")]
    if not explained:
        return {"synthesis": None, "model_error": None, "explained": 0}

    lines = []
    for item in explained:
        lines.append(f"Slide {item['slide_number']}: {item.get('title') or '(untitled)'}")
        lines.append(str(item["explanation"])[:_MAX_SLIDE_CHARS])
    prompt = DECK_SUMMARY_PROMPT.format(not_covered=NOT_COVERED) + "\n\n" + "\n\n".join(lines)

    try:
        synthesis = (llm_fn or _default_llm)(prompt)
    except Exception as error:
        return {"synthesis": None, "model_error": type(error).__name__, "explained": len(explained)}
    return {
        "synthesis": str(synthesis or "").strip() or None,
        "model_error": None,
        "explained": len(explained),
    }


def estimate_deck_usage(deck: Deck, interpret_images: bool = True) -> Dict[str, Any]:
    """Planned work for a deck *before* any call (F3: estimates are never cost figures)."""
    pictures = sum(len(slide.pictures) for slide in deck.slides)
    estimated_vision = pictures if interpret_images else 0
    return {
        "slides": deck.slide_count,
        "estimated_explanation_calls": deck.slide_count + 1,  # per slide + synthesis
        "estimated_vision_calls": estimated_vision,
        "estimated_total_calls": deck.slide_count + 1 + estimated_vision,
        "pictures": pictures,
        "note": "Estimates from the deck structure only; no provider pricing is applied.",
    }


def _budget_snapshot() -> Optional[Dict[str, Any]]:
    try:
        from core.model_routing import budget_status

        return budget_status()
    except Exception:
        return None


def explain_deck(
    deck: Deck,
    *,
    retrieve_fn: Optional[Callable] = None,
    llm_fn: Optional[Callable] = None,
    vision_fn: Optional[Callable] = None,
    interpret_images: bool = True,
    generate: bool = True,
    top_k: int = 6,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Explain every slide in order, then synthesise. Never skips a slide silently."""
    results = []
    total = deck.slide_count
    for slide in deck.slides:
        results.append(
            explain_slide(
                deck,
                slide.number,
                retrieve_fn=retrieve_fn,
                llm_fn=llm_fn,
                vision_fn=vision_fn,
                interpret_images=interpret_images,
                generate=generate,
                top_k=top_k,
            )
        )
        if on_progress is not None:
            try:
                on_progress(slide.number, total)
            except Exception:
                pass

    payload = {
        "slide_count": total,
        "explained_count": sum(1 for r in results if r.get("explanation")),
        "slides": results,
        "warnings": list(deck.warnings),
        "synthesis": None,
        "synthesis_error": None,
    }
    if generate:
        synthesis = synthesise_deck(deck, results, llm_fn=llm_fn)
        payload["synthesis"] = synthesis["synthesis"]
        payload["synthesis_error"] = synthesis["model_error"]

    payload["usage"] = _usage_report(deck, results, generate, interpret_images)
    return payload


def _usage_report(deck: Deck, results: List[Dict[str, Any]], generate: bool,
                  interpret_images: bool) -> Dict[str, Any]:
    """Measured-versus-estimated accounting for one deck run (F3).

    Only counts are reported. Token counts are shown only when a caller supplies them
    (the default chat path does not return usage metadata), and no price is invented.
    """
    estimate = estimate_deck_usage(deck, interpret_images=interpret_images)
    vision_calls = sum(int(r.get("images", {}).get("analysed") or 0) for r in results)
    explanation_calls = sum(1 for r in results if r.get("explanation")) if generate else 0
    if generate and any(r.get("explanation") for r in results):
        explanation_calls += 1  # deck synthesis
    return {
        "estimated": estimate,
        "actual": {
            "explanation_calls": explanation_calls,
            "vision_calls": vision_calls,
            "failed_vision_calls": sum(1 for r in results if r.get("images", {}).get("error")),
            "tokens": None,
            "token_note": "Provider usage metadata is not returned by this call path; "
                          "no token or price figure is estimated.",
        },
        "budget": _budget_snapshot(),
        "calls_avoided": max(0, estimate["estimated_vision_calls"] - vision_calls),
    }
