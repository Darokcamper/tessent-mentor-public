"""Fresher assessment mode for uploaded decks (WP-11).

The explainer answers "what does this slide mean to an engineer". A fresher assessment
needs the opposite shape: what a new hire must be able to state, the mistake they usually
make, and a question with an answer a reviewer can mark. This module turns the same
grounded pipeline into that shape:

* :func:`recommended_scope` - the conceptual slides, with LAB exercise slides skipped and a
  balanced spread across the deck's flows (scan insertion -> ATPG -> patterns -> at-speed/OCC
  -> design rules), so coverage can be stated honestly instead of "all 115 slides";
* :func:`assess_slide` - one slide: three must-knows, one misconception, one check question
  with its expected answer, and the manual pages behind them;
* :func:`build_quiz` - a deck-level quiz drawn from those assessments, never from new claims.

Grounding is unchanged: only retrieved chunks may be cited, an uncovered topic is labelled,
and picture descriptions stay unverified. Answers are requested as JSON so the handout can be
structured, but a malformed answer degrades to readable text instead of failing the run.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

from core.ppt_explainer import (
    Deck,
    Slide,
    _default_llm,
    _default_retrieve,
    _evidence_block,
    _MAX_SLIDE_CHARS,
    gather_evidence,
    interpret_slide_images,
    visual_query,
)

NOT_COVERED = (
    "The available knowledge base does not contain sufficient supporting information "
    "for this part of the presentation."
)

_LAB_RE = re.compile(r"\bLAB\s*\d+|\bEX\s*\d+\b|\bexercise\b", re.IGNORECASE)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# Flow buckets used only to spread a recommended scope across the deck evenly.
# Order matters: the most specific flow is tested first, so "At-Speed Patterns and OCC"
# is classed as at-speed work and not as the generic pattern bucket.
FLOW_BUCKETS: List[Dict[str, Any]] = [
    {"name": "DFT basics and scan insertion",
     "pattern": r"dft|motivation|basic concept|scan (test|insertion|chain|mode)|test insertion"},
    {"name": "At-speed patterns and OCC",
     "pattern": r"at[- ]speed|occ\b|transition|path delay|launch|iddq"},
    {"name": "Design rule checks",
     "pattern": r"design rule|\bdrc\b|\bc[1-9]\b|\bd[1-9]\b"},
    {"name": "Faults, coverage and transcripts",
     "pattern": r"fault|coverage|transcript|logfile"},
    {"name": "Pattern types and management",
     "pattern": r"pattern|clock sequential|multiple load|mask|transl|annotation"},
    {"name": "ATPG setup and flow",
     "pattern": r"atpg|dft library|black box|procedure|dofile|tsdb|flow"},
]


def is_lab_slide(slide: Slide) -> bool:
    """True for LAB / EX exercise slides - they are workbook steps, not teachable content."""
    if _LAB_RE.search(str(slide.title or "")):
        return True
    head = "\n".join(str(slide.text or "").splitlines()[:4])
    return bool(_LAB_RE.search(head))


def _flow_of(slide: Slide) -> int:
    haystack = f"{slide.title}\n{slide.text[:600]}".lower()
    for position, bucket in enumerate(FLOW_BUCKETS):
        if re.search(bucket["pattern"], haystack):
            return position
    return -1


def recommended_scope(deck: Deck, limit: int = 22) -> Dict[str, Any]:
    """A defensible teaching subset: no LAB slides, spread across the deck's flows.

    Slides are taken in deck order within each flow and the flows are interleaved, so the
    selection covers the whole arc of the course instead of clustering in the first third.
    """
    candidates = [slide for slide in deck.slides if not slide.is_empty and not is_lab_slide(slide)]
    skipped_labs = [slide.number for slide in deck.slides if is_lab_slide(slide)]

    buckets: Dict[int, List[Slide]] = {}
    unclassified: List[Slide] = []
    for slide in candidates:
        flow = _flow_of(slide)
        if flow < 0:
            unclassified.append(slide)
        else:
            buckets.setdefault(flow, []).append(slide)

    chosen: List[Slide] = []
    cursor = 0
    while len(chosen) < limit and any(buckets.get(key) for key in buckets):
        order = sorted(buckets)
        progressed = False
        for key in order:
            queue = buckets.get(key) or []
            if cursor < len(queue) and len(chosen) < limit:
                chosen.append(queue[cursor])
                progressed = True
        cursor += 1
        if not progressed:
            break
    for slide in unclassified:
        if len(chosen) >= limit:
            break
        chosen.append(slide)

    chosen_numbers = sorted({slide.number for slide in chosen})
    return {
        "slides": chosen_numbers,
        "limit": limit,
        "candidates": len(candidates),
        "skipped_labs": skipped_labs,
        "flows": [
            {"name": bucket["name"],
             "slides": [s.number for s in (buckets.get(position) or []) if s.number in chosen_numbers]}
            for position, bucket in enumerate(FLOW_BUCKETS)
        ],
        "note": (f"{len(chosen_numbers)} of {deck.slide_count} slide(s); "
                 f"{len(skipped_labs)} LAB slide(s) skipped."),
    }


ASSESS_PROMPT = """You are preparing a trainer's assessment sheet for a fresher engineer,
based on ONE slide of a Tessent / DFT training presentation.

GROUNDING RULES (MANDATORY):
1. "SLIDE CONTENT" is the presentation. Treat it as the subject, never as a source of facts.
2. "REFERENCE CONTEXT" is the only permitted source for technical statements. Where a
   statement comes from a page, cite it as [Source: <file>, Page <N>] using a file and page
   that literally appear in the reference context.
3. Never invent a command, option, value, citation or page number.
4. Write only about the slide and Tessent. Never mention the reference context, the prompt,
   picture file names or what is missing from the sources.
5. Keep every sentence short enough for a fresher to repeat it from memory.

SLIDE CONTENT (slide {number} of {total}):
{slide_body}

REFERENCE CONTEXT:
{evidence}

Reply with ONE JSON object and nothing else, in exactly this shape:
{{
  "must_know": ["<3 short statements the fresher must be able to say>", "...", "..."],
  "misconception": "<the mistake freshers usually make on this topic, one or two sentences>",
  "check_question": "<one question that tests understanding, not recall of the slide title>",
  "expected_answer": "<the answer, 1-3 sentences, citing pages where applicable>",
  "manual_refs": ["[Source: <file>, Page <N>]", "..."],
  "not_covered": "<anything you could not verify, or the single word None>"
}}
"""

QUIZ_PROMPT = """You are writing the end-of-session quiz for a fresher assessment on the
Tessent scan/ATPG flow.

Below are per-slide assessment notes that were already grounded in the reference context.

Write {count} questions as ONE JSON object, nothing else:
{{
  "questions": [
    {{"question": "...", "answer": "...", "slide": <slide number the question comes from>}},
    ...
  ]
}}

Rules: no new technical claims beyond the notes; mix recall, understanding and "what would
you see in the transcript" style; keep each answer to one or two sentences.

PER-SLIDE NOTES:
{notes}
"""

ASSESS_FIELDS = ("must_know", "misconception", "check_question", "expected_answer",
                 "manual_refs", "not_covered")


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """First JSON object in a model reply, tolerating prose or code fences around it."""
    body = str(text or "").strip()
    body = re.sub(r"^```(?:json)?|```$", "", body, flags=re.MULTILINE).strip()
    match = _JSON_BLOCK_RE.search(body)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _as_list(value: Any, limit: int = 5) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out[:limit]


def normalise_assessment(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Shape a model reply into the assessment fields, filled or empty - never None."""
    data = payload or {}
    must_know = _as_list(data.get("must_know"), limit=4)
    not_covered = str(data.get("not_covered") or "").strip()
    if not_covered.lower() in {"none", "n/a", "-", ""}:
        not_covered = ""
    return {
        "must_know": must_know,
        "misconception": str(data.get("misconception") or "").strip(),
        "check_question": str(data.get("check_question") or "").strip(),
        "expected_answer": str(data.get("expected_answer") or "").strip(),
        "manual_refs": _as_list(data.get("manual_refs"), limit=8),
        "not_covered": not_covered,
    }


def build_assess_prompt(deck: Deck, slide: Slide, evidence: List[Dict[str, Any]],
                        images: Optional[Dict[str, Any]] = None) -> str:
    from core.ppt_explainer import build_slide_prompt  # reuse the same body assembly

    full = build_slide_prompt(deck, slide, evidence, images)
    body = full.split("SLIDE CONTENT", 1)[1].split("REFERENCE CONTEXT", 1)[0]
    return ASSESS_PROMPT.format(
        number=slide.number,
        total=deck.slide_count,
        slide_body=body.strip()[:_MAX_SLIDE_CHARS],
        evidence=_evidence_block(evidence),
    )


def assess_slide(
    deck: Deck,
    index: int,
    *,
    retrieve_fn: Optional[Callable] = None,
    llm_fn: Optional[Callable] = None,
    vision_fn: Optional[Callable] = None,
    interpret_images: bool = False,
    generate: bool = True,
    top_k: int = 6,
) -> Dict[str, Any]:
    """Assessment notes for one slide (1-based *index*).

    ``interpret_images`` defaults to False here: a fresher is assessed on concepts, and every
    vision call is a call the operator pays for. Passing True behaves exactly as the explainer.
    """
    from core.ppt_explainer import NOT_COVERED as _NOT_COVERED  # noqa: F401 - documents intent

    if not 1 <= int(index) <= deck.slide_count:
        raise ValueError(f"Slide {index} is out of range (1-{deck.slide_count}).")
    slide = deck.slides[int(index) - 1]

    images = (interpret_slide_images(slide, vision_fn=vision_fn)
              if interpret_images and slide.pictures else
              {"available": False, "analysed": 0, "skipped": len(slide.pictures),
               "observations": [], "error": None, "kind": "visual_observation",
               "verified": False})
    picture_terms = visual_query(images)
    evidence = gather_evidence(slide, retrieve_fn=retrieve_fn, top_k=top_k,
                               extra_query=picture_terms)

    result: Dict[str, Any] = {
        "slide_number": slide.number,
        "label": slide.display_label(),
        "title": slide.title,
        "is_lab": is_lab_slide(slide),
        "evidence": evidence,
        "citations": [item["citation"] for item in evidence],
        "must_know": [],
        "misconception": "",
        "check_question": "",
        "expected_answer": "",
        "manual_refs": [],
        "not_covered": "",
        "citation_check": None,
        "raw_text": None,
        "parse_error": None,
        "model_error": None,
    }
    if not generate:
        return result

    prompt = build_assess_prompt(deck, slide, evidence, images)
    try:
        reply = (llm_fn or _default_llm)(prompt)
    except Exception as error:
        result["model_error"] = type(error).__name__
        result["not_covered"] = NOT_COVERED
        return result

    payload = _extract_json(reply)
    if payload is None:
        # Keep the run alive and the author informed: unparsed text is still useful to a
        # trainer, and it is labelled rather than silently dropped.
        result["raw_text"] = str(reply or "").strip()
        result["parse_error"] = "the model did not return the requested JSON"
        return result

    result.update(normalise_assessment(payload))
    if not result["must_know"] and not result["check_question"]:
        result["parse_error"] = "the model returned JSON without the assessment fields"
        result["raw_text"] = str(reply or "").strip()

    # Every reference the notes quote is checked against the page that holds it. The claim
    # text is the whole note, not one sentence: a paraphrase matches more of the page.
    try:
        from core.citation_check import verify_text

        claim = " ".join(
            [str(item) for item in (result["must_know"] or [])]
            + [result["misconception"] or "", result["check_question"] or "",
               result["expected_answer"] or ""]
        ).strip()
        report = verify_text((claim + " " + " ".join(result["manual_refs"])).strip())
        result["citation_check"] = report
        if report.get("changed"):
            result["expected_answer"] = report["corrected_text"]
    except Exception:
        result["citation_check"] = None
    return result


def _notes_block(assessments: List[Dict[str, Any]]) -> str:
    lines = []
    for item in assessments:
        lines.append(f"Slide {item['slide_number']}: {item.get('title') or '(untitled)'}")
        for point in item.get("must_know") or []:
            lines.append(f"- must know: {point}")
        if item.get("misconception"):
            lines.append(f"- typical mistake: {item['misconception']}")
        if item.get("check_question"):
            lines.append(f"- question: {item['check_question']}")
        if item.get("expected_answer"):
            lines.append(f"- answer: {item['expected_answer']}")
        if item.get("not_covered"):
            lines.append(f"- not covered: {item['not_covered']}")
        lines.append("")
    return "\n".join(lines)


def build_quiz(assessments: List[Dict[str, Any]], llm_fn: Optional[Callable] = None,
               count: int = 10) -> Dict[str, Any]:
    """Deck-level quiz drawn only from the per-slide notes. Never raises."""
    usable = [item for item in assessments if item.get("check_question")]
    if not usable:
        return {"questions": [], "model_error": None, "skipped": "no assessed slides"}
    prompt = QUIZ_PROMPT.format(count=count, notes=_notes_block(usable)[:20000])
    try:
        reply = (llm_fn or _default_llm)(prompt)
    except Exception as error:
        return {"questions": [], "model_error": type(error).__name__, "skipped": None}

    payload = _extract_json(reply) or {}
    raw_questions = payload.get("questions")
    questions: List[Dict[str, Any]] = []
    if isinstance(raw_questions, list):
        for entry in raw_questions[:max(1, count)]:
            if not isinstance(entry, dict):
                continue
            question = str(entry.get("question") or "").strip()
            answer = str(entry.get("answer") or "").strip()
            if not question or not answer:
                continue
            slide_number = entry.get("slide")
            try:
                slide_number = int(slide_number)
            except (TypeError, ValueError):
                slide_number = None
            questions.append({"question": question, "answer": answer, "slide": slide_number})

    result = {"questions": questions, "model_error": None, "skipped": None}
    if not questions:
        result["skipped"] = "the model did not return usable questions"
        result["raw_text"] = str(reply or "").strip()[:4000]
    return result


def assessment_coverage(deck: Deck, assessments: List[Dict[str, Any]],
                        scope: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """What the sheet covers, stated plainly for a reviewer."""
    assessed = [item for item in assessments if item.get("must_know") or item.get("check_question")]
    labs = [item["slide_number"] for item in assessments if item.get("is_lab")]
    gaps = [{"slide": item["slide_number"], "title": item.get("title"),
             "reason": item.get("not_covered")} for item in assessments if item.get("not_covered")]
    return {
        "deck_slides": deck.slide_count,
        "selected": len(assessments),
        "assessed": len(assessed),
        "lab_slides_in_scope": labs,
        "slides_with_gaps": gaps,
        "flows": (scope or {}).get("flows") or [],
        "statement": (f"{len(assessed)} of {deck.slide_count} slide(s) assessed"
                      + (f"; {len(labs)} LAB slide(s) inside the scope" if labs else "")
                      + (f"; {len(gaps)} slide(s) carry an explicit knowledge-base gap" if gaps else "")),
    }


def assess_deck(
    deck: Deck,
    *,
    slides: Optional[Iterable[int]] = None,
    retrieve_fn: Optional[Callable] = None,
    llm_fn: Optional[Callable] = None,
    vision_fn: Optional[Callable] = None,
    interpret_images: bool = False,
    generate: bool = True,
    top_k: int = 6,
    quiz_count: int = 10,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Run the assessment over the selected slides, then build the quiz.

    ``slides=None`` uses :func:`recommended_scope` rather than the whole deck: a fresher sheet
    over 115 slides including 27 LAB exercises is not a credible assessment, and the default
    should be the shape a reviewer expects.
    """
    scope = None
    if slides is None:
        scope = recommended_scope(deck)
        selected = list(scope["slides"])
    else:
        selected = []
        for number in slides:
            try:
                value = int(number)
            except (TypeError, ValueError):
                continue
            if 1 <= value <= deck.slide_count and value not in selected:
                selected.append(value)
        if not selected:
            raise ValueError(f"No valid slide in the selection (deck has {deck.slide_count}).")

    assessments: List[Dict[str, Any]] = []
    total = len(selected)
    for position, number in enumerate(selected, start=1):
        assessments.append(assess_slide(
            deck, number,
            retrieve_fn=retrieve_fn, llm_fn=llm_fn, vision_fn=vision_fn,
            interpret_images=interpret_images, generate=generate, top_k=top_k,
        ))
        if on_progress is not None:
            try:
                on_progress(position, total)
            except Exception:
                pass

    quiz = (build_quiz(assessments, llm_fn=llm_fn, count=quiz_count)
            if generate and quiz_count > 0 else
            {"questions": [], "model_error": None, "skipped": "quiz not requested"})
    return {
        "deck_slides": deck.slide_count,
        "selected_slides": selected,
        "assessments": assessments,
        "quiz": quiz,
        "scope": scope,
        "coverage": assessment_coverage(deck, assessments, scope),
        "usage": {
            "assessment_calls": total if generate else 0,
            "quiz_calls": 1 if quiz.get("questions") or quiz.get("raw_text") else 0,
            "vision_calls": sum(int((item.get("images") or {}).get("analysed") or 0)
                                for item in assessments),
        },
    }


