"""Offline, evidence-gated support for the real-question exam bank."""

import csv
import json
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAM_FILE = PROJECT_ROOT / "_exam_parsed.json"
SOURCE_CATALOG = PROJECT_ROOT / "knowledge" / "sources.csv"
CANONICAL_TAXONOMY = PROJECT_ROOT / "knowledge" / "canonical_taxonomy.json"
ALLOWED_QUESTION_TYPES = {"command syntax", "conceptual", "procedural", "comparison", "troubleshooting", "MCQ"}
NON_VERIFIED_STATUSES = {"unverified", "needs_review", "unsupported"}
STOP_WORDS = {"a", "an", "are", "as", "at", "for", "how", "in", "is", "of", "the", "to", "what", "why"}

# Prior per-question topic grouping, preserved verbatim from the pre-evidence-gating
# implementation of this module. Its own provenance was "exam order and content", not
# corpus evidence, so it is kept strictly as a provisional review grouping: it is never
# written into a record's `canonical` mapping, never resolves in the approved taxonomy,
# and never affects verification or scoring. Questions 49 and 50 were never mapped by
# that implementation and stay unmapped here.
QUESTION_TOPICS = {
    1: "ATPG Flow", 2: "ATPG Flow", 3: "Scan Insertion", 4: "Scan Insertion",
    5: "Scan Insertion", 6: "ATPG Flow", 7: "ATPG Flow", 8: "Scan Insertion",
    9: "Scan Insertion", 10: "ATPG Flow", 11: "ATPG Flow", 12: "ATPG Flow",
    13: "Fault Coverage", 14: "Fault Coverage", 15: "ATPG Flow", 16: "Scan Insertion",
    17: "ATPG Flow", 18: "ATPG Flow", 19: "ATPG Flow", 20: "ATPG Flow",
    21: "Scan Insertion", 22: "ATPG Flow", 23: "ATPG Flow", 24: "ATPG Flow",
    25: "Fault Coverage", 26: "ATPG Flow", 27: "ATPG Flow", 28: "ATPG Flow",
    29: "ATPG Flow", 30: "ATPG Flow", 31: "Scan Insertion", 32: "ATPG Flow",
    33: "ATPG Flow", 34: "ATPG Flow", 35: "ATPG Flow", 36: "ATPG Flow",
    37: "ATPG Flow", 38: "ATPG Flow", 39: "ATPG Flow", 40: "ATPG Flow",
    41: "ATPG Flow", 42: "ATPG Flow", 43: "ATPG Flow", 44: "ATPG Flow",
    45: "ATPG Flow", 46: "ATPG Flow", 47: "ATPG Flow", 48: "ATPG Flow",
}

EXAM_SOURCE_INDEX_CACHE = {}


def clean_text(text):
    """Strip portal header noise without altering the underlying source record."""
    text = str(text or "")
    position, marker = max((text.rfind(item), item) for item in ("Core Topics", "ASSESSMENT"))
    if position > 0:
        text = text[position + len(marker):]
    return re.sub(r"^\d+\s*", "", text.strip())


def _normalized(value):
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _claim_terms(value):
    return {token for token in re.findall(r"[a-z0-9_]+", _normalized(value)) if len(token) > 1 and token not in STOP_WORDS}


def _catalog_entries(source_catalog):
    path = Path(source_catalog)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {str(row.get("ID", "")).strip(): row for row in csv.DictReader(handle) if str(row.get("ID", "")).strip()}


def _canonical_nodes(canonical_taxonomy):
    path = Path(canonical_taxonomy)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(json.load(handle).get("nodes") or [])


def _page_blocks(content):
    """Split a corpus text into {page_number: page_text} using its page markers."""
    blocks = {}
    markers = list(re.finditer(r"--- PAGE (\d+) ---", content))
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(content)
        blocks.setdefault(int(marker.group(1)), content[marker.start():end])
    return blocks


def _page_screenshots(page_text):
    """Return the screenshot basenames a corpus page was rendered from."""
    return [
        match.group(1).strip().replace("\\", "/").rsplit("/", 1)[-1]
        for match in re.finditer(r"\[SOURCE:\s*([^\]]*?)\]", page_text)
    ]


def _exam_source_index(corpus_root, catalog, canonical_nodes):
    """Index every approved exam-bank corpus page by its source screenshot.

    Only taxonomy nodes whose ``content_type`` is ``exam_quiz`` and whose
    ``source_evidence`` names a catalog entry are indexed, so provenance can only be
    established from approved corpus files.
    """
    sources = []
    for node in canonical_nodes:
        if node.get("content_type") != "exam_quiz":
            continue
        evidence = node.get("source_evidence") or {}
        catalog_id = evidence.get("source_catalog_id")
        source_path = evidence.get("file")
        if catalog_id is None or not source_path:
            continue
        entry = catalog.get(str(catalog_id))
        if not entry:
            continue
        resolved = _safe_source_path(corpus_root, source_path)
        if not resolved or not resolved.is_file():
            continue
        sources.append((node, catalog_id, source_path, resolved))
    key = (
        str(corpus_root),
        _canonical_taxonomy_signature(canonical_nodes),
        tuple(
            (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for _, _, _, path in sources
        ),
    )
    cached = EXAM_SOURCE_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    index = {}
    for node, catalog_id, source_path, resolved in sources:
        content = resolved.read_text(encoding="utf-8", errors="ignore")
        for page, page_text in _page_blocks(content).items():
            for screenshot in _page_screenshots(page_text):
                index.setdefault(screenshot, []).append({
                    "source_catalog_id": catalog_id,
                    "source_path": str(source_path).replace("\\", "/"),
                    "page": page,
                    "node": node,
                    "page_text": page_text,
                })
    EXAM_SOURCE_INDEX_CACHE[key] = index
    return index


def _canonical_taxonomy_signature(canonical_nodes):
    """Return a stable signature for a taxonomy node list (used as a cache key)."""
    return "|".join(
        f"{node.get('module_domain')}/{node.get('topic')}/{node.get('chapter')}/"
        f"{json.dumps(node.get('source_evidence') or {}, sort_keys=True)}"
        for node in canonical_nodes
    )


def _exam_page_section(page_text):
    """Return the exam section title printed on a corpus page, if any."""
    lines = page_text.splitlines()
    for index, line in enumerate(lines):
        match = re.search(r"(?i)level\s*\d*\s*exam\s*-?\s*(.+)$", line)
        if not match:
            continue
        title = match.group(1).strip(" -|")
        if not title or len(title) < 4:
            continue
        if index + 1 < len(lines):
            following = lines[index + 1].strip()
            if following and len(following) <= 40 and not re.search(
                r"(?i)exam|assessment|question\s*\d|chapter resources|^correct$|^incorrect$", following
            ):
                title = f"{title} {following}"
        return title
    return None


def _exam_question_region(page_text, number, options):
    """Return the question text region printed on a corpus page, if it resolves."""
    lines = [line.strip() for line in page_text.splitlines()]
    marker = next(
        (index for index, line in enumerate(lines) if re.search(r"(?i)q\w*\s*%d\b" % number, line)),
        None,
    )
    if marker is None:
        return None
    end = len(lines)
    for index in range(marker + 1, len(lines)):
        line = lines[index]
        if not line:
            continue
        if line.startswith("O ") or any(
            _normalized(option) and _normalized(option) in _normalized(line) for option in options
        ):
            end = index
            break
    region = " ".join(line for line in lines[marker + 1:end] if line).strip()
    return region or None


def ground_legacy_provenance(raw, corpus_root, catalog, canonical_nodes):
    """Resolve a legacy record's provenance against the approved exam-bank corpus.

    A legacy record is only grounded when its own source screenshot resolves to a
    single approved exam corpus page that also prints the question and an exam
    section title. Grounding establishes *where the record came from* only; it never
    establishes that the stored answer key is correct, and it never verifies a record.
    """
    screenshot = str(raw.get("source_screenshot") or "").strip()
    if not screenshot:
        return None
    index = _exam_source_index(corpus_root, catalog, canonical_nodes)
    candidates = index.get(screenshot) or []
    if not candidates:
        return None
    text = str(raw.get("text") or "")
    claim_terms = _claim_terms(text)
    options = list(raw.get("options") or [])
    scored = []
    for candidate in candidates:
        overlap = len(claim_terms & _claim_terms(candidate["page_text"]))
        scored.append((overlap, candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: -item[0])
    best_overlap, best = scored[0]
    if best_overlap < min(3, len(claim_terms)):
        return None
    if len(scored) > 1 and scored[1][0] == best_overlap:
        return None
    section = _exam_page_section(best["page_text"])
    region = _exam_question_region(best["page_text"], raw.get("num") or 0, options)
    if not section or not region:
        return None
    if _normalized(section) not in _normalized(best["page_text"]):
        return None
    if _normalized(region) not in _normalized(best["page_text"]):
        return None
    if len(claim_terms & _claim_terms(region)) < min(3, len(claim_terms)):
        return None
    node = best["node"]
    return {
        "source_catalog_id": best["source_catalog_id"],
        "source_path": best["source_path"],
        "page": best["page"],
        "section": section,
        "excerpt": region,
        "canonical": {
            "module": node.get("module_domain"),
            "topic": node.get("topic"),
            "chapter": node.get("chapter"),
        },
    }


def _safe_source_path(corpus_root, source_path):
    try:
        root = Path(corpus_root).resolve()
        candidate = (root / str(source_path)).resolve()
        candidate.relative_to(root)
        return candidate
    except (OSError, ValueError):
        return None


def _page_content(content, page):
    marker = f"--- PAGE {page} ---"
    start = content.find(marker)
    if start < 0:
        return None
    following = re.search(r"\n--- PAGE \d+ ---", content[start + len(marker):])
    end = start + len(marker) + following.start() if following else len(content)
    return content[start:end]


def _evidence_reasons(evidence, claim, label, corpus_root, catalog):
    if not isinstance(evidence, dict):
        return [f"missing {label} evidence"]
    required = ("source_catalog_id", "source_path", "page", "section", "excerpt")
    missing = [field for field in required if not evidence.get(field)]
    if missing:
        return [f"missing {label} evidence fields: {', '.join(missing)}"]
    catalog_id = str(evidence["source_catalog_id"])
    catalog_entry = catalog.get(catalog_id)
    if not catalog_entry:
        return [f"{label} evidence has unknown source catalog id {catalog_id}"]
    source_path = _safe_source_path(corpus_root, evidence["source_path"])
    if not source_path or not source_path.is_file():
        return [f"{label} evidence source path is unavailable"]
    if source_path.name != str(catalog_entry.get("File", "")).strip():
        return [f"{label} evidence source path does not match source catalog"]
    page = evidence["page"]
    if not isinstance(page, int) or page <= 0:
        return [f"{label} evidence has invalid page"]
    page_content = _page_content(source_path.read_text(encoding="utf-8", errors="ignore"), page)
    if page_content is None:
        return [f"{label} evidence page does not resolve in source"]
    if _normalized(evidence["section"]) not in _normalized(page_content):
        return [f"{label} evidence section does not resolve in source"]
    excerpt = _normalized(evidence["excerpt"])
    if excerpt not in _normalized(page_content):
        return [f"{label} evidence excerpt does not resolve in source"]
    if label == "answer":
        if _normalized(claim) not in excerpt:
            return ["answer evidence does not support the answer key"]
    else:
        terms = _claim_terms(claim)
        if len(terms.intersection(_claim_terms(evidence["excerpt"]))) < min(3, len(terms)):
            return ["question evidence does not support the question text"]
    return []


def _evidence_text_is_supported(text, record):
    if not isinstance(text, str) or not text.strip():
        return False
    excerpts = [
        _normalized(record[field].get("excerpt"))
        for field in ("question_source", "answer_source")
        if isinstance(record.get(field), dict)
    ]
    if not excerpts:
        return False

    # Split text into sentences to verify each claim independently.
    # We use a simple split on punctuation followed by space.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return False

    # An explanation is supported if every sentence is supported by at least one excerpt.
    # We use normalized substring matching for each sentence.
    for sentence in sentences:
        normalized_sentence = _normalized(sentence)
        if not any(normalized_sentence in excerpt for excerpt in excerpts):
            return False
    return True


def _legacy_record(raw, provenance=None):
    try:
        number = int(raw.get("num"))
    except (TypeError, ValueError):
        number = 0
    record = {
        "question_id": f"exam-{number:03d}" if number else "exam-unknown",
        "num": number,
        "text": clean_text(raw.get("text")),
        "options": list(raw.get("options") or []),
        "correct": list(raw.get("correct") or []),
        "validation_status": "needs_review" if raw.get("needs_review") else "unverified",
        "validation_reasons": ["No approved corpus evidence mapping is available for this legacy record."],
    }
    if raw.get("source_screenshot"):
        record["source_screenshot"] = raw["source_screenshot"]
    if provenance:
        # Provenance is grounded in the approved exam corpus, but grounding only
        # establishes where the record came from. It does not establish that the
        # stored answer key is correct, so the record stays unverified and is not
        # scored until an answer-evidence mapping is separately reviewed.
        record["question_source"] = {
            "source_catalog_id": provenance["source_catalog_id"],
            "source_path": provenance["source_path"],
            "page": provenance["page"],
            "section": provenance["section"],
            "excerpt": provenance["excerpt"],
        }
        record["canonical"] = dict(provenance["canonical"])
        record["validation_reasons"] = [
            "Question provenance is grounded in the approved exam corpus, but no reviewed "
            "answer-evidence mapping exists for the stored answer key."
        ]
    return record


def _normalize_record(raw, provenance=None):
    if not raw.get("question_id") and not raw.get("question_source"):
        return _legacy_record(raw, provenance)
    record = dict(raw)
    record["text"] = clean_text(record.get("text"))
    record["options"] = list(record.get("options") or [])
    record["correct"] = list(record.get("correct") or [])
    record.setdefault("validation_status", "unverified")
    record.setdefault("validation_reasons", [])
    return record


def validate_questions(records, corpus_root=PROJECT_ROOT, source_catalog=SOURCE_CATALOG, canonical_taxonomy=CANONICAL_TAXONOMY):
    """Return copied records with deterministic, corpus-only validation results."""
    catalog = _catalog_entries(source_catalog)
    canonical_nodes = _canonical_nodes(canonical_taxonomy)
    validated = []
    for record in records:
        provenance = None
        if not record.get("question_id") and not record.get("question_source"):
            provenance = ground_legacy_provenance(record, corpus_root, catalog, canonical_nodes)
        validated.append(_normalize_record(record, provenance))
    seen_ids, seen_texts = {}, {}
    for index, record in enumerate(validated):
        reasons = list(record.get("validation_reasons") or [])
        question_id = str(record.get("question_id") or "").strip()
        text = str(record.get("text") or "").strip()
        options, answers = record.get("options"), record.get("correct")
        if not question_id:
            reasons.append("missing question_id")
        elif question_id in seen_ids:
            reasons.append("duplicate question_id")
            validated[seen_ids[question_id]]["validation_reasons"].append("duplicate question_id")
        else:
            seen_ids[question_id] = index
        normalized_text = _normalized(text)
        if not normalized_text:
            reasons.append("missing question text")
        elif normalized_text in seen_texts:
            reasons.append("duplicate question text")
            validated[seen_texts[normalized_text]]["validation_reasons"].append("duplicate question text")
        else:
            seen_texts[normalized_text] = index
        unique_options = {_normalized(option) for option in options} if isinstance(options, list) else set()
        if not isinstance(options, list) or not options or len(unique_options) != len(options):
            reasons.append("options must be present and unique")
        if not isinstance(answers, list) or not answers:
            reasons.append("missing answer key")
        elif any(_normalized(answer) not in unique_options for answer in answers):
            reasons.append("answer key is not present in options")
        if record.get("question_type") not in ALLOWED_QUESTION_TYPES:
            reasons.append("question_type is not an approved classification")
        canonical = record.get("canonical")
        if not isinstance(canonical, dict):
            reasons.append("missing canonical mapping")
        elif not any(
            node.get("module_domain") == canonical.get("module")
            and node.get("topic") == canonical.get("topic")
            and node.get("chapter") == canonical.get("chapter")
            for node in canonical_nodes
        ):
            reasons.append("canonical mapping does not resolve in approved taxonomy")
        if not _evidence_text_is_supported(record.get("question_type_evidence"), record):
            reasons.append("question_type evidence is not supported by cited evidence")
        if not _evidence_text_is_supported(record.get("difficulty_evidence"), record):
            reasons.append("difficulty evidence is not supported by cited evidence")
        reasons.extend(_evidence_reasons(record.get("question_source"), text, "question", corpus_root, catalog))
        for answer in answers if isinstance(answers, list) else []:
            reasons.extend(_evidence_reasons(record.get("answer_source"), answer, "answer", corpus_root, catalog))
        record["validation_reasons"] = list(dict.fromkeys(reasons))
    for record in validated:
        if record["validation_reasons"]:
            record["validation_status"] = "needs_review" if record.get("validation_status") == "needs_review" else "unverified"
        else:
            record["validation_status"] = "verified"
    return validated


def load_questions(exam_file=EXAM_FILE, corpus_root=PROJECT_ROOT, source_catalog=SOURCE_CATALOG, canonical_taxonomy=CANONICAL_TAXONOMY):
    """Load records without changing source JSON, then apply offline validation."""
    path = Path(exam_file)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    return validate_questions(records, corpus_root=corpus_root, source_catalog=source_catalog, canonical_taxonomy=canonical_taxonomy)


def get_filter_options(records):
    """Derive filters solely from canonical fields carried by validated records."""
    canonical = [record.get("canonical") for record in records if isinstance(record.get("canonical"), dict)]
    return {
        "modules": sorted({item.get("module") for item in canonical if item.get("module")} ),
        "topics": sorted({item.get("topic") for item in canonical if item.get("topic")} ),
        "chapters": sorted({item.get("chapter") for item in canonical if item.get("chapter")} ),
    }


def get_review_topics(records):
    """Return the preserved prior per-question review groupings for legacy records.

    These are the pre-evidence-gating topics for questions 1-48. They are review
    labels only: they are not canonical taxonomy mappings, they do not resolve in
    the approved taxonomy, and they never affect verification or scoring.
    """
    numbers = {
        record.get("num") for record in records
        if isinstance(record.get("num"), int) and record.get("num") in QUESTION_TOPICS
    }
    return sorted({QUESTION_TOPICS[number] for number in numbers})


def get_corpus_status(records):
    """Summarize how much of the bank is verified, grounded, and awaiting review."""
    verified = [record for record in records if record.get("validation_status") == "verified"]
    grounded = [
        record for record in records
        if record.get("validation_status") != "verified" and record.get("question_source")
    ]
    unmapped = [record for record in records if not record.get("question_source")]
    return {
        "total": len(records),
        "verified": len(verified),
        "grounded": len(grounded),
        "unmapped": len(unmapped),
    }


def default_include_unverified(records):
    """Whether the bank should open in the labelled review pool.

    When no record has verified answer evidence there is nothing scorable to show,
    so the mode opens on the clearly labelled review pool instead of an empty quiz.
    """
    return not any(record.get("validation_status") == "verified" for record in records)


def select_questions(records, *, module=None, topic=None, chapter=None, include_unverified=False, unclassified_only=False, count=None, seed=None, question_ids=None, review_topic=None):
    """Select unique records deterministically, with optional seeded presentation order."""
    selected = []
    for record in sorted(records, key=lambda item: str(item.get("question_id", ""))):
        canonical = record.get("canonical") if isinstance(record.get("canonical"), dict) else {}
        if not include_unverified and record.get("validation_status") != "verified":
            continue
        if unclassified_only and canonical:
            continue
        if review_topic is not None:
            number = record.get("num")
            if not isinstance(number, int) or QUESTION_TOPICS.get(number) != review_topic:
                continue
        if module and canonical.get("module") != module:
            continue
        if topic and canonical.get("topic") != topic:
            continue
        if chapter and canonical.get("chapter") != chapter:
            continue
        selected.append(record)
    if question_ids is not None:
        by_id = {record.get("question_id"): record for record in selected}
        return [by_id[item] for item in dict.fromkeys(question_ids) if item in by_id]
    if seed is not None:
        selected = selected.copy()
        random.Random(seed).shuffle(selected)
    return selected[:max(0, int(count))] if count is not None else selected


def get_questions_by_topic(topic=None, count=None, shuffle=False, seed=None, include_unverified=False):
    return select_questions(load_questions(), topic=topic, count=count, seed=seed if shuffle else None, include_unverified=include_unverified)


def get_topics():
    return get_filter_options(load_questions())["topics"]


def format_question(question, show_answer=False):
    lines = [f"**Q{question.get('num', '?')}:** {question.get('text', '')}"]
    lines.extend(f"  {index}. {option}" for index, option in enumerate(question.get("options", []), 1))
    if question.get("validation_status") != "verified":
        # Never present an unreviewed key as an answer. Say plainly that this is
        # review material, so a review-mode rendering cannot be mistaken for a
        # verified, scored question.
        lines.append("🟡 **Status:** unverified review material — not scorable, answer not shown.")
        if question.get("validation_reasons"):
            lines.append("   _Not verified: " + "; ".join(question["validation_reasons"]) + "_")
        return "\n".join(lines)
    if show_answer:
        lines.append(f"✅ **Answer:** {', '.join(question.get('correct', []))}")
    return "\n".join(lines)


def format_quiz(questions, show_answers=False):
    return "\n\n---\n\n".join(format_question(question, show_answers) for question in questions)


def _searchable_text(record):
    """Question text + options + number, lowercased (never the answer key)."""
    parts = [str(record.get("num", "")), str(record.get("text", ""))]
    parts.extend(str(option) for option in record.get("options") or [])
    return " ".join(parts).lower()


def search_questions(records, query):
    """Deterministic free-text search over question text and options (WP-7).

    Behaviour:
    * empty query            -> every record, in the caller's order
    * ``12`` / ``q12`` / ``#12`` -> the record with that number (when present)
    * otherwise              -> records containing every query term, ranked by term
      frequency, then by question number

    The answer key is never part of the searchable text, so search cannot be used to
    probe an unverified answer.
    """
    text = (query or "").strip()
    if not text:
        return list(records)

    numbered = re.fullmatch(r"(?:q|#)?\s*(\d{1,3})", text.lower())
    if numbered:
        wanted = int(numbered.group(1))
        exact = [record for record in records if record.get("num") == wanted]
        if exact:
            return exact

    terms = [term for term in re.split(r"[^a-z0-9_]+", text.lower()) if term]
    if not terms:
        return list(records)

    scored = []
    for record in records:
        haystack = _searchable_text(record)
        if all(term in haystack for term in terms):
            scored.append((sum(haystack.count(term) for term in terms), record.get("num") or 0, record))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [record for _score, _num, record in scored]


def coverage_report(records):
    """Coverage and blocking-reason summary for the bank (read-only)."""
    status = {}
    reasons = {}
    unmapped = []
    for record in records:
        state = record.get("validation_status") or "unknown"
        status[state] = status.get(state, 0) + 1
        if not record.get("question_source"):
            unmapped.append(record.get("num"))
        for reason in record.get("validation_reasons") or []:
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "total": len(records),
        "status_counts": dict(sorted(status.items())),
        "verified": status.get("verified", 0),
        "scorable": status.get("verified", 0),
        "unmapped_numbers": sorted(n for n in unmapped if isinstance(n, int)),
        "reason_counts": dict(sorted(reasons.items(), key=lambda item: -item[1])),
    }


def export_questions_markdown(records, include_answers=False, title="Tessent Level 1 Exam Question Bank"):
    """Deterministic Markdown export that can never reveal an unverified key."""
    lines = [
        f"# {title}",
        "",
        f"- Records exported: {len(records)}",
        "- Answer keys are printed only for records whose answer evidence is verified.",
        "",
    ]
    for record in records:
        verified = record.get("validation_status") == "verified"
        lines.append(f"## Q{record.get('num')}: {record.get('text', '')}")
        lines.append("")
        for index, option in enumerate(record.get("options") or [], 1):
            lines.append(f"{index}. {option}")
        lines.append("")

        source = record.get("question_source")
        if source:
            lines.append(
                f"- Question evidence: [Source: {source.get('source_path')}, "
                f"Page {source.get('page')}, Section {source.get('section')}]"
            )
        else:
            lines.append("- Question evidence: not mapped to the approved corpus.")

        if verified and include_answers:
            lines.append(f"- **Answer:** {', '.join(record.get('correct') or [])}")
            answer_source = record.get("answer_source")
            if answer_source:
                lines.append(
                    f"- Answer evidence: [Source: {answer_source.get('source_path')}, "
                    f"Page {answer_source.get('page')}, Section {answer_source.get('section')}]"
                )
        elif verified:
            lines.append("- Status: verified (answers omitted; re-export with answers enabled to include them).")
        else:
            lines.append("- Status: 🟡 review material — not scorable, answer not shown.")
            for reason in record.get("validation_reasons") or []:
                lines.append(f"  - Not verified: {reason}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def evaluate_answer(question, user_answer):
    """Score only verified evidence-backed keys; never reveal unverified keys."""
    if question.get("validation_status") != "verified":
        return {"correct": False, "scorable": False, "correct_answer": [], "feedback": "This question's answer is not verified against the approved corpus and cannot be scored.", "answer_source": None}
    answers, options = question.get("correct") or [], question.get("options") or []
    if any(_normalized(answer) not in {_normalized(option) for option in options} for answer in answers):
        return {"correct": False, "scorable": False, "correct_answer": [], "feedback": "This question has an inconsistent answer key and cannot be scored.", "answer_source": None}
    supplied = user_answer if isinstance(user_answer, (list, tuple, set)) else [user_answer]
    correct = {_normalized(answer) for answer in supplied if str(answer).strip()} == {_normalized(answer) for answer in answers}
    return {
        "correct": correct,
        "scorable": True,
        "correct_answer": list(answers),
        "feedback": "✅ Correct!" if correct else f"❌ Incorrect. The correct answer is: **{', '.join(answers)}**",
        "answer_source": question.get("answer_source"),
        "explanation": question.get("explanation") if _evidence_text_is_supported(question.get("explanation"), question) else None,
    }
