"""Deep Study Agent - detailed per-video and per-lab study reports.

Grounded in the OCR-ed video transcripts and slide text under
knowledge/txt/atpg_video_*, plus RAG retrieval from the Tessent manuals,
with citations.
"""
from pathlib import Path

from core.llm import llm
from core.rag_builder import retrieve
from core.session_logger import log_deep_study

KNOWLEDGE_TXT_DIR = Path("knowledge") / "txt"
SUB_PREFIX = "atpg_video_subs__"
SLIDE_PREFIX = "atpg_video_slides__"
LAB_SLIDES_FILE = "atpg_video_slides__10_labs.txt"


def _read_text(path, cap=None):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    if cap and len(text) > cap:
        half = cap // 2
        text = text[:half] + "\n\n[... middle omitted ...]\n\n" + text[-half:]
    return text


def _module_name(fname):
    stem = fname[len(SUB_PREFIX):-4]
    num, _, title = stem.partition("_")
    if num.isdigit():
        return int(num), title.replace("_", " ").title()
    return None, stem.replace("_", " ").title()


def list_video_modules():
    """Returns {num: display} for every video transcript found."""
    mods = {}
    if KNOWLEDGE_TXT_DIR.exists():
        for p in sorted(KNOWLEDGE_TXT_DIR.glob(SUB_PREFIX + "*.txt")):
            num, title = _module_name(p.name)
            if num is not None:
                mods[num] = "Module " + str(num) + ": " + title
    return mods


def list_lab_exercises():
    """Returns display names for every lab exercise folder on disk."""
    labs = []
    base = Path("knowledge") / "Tessent Atpg Core Topics" / "10.Labs"
    if base.exists():
        for lab_dir in sorted(base.iterdir()):
            if not lab_dir.is_dir():
                continue
            for ex_dir in sorted(lab_dir.iterdir()):
                if ex_dir.is_dir():
                    labs.append(lab_dir.name + " / " + ex_dir.name)
                elif ex_dir.suffix.lower() in (".tcl", ".txt", ".doc", ".docx", ".pdf"):
                    labs.append(lab_dir.name + " / " + ex_dir.name)
    return labs


def _split_pages(text):
    """Splits an ingest text file into (page_label, body) tuples."""
    pages = []
    marker = "[SOURCE:"
    idx = text.find(marker)
    while idx != -1:
        end_line = text.find("]", idx)
        nxt = text.find(marker, idx + 1)
        if end_line == -1:
            end_line = idx + 10
        label = text[idx + 1:end_line]
        body = text[end_line + 1:nxt if nxt != -1 else len(text)]
        pages.append((label.strip(), body.strip()))
        idx = nxt
    return pages


def _format_chunks(chunks):
    """Turns retrieved chunks into a citation-friendly context block."""
    parts = []
    for c in chunks or []:
        if isinstance(c, dict):
            text = c.get("text") or c.get("content") or ""
            src = c.get("source") or c.get("source_name") or ""
            page = c.get("page") or c.get("page_num") or ""
            cite = str(src)
            if page:
                cite = cite + ", Page " + str(page)
            if cite.strip():
                block = "[Source: " + cite + "]\n" + str(text)
            else:
                block = str(text)
        else:
            block = str(c)
        if block.strip():
            parts.append(block)
    return "\n\n".join(parts)


def _llm(prompt):
    """Single LLM call with defensive content normalization."""
    try:
        resp = llm.invoke(prompt)
        content = getattr(resp, "content", resp)
        if isinstance(content, list):
            content = " ".join(str(p) for p in content)
        return str(content).strip()
    except Exception as exc:
        return "**LLM error:** " + str(exc)


VIDEO_PROMPT = """You are a senior Tessent DFT trainer preparing a trainee for a Wipro POC
assessment. The POC asks "why" questions, demands whiteboard-style step sequences,
and expects answers phrased the way the Tessent manuals phrase them.

Below is the SLIDE TEXT (if any) and the FULL TRANSCRIPT of ONE training video
lesson, followed by retrieved excerpts from the official Tessent reference manuals.

Task - produce the following sections IN THIS ORDER:

0. "## Layman's Explanation" — explain this topic as if teaching a complete beginner.
   Use analogies (e.g., "scan chain is like a shift register made of flip-flops").
   No jargon without defining it first. 2-3 short paragraphs. This section MUST come
   first so a fresher can understand before diving deep.

1. "## Lesson Explanation" - explain the lesson in detail, topic by topic, in
   clear engineering terms. Ground every major claim in the manual excerpts and
   cite like [Source: file, Page N]. Do not just restate the transcript; add the
   reasoning the transcript implies (e.g. WHY scan makes sequential ATPG become
   combinational ATPG, what breaks without it).

2. "## How It Works Internally" - the mechanism behind the concept: signal
   behavior, cycle-by-cycle or step-by-step operation, and small ASCII diagrams
   (e.g. FF -> Logic -> FF  vs  SI -> FF -> FF -> SO, or a load/capture/unload
   timing list) wherever they clarify the answer.

3. "## Where This Fits in the Tessent Flow" - which Tessent commands, files,
   reports or flow stages this lesson relates to, in typical execution order,
   with a one-line "why it is needed" per item and citations.

4. "## POC Expected Questions" — 12-15 realistic viva questions an experienced
   DFT POC would ask about THIS lesson, covering:
   - Why questions (why this step, why not skip it, why this order)
   - What-if questions (what happens if this fails, what if we skip it)
   - Order-of-operations questions (why this sequence)
   - Command/purpose questions (what does X command do, what file does it create)
   Each with a 3-5 line model answer that uses the manual's terminology and cites
   the manual like [Source: file, Page N].

5. "## Common Mistakes" — 6-8 mistakes freshers typically make when explaining
   this topic (wrong terminology, wrong order, confusing scan shift with capture,
   mixing ATPG with simulation, etc.), each with the correct statement.

6. "## 30-Second Interview Answer" - a tight, memorizable spoken answer to
   "explain this lesson/topic" as if answering the POC out loud.

7. "## 2-Minute Detailed Answer" - the expanded spoken version including one
   concrete example with values/steps and one citation.

8. "## Rapid Viva Questions" - 10 rapid-fire one-line questions with 1-2 line
   model answers.

SLIDE TEXT AND TRANSCRIPT:
{transcript}

MANUAL EXCERPTS:
{context}
"""

LAB_PROMPT = """You are a senior Tessent DFT trainer preparing a trainee for a Wipro POC
assessment. The POC asks about THE LAB THE TRAINEE PRESENTED: why each command ran,
in that order, what it produced, and what breaks if it is skipped or fails.

Below is the material for ONE lab exercise (slide/OCR text plus any TCL or text
files found in the lab folder), followed by retrieved excerpts from the official
Tessent reference manuals.

Task - produce the following sections IN THIS ORDER:

0. "## Layman's Explanation" — explain this lab as if teaching a complete beginner.
   What is the goal of this lab? Why does it matter? Use analogies where possible.
   2-3 short paragraphs. This section MUST come first.

1. "## Lab Walkthrough" - walk through the lab exercise step by step. For every
   Tessent command used, explain: what it does, why it is needed HERE, what it
   creates/outputs (file or report names), and what happens if it is skipped.
   Ground every command explanation in the manual excerpts and cite like
   [Source: file, Page N].

2. "## Lab Command Sequence" - the ordered command list as a compact annotated
   sequence (command -> purpose -> key output -> failure mode if skipped), so the
   trainee can reproduce the flow on the whiteboard from memory.

3. "## How the Flow Connects" - the state this lab stage produces (e.g. TSDB
   entries, test mode files, pattern files, reports) and how the NEXT stage
   depends on it. Explain order-of-operations: e.g. why analyze_scan_chains
   before insert_test_logic, why check_design_rules before ATPG, why
   add_scan_mode/add_clocks before pattern generation.

4. "## POC Expected Questions" — 12-15 realistic questions the POC would ask
   about THIS lab, covering:
   - Why did you run this command? What happens if it fails?
   - Why this order? What if we swap steps?
   - What file did that command create? What report does it produce?
   - What does this error mean? How do you debug it?
   Each with a 3-5 line model answer using manual terminology and citations.

5. "## Common Mistakes" — 6-8 mistakes freshers make in this lab (wrong command
   order, missing setup, misreading the DRC report, forgetting to save patterns,
   confusing contexts/system modes), each with the correct practice.

6. "## 30-Second Interview Answer" - a tight spoken answer to "walk me through
   this lab" as if answering the POC out loud.

7. "## 2-Minute Detailed Answer" - the expanded spoken version covering the full
   command flow, the key reports, and one failure scenario, with citations.

8. "## Rapid Viva Questions" - 10 rapid-fire one-line questions with model answers.

LAB MATERIAL:
{lab_text}

MANUAL EXCERPTS:
{context}
"""


def explain_video_lesson(module_num, top_k=12):
    """Generates the deep-study report for one video module."""
    matches = sorted(KNOWLEDGE_TXT_DIR.glob(SUB_PREFIX + str(module_num) + "_*.txt"))
    if not matches:
        return "No transcript found for module " + str(module_num)
    display = list_video_modules().get(module_num, "Module " + str(module_num))
    transcript = _read_text(matches[0], cap=24000)

    slides = sorted(KNOWLEDGE_TXT_DIR.glob(SLIDE_PREFIX + str(module_num) + "_*.txt"))
    slide_text = _read_text(slides[0], cap=12000) if slides else ""

    query = "Tessent " + display + " " + transcript[:600]
    context = _format_chunks(retrieve(query, top_k=top_k))

    prompt = VIDEO_PROMPT.format(
        transcript=(slide_text + "\n\n" + transcript).strip()[:30000],
        context=context,
    )
    report = _llm(prompt)
    log_deep_study("video", display, matches[0].name, report)
    return report


def _lab_source_text(lab_display):
    """Collects slide pages and TCL/text files belonging to one lab exercise."""
    head = lab_display.split(" / ")[0].lower().replace(" ", "")
    sub = ""
    if " / " in lab_display:
        sub = lab_display.split(" / ")[1].lower().replace(" ", "")

    picked = []
    pages = _split_pages(_read_text(KNOWLEDGE_TXT_DIR / LAB_SLIDES_FILE))
    for label, body in pages:
        blob = (label + "\n" + body).lower().replace(" ", "")
        if (head and head in blob) or (sub and sub in blob):
            picked.append(body)
    text = "\n\n".join(picked)

    extra = ""
    base = Path("knowledge") / "Tessent Atpg Core Topics" / "10.Labs"
    target = base / lab_display.replace(" / ", "/")
    if target.exists():
        if target.is_file():
            extra = _read_text(target, cap=20000)
        else:
            files = []
            for p in sorted(target.rglob("*")):
                if p.is_file() and p.suffix.lower() in (".tcl", ".txt", ".do", ".f"):
                    files.append("--- " + p.name + " ---\n" + _read_text(p, cap=8000))
            extra = "\n\n".join(files)
    return (text + "\n\n" + extra).strip()


def explain_lab_exercise(lab_display, top_k=12):
    """Generates the deep-study report for one lab exercise."""
    lab_text = _lab_source_text(lab_display)
    if not lab_text:
        return "Could not read any text material for this lab exercise."

    query = "Tessent lab " + lab_display + " " + lab_text[:600]
    context = _format_chunks(retrieve(query, top_k=top_k))

    prompt = LAB_PROMPT.format(lab_text=lab_text[:30000], context=context)
    report = _llm(prompt)
    item = lab_display.split(" / ")[-1]
    log_deep_study("lab", lab_display, item, report)
    return report


def explain_lab_exercise_v2(lab_display, top_k=12, generate_ppt=True, include_diagrams=True):
    """Generates the DEEP study report for ONE lab exercise — fusing slide text,
    TCL files, screenshot descriptions, manual references, command definitions.

    Shows ALL files at once with hyperlinks to explanations.
    Set include_diagrams=False for a fast report (skips vision analysis).
    """
    from agents.srt_analyzer import analyze_text
    from agents.ppt_generator import generate_lesson_ppt

    # 1. Get lab source text (slides + TCL files)
    lab_text = _lab_source_text(lab_display)
    if not lab_text:
        return "Could not read any text material for this lab exercise.", None, ""

    # 2. Analyze text for manual refs, commands, tables
    analysis = analyze_text(lab_text)

    # 3. Get diagram descriptions for ALL exercises at once (optional — slow)
    lab_name = lab_display.split(" / ")[0]
    exercise = lab_display.split(" / ")[1] if " / " in lab_display else None
    diagram_descriptions = ""
    if include_diagrams:
        from agents.diagram_vision import analyze_lab_diagrams
        diagram_descriptions = analyze_lab_diagrams(lab_name, exercise)

    # 4. Build enriched context
    context_parts = []

    # Base RAG retrieval
    query = "Tessent lab " + lab_display + " " + lab_text[:600]
    base_context = _format_chunks(retrieve(query, top_k=top_k))
    if base_context:
        context_parts.append(base_context)

    # Manual references from lab text
    if analysis["manual_context"]:
        context_parts.append("[Manual References from Lab]\n" + analysis["manual_context"])

    # Command definitions
    if analysis["command_context"]:
        context_parts.append("[Commands Mentioned in Lab]\n" + analysis["command_context"])

    # Tables
    if analysis["tables"]:
        tables_text = "\n\n".join(["[Table]\n" + t for t in analysis["tables"]])
        context_parts.append("[Tables from Lab]\n" + tables_text)

    # Diagram descriptions
    if diagram_descriptions:
        context_parts.append("[Diagram Descriptions]\n" + diagram_descriptions)

    full_context = "\n\n".join(context_parts)

    # 5. Build prompt
    prompt = LAB_PROMPT.format(
        lab_text=lab_text[:30000],
        context=full_context[:20000],
    )

    # 6. Generate report
    report = _llm(prompt)

    # 7. Log it
    log_deep_study("lab", lab_display, exercise or lab_name, report)

    # 8. Generate PPT if requested
    ppt_path = None
    if generate_ppt:
        try:
            ppt_path = generate_lesson_ppt(
                module_num=0,  # 0 for labs
                module_name=lab_name,
                lesson_title=exercise or lab_display,
                report_text=report,
                diagram_descriptions=diagram_descriptions,
                tables=analysis["tables"],
            )
        except Exception as exc:
            ppt_path = f"PPT generation failed: {exc}"

    return report, ppt_path, diagram_descriptions


def generate_lab_ppt_from_report(lab_display, report, diagram_descriptions=""):
    """Generates ONLY the PPT from an already-generated lab report (no LLM call).

    Re-extracts tables/analysis cheaply from source text — vision and report
    generation are skipped entirely.
    """
    from agents.srt_analyzer import analyze_text
    from agents.ppt_generator import generate_lesson_ppt

    lab_name = lab_display.split(" / ")[0]
    exercise = lab_display.split(" / ")[1] if " / " in lab_display else None

    analysis = {"tables": []}
    lab_text = _lab_source_text(lab_display)
    if lab_text:
        try:
            analysis = analyze_text(lab_text)
        except Exception:
            pass

    return generate_lesson_ppt(
        module_num=0,  # 0 for labs
        module_name=lab_name,
        lesson_title=exercise or lab_display,
        report_text=report,
        diagram_descriptions=diagram_descriptions,
        tables=analysis.get("tables", []),
    )


def generate_lesson_ppt_from_report(module_num, lesson_title, report, diagram_descriptions=""):
    """Generates ONLY the PPT from an already-generated lesson report (no LLM call)."""
    from agents.srt_analyzer import analyze_text
    from agents.ppt_generator import generate_lesson_ppt

    display = list_video_modules().get(module_num, "Module " + str(module_num))
    transcript = _lesson_block(module_num, lesson_title) or ""
    slide_text = _slides_for_lesson(module_num, lesson_title) or ""

    analysis = {"tables": []}
    combined = (slide_text + "\n\n" + transcript).strip()
    if combined:
        try:
            analysis = analyze_text(combined)
        except Exception:
            pass

    return generate_lesson_ppt(
        module_num=module_num,
        module_name=display,
        lesson_title=lesson_title,
        report_text=report,
        diagram_descriptions=diagram_descriptions,
        tables=analysis.get("tables", []),
    )


def list_lab_files_at_once(lab_name: str) -> dict:
    """List ALL files (screenshots, TCL, text, etc.) for a lab at once.

    Returns:
        Dict with structure:
        {
            "lab_name": "Lab2",
            "exercises": {
                "EX1": {
                    "screenshots": [Path, ...],
                    "tcl_files": [Path, ...],
                    "text_files": [Path, ...],
                    "all_files": [Path, ...]
                },
                "EX2": { ... },
                ...
            },
            "total_screenshots": 25,
            "total_tcl": 3,
            "total_text": 2
        }
    """
    from agents.diagram_vision import find_lab_screenshots, SCREENSHOT_BASE

    labs_base = SCREENSHOT_BASE / "10.Labs"
    result = {
        "lab_name": lab_name,
        "exercises": {},
        "total_screenshots": 0,
        "total_tcl": 0,
        "total_text": 0,
    }

    lab_folder = None
    for folder in labs_base.iterdir():
        if folder.is_dir() and folder.name.lower() == lab_name.lower():
            lab_folder = folder
            break

    if not lab_folder:
        return result

    for ex_folder in sorted(lab_folder.iterdir()):
        if not ex_folder.is_dir():
            continue

        ex_data = {
            "screenshots": [],
            "tcl_files": [],
            "text_files": [],
            "all_files": [],
        }

        # Find all files
        for f in sorted(ex_folder.iterdir()):
            if f.is_file():
                ex_data["all_files"].append(f)
                if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    ex_data["screenshots"].append(f)
                    result["total_screenshots"] += 1
                elif f.suffix.lower() in (".tcl", ".do", ".f"):
                    ex_data["tcl_files"].append(f)
                    result["total_tcl"] += 1
                elif f.suffix.lower() in (".txt", ".md", ".pdf"):
                    ex_data["text_files"].append(f)
                    result["total_text"] += 1

        result["exercises"][ex_folder.name] = ex_data

    return result

# ---------------------------------------------------------------------------
# Per-lesson (individual video) support
# ---------------------------------------------------------------------------

LESSON_MARKER = "--- PAGE"


def _split_blocks(text):
    """Splits ingest text into blocks keyed by their --- PAGE N --- headers."""
    blocks = []
    cur = None
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(LESSON_MARKER) and stripped.endswith("---"):
            if cur is not None:
                blocks.append(cur)
            cur = ""
        elif cur is not None:
            cur += line + "\n"
    if cur is not None:
        blocks.append(cur)
    return blocks


def _block_source(block):
    marker = "[SOURCE:"
    idx = block.find(marker)
    if idx == -1:
        return ""
    end = block.find("]", idx)
    if end == -1:
        return ""
    return block[idx + len(marker):end].strip()


def _lesson_title_from_source(source):
    if "| Lesson:" in source:
        return source.split("| Lesson:", 1)[1].strip()
    if "|Lesson:" in source:
        return source.split("|Lesson:", 1)[1].strip()
    stem = source.split("/")[-1]
    if stem.lower().endswith(".srt"):
        stem = stem[:-4]
    return stem


def _norm_title(s):
    return " ".join(s.lower().split())


def _titles_match(a, b):
    """Fuzzy title match: containment or >=60% word overlap."""
    na, nb = _norm_title(a), _norm_title(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.6


def list_video_lessons(module_num):
    """Returns the ordered individual lesson titles for one video module."""
    matches = sorted(KNOWLEDGE_TXT_DIR.glob(SUB_PREFIX + str(module_num) + "_*.txt"))
    if not matches:
        return []
    titles = []
    for block in _split_blocks(_read_text(matches[0])):
        title = _lesson_title_from_source(_block_source(block))
        if title:
            titles.append(title)
    return titles


def _lesson_block(module_num, lesson_title):
    """Returns the transcript body of ONE lesson block."""
    matches = sorted(KNOWLEDGE_TXT_DIR.glob(SUB_PREFIX + str(module_num) + "_*.txt"))
    if not matches:
        return ""
    wanted = _norm_title(lesson_title)
    for block in _split_blocks(_read_text(matches[0])):
        if _norm_title(_lesson_title_from_source(_block_source(block))) == wanted:
            body = block
            if body.strip().startswith("[SOURCE:"):
                body = body[body.find("]") + 1:]
            return body.strip()
    return ""


def _slides_for_lesson(module_num, lesson_title):
    """Collects slide-page bodies whose folder matches the lesson title."""
    slides = sorted(KNOWLEDGE_TXT_DIR.glob(SLIDE_PREFIX + str(module_num) + "_*.txt"))
    if not slides:
        return ""
    picked = []
    for block in _split_blocks(_read_text(slides[0])):
        parts = _block_source(block).split("/")
        folder = parts[1] if len(parts) >= 2 else (parts[0] if parts else "")
        if folder and _titles_match(lesson_title, folder):
            body = block
            if body.strip().startswith("[SOURCE:"):
                body = body[body.find("]") + 1:]
            picked.append(body.strip())
    return "\n\n".join(picked)


def lesson_slug(text, limit=40):
    """Filesystem/cache-friendly slug for a lesson title."""
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
    slug = "".join(out).strip("_")
    return slug[:limit] or "lesson"


def explain_video_lesson_detail(module_num, lesson_title, top_k=12):
    """Generates the deep-study report for ONE individual video lesson."""
    display = list_video_modules().get(module_num, "Module " + str(module_num))
    transcript = _lesson_block(module_num, lesson_title)
    if not transcript:
        return "No transcript block found for lesson: " + lesson_title
    slide_text = _slides_for_lesson(module_num, lesson_title)

    query = "Tessent " + display + " " + lesson_title + " " + transcript[:600]
    context = _format_chunks(retrieve(query, top_k=top_k))

    prompt = VIDEO_PROMPT.format(
        transcript=(slide_text + "\n\n" + transcript).strip()[:30000],
        context=context,
    )
    report = _llm(prompt)
    log_deep_study("video", display, lesson_title, report)
    return report


# ---------------------------------------------------------------------------
# Deep Study v2 — Fusion report with SRT analysis, diagrams, PPT
# ---------------------------------------------------------------------------

def explain_video_lesson_detail_v2(module_num, lesson_title, top_k=12, generate_ppt=True, include_diagrams=True):
    """Generates the DEEP study report for ONE lesson — fusing subtitles, slides,
    manual references, command definitions, and diagram descriptions.

    Optionally generates a PPT for SPOC review.
    Set include_diagrams=False for a fast report (skips vision analysis).
    """
    from agents.srt_analyzer import analyze_text
    from agents.ppt_generator import generate_lesson_ppt

    display = list_video_modules().get(module_num, "Module " + str(module_num))
    transcript = _lesson_block(module_num, lesson_title)
    if not transcript:
        return "No transcript block found for lesson: " + lesson_title, None, ""
    slide_text = _slides_for_lesson(module_num, lesson_title)

    # 1. Analyze transcript + slides for manual refs, commands, tables
    combined_text = transcript + "\n\n" + slide_text
    analysis = analyze_text(combined_text)

    # 2. Get diagram descriptions (optional — slow)
    diagram_descriptions = ""
    if include_diagrams:
        from agents.diagram_vision import analyze_lesson_diagrams
        diagram_descriptions = analyze_lesson_diagrams(module_num, lesson_title)

    # 3. Build enriched context
    context_parts = []

    # Base RAG retrieval
    query = "Tessent " + display + " " + lesson_title + " " + transcript[:600]
    base_context = _format_chunks(retrieve(query, top_k=top_k))
    if base_context:
        context_parts.append(base_context)

    # Manual references from SRT
    if analysis["manual_context"]:
        context_parts.append("[Manual References from Video]\n" + analysis["manual_context"])

    # Command definitions
    if analysis["command_context"]:
        context_parts.append("[Commands Mentioned in Video]\n" + analysis["command_context"])

    # Tables
    tables_text = ""
    if analysis["tables"]:
        tables_text = "\n\n".join(["[Table]\n" + t for t in analysis["tables"]])
        context_parts.append("[Tables from Slides]\n" + tables_text)

    # Diagram descriptions
    if diagram_descriptions:
        context_parts.append("[Diagram Descriptions]\n" + diagram_descriptions)

    full_context = "\n\n".join(context_parts)

    # 4. Build prompt
    prompt = VIDEO_PROMPT.format(
        transcript=(slide_text + "\n\n" + transcript).strip()[:30000],
        context=full_context[:20000],
    )

    # 5. Generate report
    report = _llm(prompt)

    # 6. Log it
    log_deep_study("video", display, lesson_title, report)

    # 7. Generate PPT if requested
    ppt_path = None
    if generate_ppt:
        try:
            ppt_path = generate_lesson_ppt(
                module_num=module_num,
                module_name=display,
                lesson_title=lesson_title,
                report_text=report,
                diagram_descriptions=diagram_descriptions,
                tables=analysis["tables"],
            )
        except Exception as exc:
            ppt_path = f"PPT generation failed: {exc}"

    return report, ppt_path, diagram_descriptions
