"""Local ingestion for the "Tessent ATPG Core Topics" video-course content.

The course consists of screenshots (PNG, digital-text slides) and video subtitles
(SRT, narration transcripts). This module:

  1. Parses every SRT into a clean narration transcript.
  2. OCRs every PNG screenshot with RapidOCR (local, offline) and stitches the
     detected text lines back in visual (top-to-bottom) order.
  3. Writes results into knowledge/txt/ as source files that the existing
     FAISS index builder (build_index in rag_builder) can add to the vector store.

Runs fully offline - no Gemini/cloud needed.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
TXT_DIR = KNOWLEDGE_DIR / "txt"
COURSE_DIR = KNOWLEDGE_DIR / "Tessent Atpg Core Topics"

_OCR = None


def _get_ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


# --------------------------------------------------------------------------- #
# SRT parsing
# --------------------------------------------------------------------------- #
_SRT_INDEX_RE = re.compile(r"^\s*\d+\s*$")
_SRT_TIME_RE = re.compile(
    r"^\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def parse_srt(text):
    """Return a list of subtitle text lines from an SRT document."""
    lines = []
    current = []
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        s = line.strip()
        if not s:
            if current:
                lines.append(" ".join(current))
                current = []
            continue
        if _SRT_INDEX_RE.match(s) or _SRT_TIME_RE.match(s):
            continue
        clean = re.sub(r"^\s*>>\s*", "", s).strip()
        if clean:
            current.append(clean)
    if current:
        lines.append(" ".join(current))
    return [ln for ln in lines if ln.strip()]


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _topic_slug(topic_dir):
    return _slug(topic_dir.name)


def _subtopic_of(png, topic_dir):
    rel = png.relative_to(topic_dir)
    parts = rel.parts
    return parts[0] if len(parts) > 1 else "root"


def ocr_image(ocr, png_path):
    """OCR a single PNG and return detected text in visual order."""
    try:
        result, _ = ocr(str(png_path))
    except Exception as exc:
        return f"[OCR ERROR: {exc}]"
    if not result:
        return ""
    return "\n".join(txt for _, txt, _ in result if txt and txt.strip())


def _write_source(entries, out_path, page_label="PAGE"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, (label, text) in enumerate(entries, start=1):
        if not text or not text.strip():
            continue
        parts.append(f"--- {page_label} {i} --- \n{label}\n{text}\n")
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return len(parts)


def ingest_subtitles(topic_only=None):
    """Parse all SRT files (or a single topic) into transcript source files."""
    if not COURSE_DIR.exists():
        raise FileNotFoundError(f"Course dir not found: {COURSE_DIR}")
    written = []
    for topic_dir in sorted(COURSE_DIR.iterdir()):
        if not topic_dir.is_dir():
            continue
        if topic_only and _slug(topic_only) not in _slug(topic_dir.name):
            continue
        srt_files = sorted(topic_dir.rglob("*.srt"))
        if not srt_files:
            continue
        out = TXT_DIR / f"atpg_video_subs__{_topic_slug(topic_dir)}.txt"
        entries = []
        for srt in srt_files:
            label = f"[SOURCE: {srt.name} | Lesson: {srt.parent.name}]"
            try:
                text = srt.read_text(encoding="utf-8-sig", errors="ignore")
            except Exception:
                continue
            lines = parse_srt(text)
            if lines:
                entries.append((label, "\n".join(lines)))
        n = _write_source(entries, out)
        print(f"  subtitles {out.name}: {n} lesson notebook(s)")
        if n:
            written.append(out)
    return written


def ingest_slides(topic_only=None, sample=None):
    """OCR all PNG slides (optionally one topic, or only `sample` images) and
    write OCR text source files per topic."""
    if not COURSE_DIR.exists():
        raise FileNotFoundError(f"Course dir not found: {COURSE_DIR}")
    ocr = _get_ocr()
    written = []
    for topic_dir in sorted(COURSE_DIR.iterdir()):
        if not topic_dir.is_dir():
            continue
        if topic_only and _slug(topic_only) not in _slug(topic_dir.name):
            continue
        png_files = sorted(topic_dir.rglob("*.png"))
        if not png_files:
            continue
        is_quiz = bool(re.search(r"exam|knowledge check|assessment", topic_dir.name, re.I))
        kind = "quiz" if is_quiz else "slides"
        out = TXT_DIR / f"atpg_video_{kind}__{_topic_slug(topic_dir)}.txt"

        if sample is None and out.exists() and out.stat().st_size > 200:
            print(f"  skip {out.name} (already present)")
            written.append(out)
            continue

        pick = png_files if sample is None else png_files[:sample]
        entries = []
        for i, png in enumerate(pick, start=1):
            sub = _subtopic_of(png, topic_dir)
            label = f"[SOURCE: {_topic_slug(topic_dir)}/{sub}/{png.name}]"
            txt = ocr_image(ocr, png)
            if txt.strip():
                entries.append((label, txt))
            if i % 25 == 0:
                print(f"    ... {i}/{len(pick)} images ({topic_dir.name})")
                sys.stdout.flush()
        n = _write_source(entries, out)
        print(f"  slides {out.name}: {n}/{len(pick)} -> {_sz(out)}")
        if n:
            written.append(out)
        sys.stdout.flush()
    return written


def _sz(p):
    try:
        return f"{p.stat().st_size} bytes"
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser(description="Ingest Tessent ATPG Core Topics content")
    ap.add_argument("--subtitles", action="store_true", help="Parse SRT subtitles")
    ap.add_argument("--slides", action="store_true", help="OCR PNG slides")
    ap.add_argument("--all", action="store_true", help="Run both subtitles and slides")
    ap.add_argument("--topic", default=None, help="Only process this topic substring")
    ap.add_argument("--sample", type=int, default=None, help="OCR this many images per topic")
    args = ap.parse_args()

    if not (args.subtitles or args.slides or args.all):
        ap.error("choose --subtitles, --slides, or --all")

    t0 = time.time()
    if args.subtitles or args.all:
        print("=== Ingesting SRT subtitles ===")
        res = ingest_subtitles(args.topic)
        print(f"  wrote {len(res)} subtitle source file(s)")

    if args.slides or args.all:
        print("=== OCR-ing PNG slides ===")
        res = ingest_slides(args.topic, sample=args.sample)
        print(f"  wrote {len(res)} slide source file(s)")

    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()