"""
core/file_reader.py

Extracts text from user-uploaded documents (PDF, DOCX, PPTX, TXT) so they can be
attached to a question or permanently indexed into the RAG knowledge base.

Returns a "paged" structure compatible with the existing knowledge/txt caching
format (--- PAGE N --- markers) so downstream citation logic works uniformly.
"""
from __future__ import annotations

import io
from pathlib import Path

# Supported extensions for text extraction
TEXT_EXTS = {".pdf", ".docx", ".pptx", ".ppt", ".txt"}
# Supported image extensions (sent to the vision model, not indexed)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def is_text_ext(filename: str) -> bool:
    return Path(filename).suffix.lower() in TEXT_EXTS


def is_image_ext(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMAGE_EXTS


def _pdf_to_pages(file_bytes: bytes) -> list[str]:
    """Extract per-page text from a PDF using PyMuPDF."""
    import fitz  # PyMuPDF (already a project dependency)

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page_num in range(len(doc)):
        page_text = doc[page_num].get_text("text").strip()
        pages.append(f"--- PAGE {page_num + 1} ---\n{page_text}\n")
    doc.close()
    return pages


def _docx_to_pages(file_bytes: bytes) -> list[str]:
    """Extract text from a .docx file, chunking by paragraph to mimic page breaks.

    A docx has no hard pages, so we approximate "pages" by grouping ~40 paragraphs.
    The first group is labeled Page 1, the next Page 2, etc.
    """
    import docx  # python-docx

    document = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]

    pages = []
    batch = []
    for text in paragraphs:
        batch.append(text)
        # ~40 paragraphs per pseudo-page, or ~1500 words
        if len(batch) >= 40 or sum(len(p.split()) for p in batch) >= 1500:
            pages.append(f"--- PAGE {len(pages) + 1} ---\n" + "\n".join(batch) + "\n")
            batch = []
    if batch:
        pages.append(f"--- PAGE {len(pages) + 1} ---\n" + "\n".join(batch) + "\n")

    if not pages:
        pages = ["--- PAGE 1 ---\n(No extractable paragraph text found in this .docx.)\n"]
    return pages
def _pptx_to_pages(file_bytes: bytes) -> list[str]:
    """Extract text from a .pptx/.ppt file, treating each slide as one page."""
    from pptx import Presentation  # python-pptx

    prs = Presentation(io.BytesIO(file_bytes))
    pages = []
    for slide_idx, slide in enumerate(prs.slides, start=1):
        slide_texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in para.runs).strip()
                    if text:
                        slide_texts.append(text)
            if shape.has_table:
                for row in shape.table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    slide_texts.append(" | ".join(c for c in cells if c))
        page_text = "\n".join(slide_texts).strip()
        pages.append(f"--- PAGE {slide_idx} ---\n{page_text}\n" if page_text
                     else f"--- PAGE {slide_idx} ---\n(No text found on this slide.)\n")
    if not pages:
        pages = ["--- PAGE 1 ---\n(No extractable slide text found in this presentation.)\n"]
    return pages


def _txt_to_pages(file_bytes: bytes, fallback_name: str) -> list[str]:
    """Extract text from a plain .txt file, chunking to mimic pages.

    If the txt already contains '--- PAGE N ---' markers they are preserved;
    otherwise it is chunked into ~3500-char pseudo-pages starting at Page 1.
    """
    import re

    raw = file_bytes.decode("utf-8", errors="replace")
    markers = list(re.finditer(r"--- PAGE (\d+) ---", raw))
    if markers:
        pages = []
        for idx, m in enumerate(markers):
            start = m.start()
            end = markers[idx + 1].start() if idx + 1 < len(markers) else len(raw)
            pages.append(raw[start:end])
        return pages

    words = raw.split()
    pages = []
    current = []
    current_len = 0
    for w in words:
        current.append(w)
        current_len += len(w) + 1
        if current_len >= 3500:
            pages.append(f"--- PAGE {len(pages) + 1} ---\n" + " ".join(current) + "\n")
            current = []
            current_len = 0
    if current:
        pages.append(f"--- PAGE {len(pages) + 1} ---\n" + " ".join(current) + "\n")
    if not pages:
        pages = [f"--- PAGE 1 ---\n(File '{fallback_name}' appears to be empty.)\n"]
    return pages


def extract_document_text(file_bytes: bytes, filename: str) -> list[str]:
    """Return a list of paged text blocks (each prefixed with '--- PAGE N ---').

    Raises ValueError for unsupported extensions or extraction failures.
    """
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".pdf":
            return _pdf_to_pages(file_bytes)
        if ext == ".docx":
            return _docx_to_pages(file_bytes)
        if ext in (".pptx", ".ppt"):
            return _pptx_to_pages(file_bytes)
        if ext == ".txt":
            return _txt_to_pages(file_bytes, filename)
    except Exception as e:
        raise ValueError(f"Failed to extract text from '{filename}': {e}") from e

    raise ValueError(
        f"Unsupported file type '{ext}' for '{filename}'. Supported: {', '.join(sorted(TEXT_EXTS))}."
    )


def document_text_to_string(file_bytes: bytes, filename: str, max_chars: int = 60000) -> str:
    """Flatten extracted paged text into a single string for inlining in a prompt,
    truncated to max_chars to protect the prompt window."""
    pages = extract_document_text(file_bytes, filename)
    combined = "\n\n".join(pages).strip()
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n... [attachment truncated for length] ..."
    return combined