"""SRT Analyzer — extracts manual references, command mentions, and table
structures from video subtitle/slide text.

Used by the Deep Study v2 report generator to:
  - Detect "refer to section X" / "see the Shell Reference" callouts
  - Detect Tessent command mentions (add_*, set_*, analyze_*, insert_*, etc.)
  - Preserve table structures from OCR'd slide text
"""
import re
from pathlib import Path
from typing import List, Tuple

from core.rag_builder import retrieve
from core.llm import llm


# ---------------------------------------------------------------------------
# Manual reference extraction
# ---------------------------------------------------------------------------

MANUAL_REF_PATTERNS = [
    re.compile(r'(?:see|refer to|in|from|check|consult)\s+(?:the\s+)?(?:section|sec\.?|chapter|chap\.?|page|pg\.?)\s*(\d+(?:\.\d+)*)', re.IGNORECASE),
    re.compile(r'(?:see|refer to|in|from|check|consult)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:manual|reference|guide|documentation)', re.IGNORECASE),
    re.compile(r'(?:as\s+)?(?:described|explained|mentioned|noted|stated)\s+(?:in|at)\s+(?:section|sec\.?|chapter|chap\.?)?\s*(\d+(?:\.\d+)*)', re.IGNORECASE),
    re.compile(r'(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:manual|reference|guide)\s+(?:section|sec\.?|chapter|chap\.?)?\s*(\d+(?:\.\d+)*)', re.IGNORECASE),
    re.compile(r'(?:see|refer to)\s+(?:the\s+)?(?:figure|fig\.?|table|tbl\.?)\s*(\d+(?:\.\d+)*)', re.IGNORECASE),
]


def extract_manual_refs(text: str) -> List[str]:
    """Extract manual reference callouts from subtitle/slide text."""
    refs = []
    seen = set()
    for pattern in MANUAL_REF_PATTERNS:
        for match in pattern.finditer(text):
            ref = match.group(0).strip()
            ref_lower = ref.lower()
            if ref_lower not in seen:
                seen.add(ref_lower)
                refs.append(ref)
    return refs


# ---------------------------------------------------------------------------
# Command mention extraction
# ---------------------------------------------------------------------------

COMMAND_PATTERNS = [
    re.compile(r'\b(add_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(set_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(analyze_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(insert_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(report_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(check_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(create_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(write_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(read_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(save_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(load_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(run_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(delete_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(remove_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(open_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(close_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(extract_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(verify_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(connect_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(drive_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(force_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(measure_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(simulate_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(generate_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(compress_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(expand_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(debug_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(help_[a-z_]+)\b', re.IGNORECASE),
    re.compile(r'\b(quit)\b', re.IGNORECASE),
    re.compile(r'\b(exit)\b', re.IGNORECASE),
    re.compile(r'\b(system)\b', re.IGNORECASE),
    re.compile(r'\b(history)\b', re.IGNORECASE),
    re.compile(r'\b(ls)\b', re.IGNORECASE),
    re.compile(r'\b(pwd)\b', re.IGNORECASE),
    re.compile(r'\b(cat)\b', re.IGNORECASE),
    re.compile(r'\b(setenv)\b', re.IGNORECASE),
    re.compile(r'\b(unsetenv)\b', re.IGNORECASE),
]


def extract_commands(text: str) -> List[str]:
    """Extract Tessent command mentions from subtitle/slide text."""
    commands = set()
    for pattern in COMMAND_PATTERNS:
        for match in pattern.finditer(text):
            cmd = match.group(1).lower().strip()
            if cmd and len(cmd) > 1:
                commands.add(cmd)
    return sorted(commands)


# ---------------------------------------------------------------------------
# Manual context retrieval (targeted)
# ---------------------------------------------------------------------------

def get_manual_context_for_refs(refs: List[str], top_k: int = 5) -> str:
    """For each manual reference, retrieve the most relevant manual excerpt."""
    if not refs:
        return ""
    parts = []
    for ref in refs:
        query = f"Tessent {ref}"
        try:
            chunks = retrieve(query, top_k=top_k)
            if chunks:
                formatted = _format_chunks(chunks)
                if formatted.strip():
                    parts.append(f"[Manual Reference: {ref}]\n{formatted}")
        except Exception:
            continue
    return "\n\n".join(parts)


def get_command_definitions(commands: List[str], top_k: int = 3) -> str:
    """For each command, retrieve its manual definition."""
    if not commands:
        return ""
    parts = []
    for cmd in commands:
        query = f"Tessent command {cmd} definition syntax purpose"
        try:
            chunks = retrieve(query, top_k=top_k)
            if chunks:
                formatted = _format_chunks(chunks)
                if formatted.strip():
                    parts.append(f"[Command: {cmd}]\n{formatted}")
        except Exception:
            continue
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Table detection and preservation
# ---------------------------------------------------------------------------

def detect_and_preserve_tables(text: str) -> Tuple[str, List[str]]:
    """Detect table-like structures in OCR text and convert to markdown tables.

    Returns (processed_text, list_of_markdown_tables).
    """
    tables = []
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        if '|' in line and _is_table_start(lines, i):
            table_lines = []
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1
            if len(table_lines) >= 2:
                md_table = _convert_to_markdown_table(table_lines)
                if md_table:
                    tables.append(md_table)
        else:
            i += 1
    return text, tables


def _is_table_start(lines: List[str], idx: int) -> bool:
    """Check if the current line starts a table."""
    if idx + 1 < len(lines):
        next_line = lines[idx + 1]
        if re.match(r'^\s*\|[\s\-:|]+\|\s*$', next_line):
            return True
        if '|' in next_line:
            return True
    return False


def _convert_to_markdown_table(table_lines: List[str]) -> str:
    """Convert pipe-delimited lines to a proper markdown table."""
    if len(table_lines) < 2:
        return ""
    rows = []
    for line in table_lines:
        cells = [c.strip() for c in line.split('|')]
        if cells and not cells[0]:
            cells = cells[1:]
        if cells and not cells[-1]:
            cells = cells[:-1]
        if cells:
            rows.append(cells)
    if len(rows) < 2:
        return ""
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")
    header = rows[0]
    separator = ['---'] * max_cols
    md_lines = [
        '| ' + ' | '.join(header) + ' |',
        '| ' + ' | '.join(separator) + ' |',
    ]
    for row in rows[1:]:
        md_lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(md_lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_chunks(chunks) -> str:
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


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_text(text: str) -> dict:
    """Full analysis of subtitle/slide text.

    Returns dict with:
      - manual_refs: list of reference strings
      - commands: list of command names
      - tables: list of markdown tables
      - manual_context: retrieved manual excerpts for refs
      - command_context: retrieved command definitions
    """
    refs = extract_manual_refs(text)
    commands = extract_commands(text)
    _, tables = detect_and_preserve_tables(text)
    manual_context = get_manual_context_for_refs(refs)
    command_context = get_command_definitions(commands)
    return {
        "manual_refs": refs,
        "commands": commands,
        "tables": tables,
        "manual_context": manual_context,
        "command_context": command_context,
    }