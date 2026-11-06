"""Bulk command explanation (WP-3).

`write*` returns 32 commands; explaining them one at a time is unusable. This module renders
every match in one pass, **from the verified command index only**:

* purpose / syntax / arguments / examples are the index's own fields;
* unavailable fields are stated as unavailable, never filled in;
* every block carries its manual citation;
* no model is called by default, so nothing can be invented. An optional `llm_fn` may polish
  prose, and is given only the retrieved definition text.

The batching cap keeps a single request small; failures are reported per command.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional

MAX_COMMANDS = 60
_MD_LIMIT = 1800


def _citation(entry: Dict[str, Any]) -> Optional[str]:
    evidence = entry.get("source_evidence") or {}
    file_name = evidence.get("file")
    page = evidence.get("page")
    if file_name and page:
        return f"[Source: {file_name}, Page {page}]"
    return None


def _clip(text: Any, limit: int = _MD_LIMIT) -> str:
    value = re.sub(r"\n{3,}", "\n\n", str(text or "")).strip()
    return value if len(value) <= limit else value[:limit].rstrip() + " …"


def explain_one(name: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """One source-grounded block. Missing data is reported, never fabricated."""
    gaps = []
    purpose = entry.get("purpose") or None
    syntax = entry.get("syntax") or None
    arguments = entry.get("arguments") or []
    examples = entry.get("examples") or None
    citation = _citation(entry)
    if not purpose:
        gaps.append("purpose")
    if not syntax:
        gaps.append("syntax")
    if not arguments:
        gaps.append("arguments")
    if not citation:
        gaps.append("citation")
    return {
        "command": name,
        "purpose": _clip(purpose) if purpose else None,
        "syntax": _clip(syntax, 400) if syntax else None,
        "arguments": [_clip(item, 220) for item in arguments[:12]],
        "examples": _clip(examples, 600) if examples else None,
        "context": entry.get("context") or None,
        "mode": entry.get("mode") or None,
        "module": entry.get("module") or None,
        "flow_step": entry.get("flow_step") or None,
        "flow_step_unavailable": bool(entry.get("flow_step_unavailable")),
        "citation": citation,
        "gaps": gaps,
        "documented": len(gaps) == 0,
    }


def bulk_explain(
    query: str,
    *,
    limit: int = MAX_COMMANDS,
    index: Optional[Dict[str, Any]] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """Explain every command matching *query* (prefix/wildcard semantics of search_commands)."""
    from core.command_search import search_commands

    if index is None:
        from core.command_indexer import load_command_index

        index = load_command_index().get("commands", {})

    matches = search_commands(query) or []
    names = [item.get("canonical_name") or item.get("command") for item in matches]
    names = [name for name in dict.fromkeys(names) if name in index]
    selected = names[:limit]

    blocks = [explain_one(name, index[name]) for name in selected]
    if llm_fn is not None:
        for block in blocks:
            source_text = " ".join(filter(None, [block["purpose"], block["syntax"],
                                                 " ".join(block["arguments"])]))
            if not source_text:
                block["gaps"].append("model_summary_skipped_no_source_text")
                block["documented"] = False
                continue
            try:
                block["summary"] = str(llm_fn(source_text))[:600]
            except Exception as error:  # partial failure is reported, not hidden
                block["summary_error"] = type(error).__name__

    return {
        "query": query,
        "matched": len(names),
        "explained": len(blocks),
        "truncated": max(0, len(names) - len(selected)),
        "commands": blocks,
        "undocumented": [b["command"] for b in blocks if not b["documented"]],
    }


def render_bulk_markdown(result: Dict[str, Any]) -> str:
    lines = [
        f"# Bulk command explanation — `{result['query']}`",
        "",
        f"{result['explained']} of {result['matched']} matching command(s) rendered"
        + (f" (first {result['explained']}; {result['truncated']} more not shown)"
           if result.get("truncated") else "")
        + ".",
        "",
        "_Every field below is copied from the verified command index. A field marked "
        "\u201cnot recorded in the sources\u201d was genuinely absent \u2014 nothing was inferred._",
        "",
    ]
    for block in result["commands"]:
        lines.append(f"## `{block['command']}`")
        if block.get("citation"):
            lines.append(f"*Source: {block['citation']}*")
        lines.append(f"**Purpose** \u2014 {block['purpose'] or 'not recorded in the sources.'}")
        if block.get("syntax"):
            lines.append("```\n" + block["syntax"] + "\n```")
        else:
            lines.append("**Syntax** \u2014 not recorded in the sources.")
        if block["arguments"]:
            lines.append("**Arguments**")
            lines += [f"- {item}" for item in block["arguments"]]
        else:
            lines.append("**Arguments** \u2014 not recorded in the sources.")
        if block.get("examples"):
            lines.append("**Examples**")
            lines.append("```\n" + block["examples"] + "\n```")
        bits = [f"context: {block['context']}" if block.get("context") else None,
                f"mode: {block['mode']}" if block.get("mode") else None,
                f"module: {block['module']}" if block.get("module") else None]
        bits = [bit for bit in bits if bit]
        if bits:
            lines.append("**Where it applies** \u2014 " + "; ".join(bits))
        if block.get("flow_step"):
            lines.append(f"**Flow step (curated)** \u2014 {block['flow_step']}")
        if block.get("summary"):
            lines.append(f"**Summary** \u2014 {block['summary']}")
        if block["gaps"]:
            lines.append(f"_Not recorded: {', '.join(block['gaps'])}._")
        lines.append("")
    return "\n".join(lines)
