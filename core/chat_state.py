"""Chat message-state helpers for the WP-9 edit / regenerate controls.

Keeping the list surgery here gives the UI a small, testable seam: the transcript
format itself stays in ``core/chat_transcript.py`` and is unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def truncate_for_edit(messages: List[Dict[str, Any]], index: int) -> List[Dict[str, Any]]:
    """Messages that survive editing the turn at *index* (that turn and after go)."""
    if index is None or index < 0:
        return list(messages)
    return list(messages[:index])


def drop_last_answer(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """Remove the final answer *and* its question; return (remaining, question to retry).

    The UI re-submits the returned question through the normal pipeline, which appends
    a fresh user/assistant pair. Both halves of the last turn must therefore be removed
    here, otherwise the retried question would appear twice.
    """
    remaining = list(messages)
    question = ""
    while remaining and remaining[-1].get("role") == "assistant":
        remaining.pop()
    if remaining and remaining[-1].get("role") == "user":
        question = str(remaining.pop().get("content") or "")
    return remaining, question


def append_turn(messages: List[Dict[str, Any]], question: str, answer: str) -> List[Dict[str, Any]]:
    """Append one user/assistant pair, keeping question→answer association exact."""
    updated = list(messages)
    updated.append({"role": "user", "content": question})
    updated.append({"role": "assistant", "content": answer})
    return updated
