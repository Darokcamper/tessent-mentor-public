"""WP-B: session usage & model-routing visibility for the Streamlit sidebar.

The WP-12 routing layer already tracks a per-session call/token budget and a
credential-free call log, but nothing surfaced them. This module renders that state.

Privacy contract (WP-10 / WP-12):
* only capability, model name, outcome, token count and retry-after are shown;
* prompts, answers, attachments, API keys and provider response bodies are never
  read here, and ``redact()`` strips anything key-shaped as a second line of defence;
* nothing is persisted — the state is in-process only and dies with the session.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

UNLIMITED = "unlimited"
NO_CALLS = "No model calls recorded in this session."
PRIVACY_NOTE = ("_Counts and model names only — no prompts, answers, files or credentials "
                "are shown or stored._")

_KEY_PATTERNS = re.compile(
    r"(AIza[0-9A-Za-z_\-]{10,}"          # Google API key shape
    r"|sk-[0-9A-Za-z]{10,}"              # OpenAI-style key shape
    r"|ghp_[0-9A-Za-z]{10,}"             # GitHub PAT shape
    r"|(?i:api[_-]?key\s*[=:]\s*\S+)"
    r"|(?i:bearer\s+\S+))"
)


def redact(text: str) -> str:
    """Remove anything that looks like a credential from rendered text."""
    return _KEY_PATTERNS.sub("[redacted]", str(text))


def _fmt(value: Optional[int]) -> str:
    return UNLIMITED if value is None else f"{value:,}"


def usage_lines(status: Dict[str, Any]) -> List[str]:
    """Calls/tokens used, remaining, and exhaustion state."""
    calls = int(status.get("calls", 0) or 0)
    tokens = int(status.get("tokens", 0) or 0)
    max_calls = status.get("max_calls")
    max_tokens = status.get("max_tokens")
    lines = [
        f"**Calls:** {calls:,} used"
        + (f" / {_fmt(max_calls)} allowed" if max_calls is not None else f" ({UNLIMITED} budget)")
        + (f" · {_fmt(status.get('remaining_calls'))} remaining" if max_calls is not None else ""),
        f"**Tokens:** {tokens:,} used"
        + (f" / {_fmt(max_tokens)} allowed" if max_tokens is not None else f" ({UNLIMITED} budget)")
        + (f" · {_fmt(status.get('remaining_tokens'))} remaining" if max_tokens is not None else ""),
    ]
    if tokens == 0:
        lines.append("_Token usage is only known when the provider returns usage metadata; "
                     "0 means no usage information was available._")
    lines.append("**Status:** " + ("exhausted — new calls will be refused" if status.get("exhausted")
                                   else "within budget"))
    return lines


def capability_lines(routing: Dict[str, str]) -> List[str]:
    """Capability -> selected model map."""
    if not routing:
        return ["_No routing table available._"]
    return [f"- `{capability}` → `{model}`" for capability, model in sorted(routing.items())]


def call_summary(history: Sequence[Dict[str, Any]]) -> List[str]:
    """Aggregate outcome counts, including retries and failures."""
    if not history:
        return [NO_CALLS]
    ok = sum(1 for entry in history if entry.get("outcome") == "ok")
    failed = [entry for entry in history if entry.get("outcome") != "ok"]
    retried = [entry for entry in history if entry.get("retry_after") is not None]
    lines = [f"**Recorded calls:** {len(history)} ({ok} ok, {len(failed)} not ok)"]
    if retried:
        last = retried[-1]
        lines.append(f"**Rate limiting:** {len(retried)} call(s) hit a retry-after; "
                     f"last wait {last.get('retry_after')}s")
    for entry in failed[-3:]:
        lines.append(
            f"- {entry.get('timestamp', '?')} · `{entry.get('capability')}` · "
            f"`{entry.get('model')}` · {entry.get('outcome')}"
        )
    return lines


def recent_calls(history: Sequence[Dict[str, Any]], limit: int = 5) -> List[str]:
    """Most recent calls, newest last, credential-free."""
    if not history:
        return [NO_CALLS]
    lines = []
    for entry in list(history)[-limit:]:
        tokens = "" if entry.get("tokens") is None else f" · {entry.get('tokens'):,} tokens"
        lines.append(
            f"- {entry.get('timestamp', '?')} · `{entry.get('capability')}` · "
            f"`{entry.get('model')}` · {entry.get('outcome')}{tokens}"
        )
    return lines


def panel_markdown(
    status: Dict[str, Any],
    routing: Optional[Dict[str, str]] = None,
    history: Iterable[Dict[str, Any]] = (),
    limit: int = 5,
) -> str:
    """Full panel body as markdown (redacted)."""
    history = list(history)
    sections = [
        "### Usage this session",
        *usage_lines(status),
        "### Capability → model",
        *capability_lines(routing or {}),
        "### Recent model calls",
        *call_summary(history),
        *recent_calls(history, limit),
        PRIVACY_NOTE,
    ]
    return redact("\n\n".join(sections))


def render_usage_panel(st, session_id: str) -> Dict[str, Any]:
    """Render the sidebar expander and return the state that was shown."""
    from core import model_routing

    status = model_routing.budget_status(session_id)
    routing = model_routing.routing_table()
    history = model_routing.call_history()

    with st.expander("📊 Session usage & model routing", expanded=False):
        st.markdown(panel_markdown(status, routing, history))
        if st.button("♻️ Reset session counters", key="vm_usage_reset"):
            model_routing.reset_session_budget(session_id)
            model_routing.clear_call_history()
            st.info("Session counters reset. Model calls already made are unaffected.")
        st.caption("Providers and models come from `VM_MODEL_*` overrides when set.")

    return {"status": status, "routing": routing, "calls": len(history)}
