"""Model routing, per-call accounting and API budget policy (WP-12).

Design goals:

* Keep the existing six-key Gemini rotation, lazy key loading and BYOK guest mode
  exactly as they are (``core/llm.py`` owns those).
* Make the model choice for each *capability* explicit and testable instead of the
  single global pair every agent used to share.
* Never record or log a credential value: only capability, provider, model name,
  outcome and token counts are recorded.
* Give the answer path a bounded, ``Retry-After``-aware backoff for HTTP 429.

Everything here is offline-testable: no network call is made by this module.
"""

from __future__ import annotations

import os
import random
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

# ── Capability → model routing ────────────────────────────────────────────────
# The capability names describe *what the call is for*, so a routing change never
# needs to touch agent code.
CAPABILITY_ROUTES: Dict[str, str] = {
    "answer_synthesis": "gemini-2.5-flash",
    "react_reasoning": "gemini-2.5-flash",
    "vision": "gemini-2.5-flash",
    "routing": "gemini-2.5-flash-lite",
    "question_generation": "gemini-2.5-flash-lite",
    "batch_report": "gemini-2.5-flash-lite",
}

# Documented agent → capability map. Single-shot domain agents synthesise an answer;
# the ReAct domains reason with tools; the router only classifies.
AGENT_CAPABILITIES: Dict[str, str] = {
    "scan": "answer_synthesis",
    "atpg": "answer_synthesis",
    "sta": "answer_synthesis",
    "edt": "react_reasoning",
    "mbist": "react_reasoning",
    "jtag": "react_reasoning",
    "ijtag": "react_reasoning",
    "wrapper": "react_reasoning",
    "occ": "react_reasoning",
    "boundaryscan": "react_reasoning",
    "gls": "react_reasoning",
    "tcl": "react_reasoning",
    "linux": "react_reasoning",
    "general": "react_reasoning",
    "router": "routing",
    "question_bank": "question_generation",
    "deep_study": "batch_report",
    "lab_explainer": "batch_report",
    "ppt_explainer": "batch_report",
    "attachment_vision": "vision",
}

# Ordered fallback chain used when a model is unavailable / rate limited.
FALLBACK_CHAIN = ("gemini-2.5-flash", "gemini-2.5-flash-lite")

_ENV_PREFIX = "VM_MODEL_"


def capability_for_agent(agent_name: str) -> str:
    """Capability for an agent name ('scan', 'edt', 'agents.edt_agent', ...)."""
    key = str(agent_name or "").strip().lower()
    key = re.sub(r"^agents\.", "", key)
    key = re.sub(r"_agent$", "", key)
    return AGENT_CAPABILITIES.get(key, "answer_synthesis")


def resolve_model(capability: str, model_index: int = 0, environ: Optional[Dict[str, str]] = None) -> str:
    """Model for a capability, honouring ``VM_MODEL_<CAPABILITY>`` overrides.

    ``model_index`` walks the documented fallback chain so the existing rotation still
    moves to the smaller model after exhausting the retry budget.
    """
    env = os.environ if environ is None else environ
    override = env.get(f"{_ENV_PREFIX}{str(capability).upper()}", "").strip()
    primary = override or CAPABILITY_ROUTES.get(capability, FALLBACK_CHAIN[0])
    if model_index and model_index % len(FALLBACK_CHAIN):
        return FALLBACK_CHAIN[model_index % len(FALLBACK_CHAIN)]
    return primary


def routing_table(environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Effective capability → model map (documentation / UI)."""
    return {capability: resolve_model(capability, 0, environ) for capability in CAPABILITY_ROUTES}


# ── 429-aware backoff ────────────────────────────────────────────────────────
_RATE_LIMIT_MARKERS = (
    "resource_exhausted",
    "resource exhausted",
    "rate limit",
    "rate_limit",
    "quota",
    "too many requests",
)

# WP-R2: the marker list used to contain the bare string "429", so ANY exception whose
# class name or message merely contained those three digits (a chunk id, a page path, an
# object address, even a test function name) was treated as a rate limit. The status code
# is now matched as a bounded token, and only against the message text.
_STATUS_429_RE = re.compile(r"(?<![\w.])429(?![\w.])")

# WP-R1: durations a provider may express in its retry hint.
_DURATION_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|sec|secs|second|seconds|m|min)?\s*$", re.IGNORECASE)
_RETRY_DELAY_RE = re.compile(
    r"""retry[_-]?delay["']?\s*[:=]\s*["']?\s*(\d+(?:\.\d+)?)\s*(ms|s)?""", re.IGNORECASE)
_RETRY_IN_RE = re.compile(r"retry in\s+(\d+(?:\.\d+)?)\s*(ms|s)", re.IGNORECASE)
_RETRY_AFTER_TEXT_RE = re.compile(
    r"""retry[_-]?after["']?\s*[:=]\s*["']?\s*(\d+(?:\.\d+)?)\s*(ms|s)?""", re.IGNORECASE)

_HINT_KEYS = ("retrydelay", "retry_delay", "retry-after", "retry_after")


def _seconds(amount: str, unit: Optional[str]) -> Optional[float]:
    """Convert an amount + unit to seconds; ``None`` for anything unusable."""
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    unit = (unit or "s").strip().lower()
    if unit == "ms":
        return round(value / 1000.0, 3)
    if unit in ("m", "min"):
        return round(value * 60.0, 3)
    return value


def _hint_from_structure(node: Any, depth: int = 0) -> Optional[float]:
    """Find a retry hint inside dicts/lists (e.g. a RetryInfo detail). Bounded, never raises."""
    if node is None or depth > 4:
        return None
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).strip().lower() in _HINT_KEYS:
                seconds = parse_retry_after(value)
                if seconds is not None:
                    return seconds
        for value in node.values():
            seconds = _hint_from_structure(value, depth + 1)
            if seconds is not None:
                return seconds
        return None
    if isinstance(node, (list, tuple)):
        for item in node:
            seconds = _hint_from_structure(item, depth + 1)
            if seconds is not None:
                return seconds
    return None


def _status_int(node: Any) -> Optional[int]:
    """Integer HTTP status carried by *node* (status_code or code), if any."""
    for attribute in ("status_code", "code"):
        value = getattr(node, attribute, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return None


def is_rate_limit_error(exc: BaseException) -> bool:
    """True when *exc* really is an HTTP 429 / quota rejection (WP-R2).

    Checks the exception and its direct cause for an integer 429 status or the
    RESOURCE_EXHAUSTED status string, then falls back to word markers and a **bounded**
    ``429`` token in the message. Unrelated text containing the digits "429"
    (``chunk 4291``, ``p.429``, a class name) no longer qualifies.
    """
    if exc is None:
        return False

    for node in (exc, getattr(exc, "__cause__", None)):
        if node is None:
            continue
        if _status_int(node) == 429:
            return True
        status = getattr(node, "status", None)
        if isinstance(status, str) and status.strip().upper() == "RESOURCE_EXHAUSTED":
            return True

    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in _RATE_LIMIT_MARKERS):
        return True
    return bool(_STATUS_429_RE.search(str(exc)))


def parse_retry_after(value: Any) -> Optional[float]:
    """Seconds from a ``Retry-After`` value.

    Accepts every shape the project actually meets: a response-like object with headers,
    a numeric string, a duration string ("17s", "500ms"), a dict/list carrying a
    ``RetryInfo``/``retryDelay`` entry, or free text that embeds one of those.
    """
    if value is None:
        return None

    if hasattr(value, "headers"):  # response-like object
        try:
            header = value.headers.get("Retry-After") or value.headers.get("retry-after")
        except Exception:
            header = None
        return parse_retry_after(header) if header is not None else None

    if isinstance(value, (dict, list, tuple)):
        return _hint_from_structure(value)

    text = str(value).strip()
    if not text:
        return None

    match = _DURATION_RE.match(text)
    if match:
        return _seconds(match.group(1), match.group(2))

    for pattern in (_RETRY_DELAY_RE, _RETRY_AFTER_TEXT_RE, _RETRY_IN_RE):
        match = pattern.search(text)
        if match:
            unit = match.group(2) if (match.lastindex or 0) >= 2 else "s"
            return _seconds(match.group(1), unit)
    return None


def retry_hint_from_exception(exc: BaseException, max_depth: int = 3) -> Optional[float]:
    """Provider retry hint (seconds) from a real SDK exception, or ``None``.

    WP-R1: the live SDK raises ``ChatGoogleGenerativeAIError`` whose ``__cause__`` is a
    ``google.genai.errors.ClientError`` exposing ``.details`` (the JSON body, including
    ``RetryInfo.retryDelay``). It carries no ``.response`` and no ``.status_code``, which
    is why the previous implementation always returned ``None`` in production. This walks
    response headers, ``details``/``error``/``message`` attributes, the rendered message,
    and the cause/context chain — and never raises.
    """
    seen: set = set()

    def walk(node: Any, depth: int) -> Optional[float]:
        if node is None or depth > max_depth or id(node) in seen:
            return None
        seen.add(id(node))

        for attribute in ("response", "details", "error", "message"):
            try:
                candidate = getattr(node, attribute, None)
            except Exception:
                candidate = None
            if candidate is None:
                continue
            seconds = parse_retry_after(candidate)
            if seconds is not None:
                return seconds

        seconds = parse_retry_after(str(node))
        if seconds is not None:
            return seconds

        for attribute in ("__cause__", "__context__"):
            try:
                nested = getattr(node, attribute, None)
            except Exception:
                nested = None
            if nested is not None:
                seconds = walk(nested, depth + 1)
                if seconds is not None:
                    return seconds
        return None

    try:
        return walk(exc, 0)
    except Exception:
        return None


def compute_backoff(
    attempt: int,
    base: float = 2.0,
    factor: float = 1.2,
    cap: float = 60.0,
    retry_after: Optional[float] = None,
    jitter: float = 0.25,
    rng: Optional[random.Random] = None,
) -> float:
    """Bounded exponential backoff, honouring ``Retry-After`` when present.

    The server's value wins (it is authoritative); the exponential value is used
    otherwise. Jitter is symmetric and bounded by ``jitter`` × delay.
    """
    if retry_after is not None:
        return min(float(retry_after), cap)
    attempt = max(1, int(attempt))
    delay = min(base * (factor ** (attempt - 1)), cap)
    if jitter:
        generator = rng or random
        delay *= 1.0 + generator.uniform(-jitter, jitter)
    return max(0.0, min(delay, cap))


# ── Per-session budget ───────────────────────────────────────────────────────
class BudgetExceeded(RuntimeError):
    """Raised when a session has spent its call or token allowance."""


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class BudgetPolicy:
    max_calls: Optional[int] = 200
    max_tokens: Optional[int] = 2_000_000
    calls: int = 0
    tokens: int = 0

    @classmethod
    def from_env(cls) -> "BudgetPolicy":
        return cls(
            max_calls=_env_int("VM_MAX_CALLS_PER_SESSION", 200),
            max_tokens=_env_int("VM_MAX_TOKENS_PER_SESSION", 2_000_000),
        )

    def check(self, tokens: int = 0) -> None:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise BudgetExceeded(
                f"Session request budget exhausted ({self.calls}/{self.max_calls} calls). "
                "Start a new session or raise VM_MAX_CALLS_PER_SESSION."
            )
        if self.max_tokens is not None and (self.tokens + max(0, tokens)) > self.max_tokens:
            raise BudgetExceeded(
                f"Session token budget exhausted ({self.tokens}/{self.max_tokens} tokens). "
                "Start a new session or raise VM_MAX_TOKENS_PER_SESSION."
            )

    def record(self, tokens: int = 0) -> None:
        self.calls += 1
        self.tokens += max(0, int(tokens or 0))

    def status(self) -> Dict[str, Any]:
        remaining_calls = None if self.max_calls is None else max(0, self.max_calls - self.calls)
        remaining_tokens = None if self.max_tokens is None else max(0, self.max_tokens - self.tokens)
        return {
            "calls": self.calls,
            "max_calls": self.max_calls,
            "remaining_calls": remaining_calls,
            "tokens": self.tokens,
            "max_tokens": self.max_tokens,
            "remaining_tokens": remaining_tokens,
            "exhausted": (
                (self.max_calls is not None and self.calls >= self.max_calls)
                or (self.max_tokens is not None and self.tokens >= self.max_tokens)
            ),
        }


_BUDGETS: Dict[str, BudgetPolicy] = {}


def session_budget(session_id: str = "default") -> BudgetPolicy:
    """Budget policy for one session (created on first use)."""
    key = str(session_id or "default")
    if key not in _BUDGETS:
        _BUDGETS[key] = BudgetPolicy.from_env()
    return _BUDGETS[key]


def reset_session_budget(session_id: str = "default") -> BudgetPolicy:
    _BUDGETS[str(session_id or "default")] = BudgetPolicy.from_env()
    return _BUDGETS[str(session_id or "default")]


def budget_status(session_id: str = "default") -> Dict[str, Any]:
    return session_budget(session_id).status()


# ── Per-call accounting (no credential values, ever) ─────────────────────────
@dataclass
class CallRecord:
    capability: str
    model: str
    provider: str = "google-gemini"
    outcome: str = "ok"
    tokens: Optional[int] = None
    retry_after: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


_CALL_LOG: deque = deque(maxlen=200)


def record_call(
    capability: str,
    model: str,
    outcome: str = "ok",
    tokens: Optional[int] = None,
    retry_after: Optional[float] = None,
    provider: str = "google-gemini",
) -> Dict[str, Any]:
    """Record one model call. Never accepts or stores credentials."""
    entry = CallRecord(
        capability=capability,
        model=model,
        provider=provider,
        outcome=outcome,
        tokens=tokens,
        retry_after=retry_after,
    ).as_dict()
    _CALL_LOG.append(entry)
    return entry


def call_history() -> List[Dict[str, Any]]:
    return list(_CALL_LOG)


def clear_call_history() -> None:
    _CALL_LOG.clear()


def format_call_log(limit: int = 10) -> str:
    """Human-readable tail of the call log (used by the sidebar/diagnostics)."""
    entries = call_history()[-limit:]
    if not entries:
        return "No model calls recorded in this session."
    lines = []
    for entry in entries:
        tokens = "" if entry["tokens"] is None else f", {entry['tokens']} tokens"
        retry = "" if entry["retry_after"] is None else f", retry-after {entry['retry_after']}s"
        lines.append(
            f"- {entry['timestamp']} · {entry['capability']} · {entry['model']} · "
            f"{entry['outcome']}{tokens}{retry}"
        )
    return "\n".join(lines)


def response_tokens(response: Any) -> Optional[int]:
    """Best-effort token count from a provider response (never raises)."""
    for attribute in ("usage_metadata", "usage"):
        usage = getattr(response, attribute, None)
        if usage is None:
            continue
        for field_name in ("total_token_count", "total_tokens"):
            value = getattr(usage, field_name, None) or (
                usage.get(field_name) if isinstance(usage, dict) else None
            )
            if isinstance(value, int):
                return value
    return None
