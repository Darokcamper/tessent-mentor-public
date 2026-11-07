"""
session_logger.py

Persistent, human-readable session logging for the Tessent Mentor AI app.

Every interaction (question asked, AI answer, generated question bank, your viva
answer, the AI's evaluation) is appended to a timestamped, category-separated
log file under project_root/session_logs/ so that nothing is ever lost.

Key design:
- One log file per date: e.g. session_logs/2026-09-02.log
- Each entry is a clearly delimited block with a header, type, timestamp, and body.
- Entries are appended immediately (flush) so data survives a crash/restart.
- Auto-syncs to private GitHub repo after each entry (if configured).
"""

import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "session_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Log clock / day boundary ───────────────────────────────────────────────────
# Streamlit Cloud containers run in UTC. A UTC-based daily file therefore rolled over at
# 05:30 IST, so the previous day's file kept growing for the first five and a half hours of
# the local day ("yesterday's log is being edited today"), and the timestamps written into
# the entries did not match the owner's clock.
#
# SESSION_LOG_TZ (env var or Streamlit secret, e.g. "Asia/Kolkata") now decides both the
# file date and the timestamps. Default is Asia/Kolkata; an unknown value falls back to the
# system clock rather than crashing.
DEFAULT_LOG_TZ = "Asia/Kolkata"
_LOG_TZ_REPORTED = None


def _log_tz_name() -> str:
    name = os.getenv("SESSION_LOG_TZ", "").strip()
    if name:
        return name
    try:
        import streamlit as st

        value = st.secrets.get("SESSION_LOG_TZ", "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return DEFAULT_LOG_TZ


def _log_tz():
    """Resolved tzinfo, or None when the name is unknown (caller uses system time)."""
    global _LOG_TZ_REPORTED
    name = _log_tz_name()
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(name)
        _LOG_TZ_REPORTED = name
        return tz
    except Exception:
        _LOG_TZ_REPORTED = None
        return None


def _now() -> datetime:
    """Current time in the configured log timezone (system time if it cannot be resolved)."""
    tz = _log_tz()
    return datetime.now(tz) if tz is not None else datetime.now()


def log_clock_info() -> dict:
    """What the log clock is doing, for display in the sidebar."""
    tz = _log_tz()
    now = _now()
    return {
        "timezone": _LOG_TZ_REPORTED or "system",
        "now": now.strftime("%Y-%m-%d %H:%M:%S"),
        "active_file": f"{now.strftime('%Y-%m-%d')}.log",
    }

# Keep a file handle cache so we don't reopen constantly.
_handles = {}

ATTACHMENT_DIR = LOG_DIR / "attachments"
ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Content-sensitive logging guard (WP-10 privacy policy)
#
# Policy: uploaded presentation files, their file names, absolute paths, attachment
# metadata and extracted slide text must NEVER reach the GitHub session logger or its
# upload payload. Ordinary, non-sensitive interaction metadata keeps being logged.
#
# The GitHub sync uploads the WHOLE daily log file, so the only safe design is to never
# write the sensitive data into that file. Two layers enforce it:
#   1. ``log_qa(content_sensitive=True)`` writes a content-free marker instead of the
#      attachment name/path/preview.
#   2. ``_auto_sync_to_github`` refuses to upload a file whose text still matches a
#      sensitive pattern (defence in depth for future callers).
# ---------------------------------------------------------------------------

CONTENT_SENSITIVE_REDACTION = "[redacted: content-sensitive upload]"

_SENSITIVE_FIELD_KEYS = ("ATTACHMENT", "ATTACHMENT STORED AT", "ATTACHMENT PREVIEW")

# Exam-bank protection (release decision 2026-11): parsed exam questions, their
# option text and the answer material must never be written to the log, and
# therefore never reach the GitHub upload payload or the webhook. Under the fields
# below only a question number plus a correctness word is legitimate.
EXAM_CONTENT_REDACTION = "[redacted: exam-bank content]"

_EXAM_FIELD_KEYS = ("ANSWERS", "EXAM ANSWERS")
_EXAM_OUTCOME_LINE = re.compile(r"^Q\d{1,3}:\s*(?:CORRECT|INCORRECT|UNSCORED)$")

# Patterns that indicate identifying or content-bearing data leaked into the log.
_LEAK_PATTERNS = (
    ("absolute_windows_path", re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|]+")),
    ("absolute_posix_path", re.compile(r"(?<![\w.])/(?:Users|home|mnt|var|tmp)/[^\s\"'<>|]+")),
    ("attachment_store_path", re.compile(r"attachments[\\/][0-9]{8}_[0-9a-f]{6,}_")),
    ("slide_file_reference", re.compile(r"[\w .\-]{1,120}\.(?:pptx|ppt|potx)(?![a-z0-9])", re.IGNORECASE)),
)


def find_sensitive_leaks(text: str) -> list:
    """Return human-readable findings for identifying/content data in *text*.

    A redacted block (``CONTENT_SENSITIVE_REDACTION``) produces no findings: it carries
    no name, path or content by construction.
    """
    body = str(text or "")
    if not body.strip():
        return []

    findings = []
    for label, pattern in _LEAK_PATTERNS:
        for match in pattern.finditer(body):
            snippet = match.group(0)
            # The redaction marker itself is not a leak.
            if CONTENT_SENSITIVE_REDACTION in snippet or snippet.strip() == CONTENT_SENSITIVE_REDACTION:
                continue
            findings.append(f"{label}: {snippet[:120]}")

    # Attachment fields must either be absent or redacted - never carry a payload.
    # Divider lines and blanks are structural, not content.
    current_key = None
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_key = stripped[1:-1].strip().upper()
            continue
        if not stripped or re.fullmatch(r"[-=]{3,}", stripped):
            continue
        if current_key in _SENSITIVE_FIELD_KEYS:
            if CONTENT_SENSITIVE_REDACTION in stripped:
                continue
            findings.append(f"{current_key.lower()}_payload: {stripped[:120]}")
        # Exam-bank fields: only "Q<n>: CORRECT|INCORRECT|UNSCORED" (or an explicit
        # redaction marker) is allowed. Anything else is exam content.
        if current_key in _EXAM_FIELD_KEYS:
            if EXAM_CONTENT_REDACTION in stripped or CONTENT_SENSITIVE_REDACTION in stripped:
                continue
            if _EXAM_OUTCOME_LINE.match(stripped):
                continue
            findings.append(f"{current_key.lower()}_payload: {stripped[:120]}")
    return findings

# Auto-sync configuration
_AUTO_SYNC_ENABLED = True
_AUTO_SYNC_REPO_URL = "https://github.com/Darokcamper/tessent_ai.git"
_AUTO_SYNC_BRANCH = "main"
_AUTO_SYNC_REMOTE = "origin"


def _get_github_token():
    """Session-log sync token: **GITHUB_TOKEN only**.

    Release decision (2026-11): session-log synchronization and the private
    knowledge-bundle download are separate purposes with separate credentials.
    ``KB_BUNDLE_TOKEN`` authenticates bundle downloads (see ``core/bootstrap.py``)
    and must never enable an upload here. ``GH_TOKEN`` and the previous
    "any secret that looks like a token" scan are no longer consulted either:
    both could pick up the bundle credential and silently start uploading logs.

    Returns '' when log synchronization is not explicitly configured, which keeps
    the sync disabled. Local logging is unaffected.
    """
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        return token
    try:
        import streamlit as st
        secrets = st.secrets
        if secrets is None:
            return ""
        name = "GITHUB_TOKEN"
        # 1. Direct top-level check
        try:
            if name in secrets and not isinstance(secrets[name], (dict, list)):
                return str(secrets[name]).strip()
        except Exception:
            pass
        # 2. Check inside sub-sections (e.g. if written under [auth])
        for k in secrets:
            sub = secrets[k]
            if isinstance(sub, dict) or hasattr(sub, "items"):
                try:
                    if name in sub and not isinstance(sub[name], (dict, list)):
                        return str(sub[name]).strip()
                except Exception:
                    pass
        return ""
    except Exception:
        return ""


DIVIDER = "=" * 78
_ENTRY_TIME_RE = re.compile(r"^TIME\s*:\s*(.+?)\s*$", re.M)


def split_entries(text: str) -> list:
    """Split a daily log into its entries (the ``====`` separated blocks _write_block emits)."""
    entries = []
    current = []
    for line in str(text or "").splitlines():
        if line.rstrip() == DIVIDER:
            if current:
                entries.append("\n".join(current).strip())
                current = []
            current.append(line.rstrip())
            continue
        if current:
            current.append(line.rstrip())
    if current:
        entries.append("\n".join(current).strip())
    return [entry for entry in entries if entry.strip(DIVIDER).strip()]


def _entry_time(entry: str) -> str:
    match = _ENTRY_TIME_RE.search(entry)
    return match.group(1) if match else ""


def merge_log_texts(*texts: str) -> str:
    """Union of log entries across versions, de-duplicated and ordered by entry time.

    Why this exists: the app appends to a **container-local** daily file and used to upload
    that file wholesale. A fresh container starts with an empty file, so its first upload
    replaced everything an earlier container had pushed for the same day (observed
    2026-11-06: 6 entries -> 1). Merging before upload makes the remote file additive, so
    the archive no longer depends on recovering old versions from git.
    """
    seen = {}
    for text in texts:
        for entry in split_entries(text or ""):
            key = re.sub(r"\s+", " ", entry).strip()
            if key and key not in seen:
                seen[key] = (entry, len(seen))
    if not seen:
        return ""
    ordered = sorted(seen.values(), key=lambda pair: (_entry_time(pair[0]), pair[1]))
    return "\n\n".join(entry for entry, _position in ordered) + "\n"


# ── Last-sync visibility (so the owner can see it in the app, not just in GitHub) ──
# One dict, replaced atomically on every attempt. Read through last_sync_status().
_last_sync: dict = {}


def _record_sync(status: str, *, entries: int = None, detail: str = "", filename: str = "") -> None:
    """Remember the outcome of the most recent upload attempt for the sidebar panel."""
    global _last_sync
    _last_sync = {
        "status": status,                 # synced | up_to_date | blocked | error | disabled
        "at": _now().strftime("%H:%M:%S"),
        "entries": entries,
        "detail": detail,
        "filename": filename,
    }


def last_sync_status() -> dict:
    """Copy of the last upload outcome: status, time, entry count and a short detail."""
    return dict(_last_sync)


def _auto_sync_to_github(filepath: Path):
    """Auto-commit and push the log file to the private GitHub repo via GitHub REST API."""
    if not _AUTO_SYNC_ENABLED:
        _record_sync("disabled", detail="auto-sync is switched off")
        return

    token = _get_github_token()
    if not token:
        _record_sync("disabled", detail="no GITHUB_TOKEN configured")
        return

    def _sync():
        import base64
        import json
        import urllib.request
        import urllib.error

        try:
            filename = filepath.name
            api_url = f"https://api.github.com/repos/Darokcamper/tessent_ai/contents/session_logs/{filename}"
            content_bytes = filepath.read_bytes()

            # WP-10 privacy guard: never upload identifying/content data. The daily log is
            # uploaded whole, so anything sensitive written into it would leave the machine.
            # If a leak pattern is present the upload is skipped (local logging continues).
            try:
                decoded = content_bytes.decode("utf-8", errors="replace")
            except Exception:
                decoded = ""
            leaks = find_sensitive_leaks(decoded)
            if leaks:
                _record_sync("blocked", entries=len(split_entries(decoded)),
                             filename=filepath.name,
                             detail=f"content-sensitivity guard: {leaks[0][:80]}")
                print(
                    "[session_logger] Upload blocked by the content-sensitivity guard "
                    f"({len(leaks)} finding(s)); the local log is unchanged. "
                    f"First finding: {leaks[0]}"
                )
                try:
                    import streamlit as st

                    st.warning(
                        "⚠️ Session-log upload paused: the log contains data classified as "
                        "content-sensitive (file names, absolute paths or extracted text). "
                        "Local logging continues."
                    )
                except Exception:
                    pass
                return

            # Fetch the current remote file: we need its sha to update it AND its content so
            # this upload can ADD to the day's archive instead of replacing it.
            sha = None
            remote_text = ""
            req_get = urllib.request.Request(
                api_url,
                headers={"Authorization": f"Bearer {token}", "User-Agent": "TessentMentorAI"}
            )
            try:
                with urllib.request.urlopen(req_get, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    sha = data.get("sha")
                    if data.get("encoding") == "base64" and data.get("content"):
                        remote_text = base64.b64decode(data["content"]).decode(
                            "utf-8", errors="replace")
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    print(f"[session_logger] GET {filename} error {e.code}: {e.reason}")
            except Exception:
                pass

            # Union of what is already on GitHub and what this container has logged. Entries
            # from earlier containers are preserved instead of being overwritten.
            merged_text = merge_log_texts(remote_text, decoded) or decoded
            if remote_text and merged_text.strip() == remote_text.strip():
                _log_entries = len(split_entries(remote_text))
                print(
                    f"[session_logger] {filename}: GitHub already holds every entry "
                    f"({_log_entries}); nothing to upload."
                )
                _record_sync("up_to_date", entries=_log_entries, filename=filename,
                             detail="remote already has every entry")
                return

            leaks = find_sensitive_leaks(merged_text)
            if leaks:
                _record_sync("blocked", entries=len(split_entries(merged_text)),
                             filename=filename,
                             detail=f"content-sensitivity guard: {leaks[0][:80]}")
                print(
                    "[session_logger] Upload blocked by the content-sensitivity guard "
                    f"({len(leaks)} finding(s)); the local log is unchanged. "
                    f"First finding: {leaks[0]}"
                )
                try:
                    import streamlit as st

                    st.warning(
                        "⚠️ Session-log upload paused: the log contains data classified as "
                        "content-sensitive (file names, absolute paths or extracted text). "
                        "Local logging continues."
                    )
                except Exception:
                    pass
                return

            content_b64 = base64.b64encode(merged_text.encode("utf-8")).decode("utf-8")
            payload = {
                "message": (f"auto-sync: {filename} @ {_now().strftime('%Y-%m-%d %H:%M:%S')}"
                            f" ({len(split_entries(merged_text))} entries)"),
                "content": content_b64,
                "branch": _AUTO_SYNC_BRANCH
            }
            if sha:
                payload["sha"] = sha

            req_put = urllib.request.Request(
                api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "User-Agent": "TessentMentorAI"
                },
                method="PUT"
            )
            with urllib.request.urlopen(req_put, timeout=15) as resp:
                _count = len(split_entries(merged_text))
                print(
                    f"[session_logger] Auto-synced {filename} to GitHub successfully! "
                    f"({_count} entries, merged with the remote copy)"
                )
                _record_sync("synced", entries=_count, filename=filename,
                             detail="merged with the remote copy")

        except urllib.error.HTTPError as e:
            _record_sync("error", filename=filepath.name,
                         detail=f"HTTP {e.code}" + (
                             " - token needs Contents: Read and write"
                             if e.code == 403 else
                             " - another upload changed the file first" if e.code == 409 else ""))
            if e.code == 403:
                print(f"[session_logger] Auto-sync 403: Token needs 'Contents: Read and write' on Darokcamper/tessent_ai to push logs.")
            else:
                print(f"[session_logger] Auto-sync HTTP error {e.code}: {e.reason}")
        except Exception as e:
            _record_sync("error", filename=filepath.name,
                         detail=f"{type(e).__name__}: {str(e)[:80]}")
            print(f"[session_logger] Auto-sync error: {e}")

    # Run in background thread to not block the UI
    thread = threading.Thread(target=_sync, daemon=True)
    thread.start()


def _send_webhook(section: str, user: str, field_lines: list):
    """Optional instant alert to owner via Discord / Telegram / Slack webhook."""
    webhook_url = os.getenv("WEBHOOK_URL", "")
    if not webhook_url:
        try:
            import streamlit as st
            webhook_url = str(st.secrets.get("WEBHOOK_URL", ""))
        except Exception:
            webhook_url = ""
    if not webhook_url:
        return

    def _post():
        import json
        import urllib.request
        try:
            summary = ""
            for k, v in field_lines:
                if k in ("QUESTION", "MY ANSWER", "PLAN", "COMMAND / SCRIPT"):
                    summary = str(v)[:300]
                    break
            payload = json.dumps({
                "content": f"📝 **[Tessent Mentor AI]** `{section}` by `{user}`:\n> {summary}"
            }).encode("utf-8")
            req = urllib.request.Request(
                webhook_url,
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "TessentMentorAI/1.0"}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass

    threading.Thread(target=_post, daemon=True).start()


def save_attachment(bytes_or_path: bytes, filename: str, source_folder: Path = ATTACHMENT_DIR,
                    persist: bool = True) -> str:
    """Persist an uploaded attachment into the session attachments folder.

    Returns the absolute path of the saved copy. The stored filename is prefixed
    with a timestamp and a random tag so different files with the same name do not
    overwrite each other, and so we can correlate them back to a log entry.

    ``persist=False`` keeps a content-sensitive upload out of the session directory
    entirely (WP-10 privacy policy): nothing is written and an empty string is
    returned, so no name or path can be logged or uploaded later.
    """
    import shutil

    if not persist:
        return ""

    safe_name = Path(filename).name
    stamp = _now().strftime("%Y%m%d_%H%M%S")
    token = os.urandom(4).hex()
    stored_name = f"{stamp}_{token}_{safe_name}"
    stored_path = source_folder / stored_name
    try:
        if isinstance(bytes_or_path, (bytes, bytearray)):
            with open(stored_path, "wb") as f:
                f.write(bytes(bytes_or_path))
        else:
            shutil.copyfile(bytes_or_path, stored_path)
        return str(stored_path)
    except Exception:
        return str(filename) if isinstance(filename, str) else ""


def _today_path() -> Path:
    return LOG_DIR / f"{_now().strftime('%Y-%m-%d')}.log"


def _get_handle() -> object:
    """Returns an open text file handle for today's log file (append mode)."""
    path = _today_path()
    key = str(path)
    if key not in _handles or _handles[key].closed:
        _handles[key] = open(path, "a", encoding="utf-8")
    return _handles[key]


def _divider(char: str = "=", width: int = 78) -> str:
    return char * width


def _write_block(section: str, field_lines: list, notify_webhook: bool = True) -> None:
    """Writes a formatted LOG ENTRY block for a given section.

    ``notify_webhook=False`` suppresses the instant-notification copy of the entry; a
    content-sensitive interaction must not be mirrored to another destination either.
    """
    ts = _now().strftime("%Y-%m-%d %H:%M:%S")
    user = "user"
    try:
        from core.auth import current_user
        user = current_user()
    except Exception:
        pass

    fh = _get_handle()
    lines = [
        _divider("="),
        f"TYPE   : {section}",
        f"TIME   : {ts}",
        f"USER   : {user}",
        _divider("-"),
    ]
    for key, body in field_lines:
        lines.append(f"[{key}]")
        lines.append(str(body).rstrip())
        lines.append(_divider("-"))
    lines.append("")
    lines.append("")
    try:
        fh.write("\n".join(lines))
        fh.flush()
    except Exception:
        # Never crash the UI because of logging.
        try:
            fh = None
        except Exception:
            pass
        # Write to a fallback file if the primary fails.
        try:
            fallback = LOG_DIR / "fallback.log"
            with open(fallback, "a", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception:
            pass

    # Optional webhook notification
    if notify_webhook:
        try:
            _send_webhook(section, user, field_lines)
        except Exception:
            pass

    # Auto-sync to GitHub after each entry
    try:
        _auto_sync_to_github(_today_path())
    except Exception:
        pass


def close():
    """Closes all open log file handles (call on app shutdown if desired)."""
    for k, h in list(_handles.items()):
        try:
            h.close()
        except Exception:
            pass
    _handles.clear()


# --------------------------------------------------------------------------
# Public logging helpers (one per mode in the UI)
# --------------------------------------------------------------------------

def log_qa(question: str, answer: str, attachment_name: str = None,
           attachment_path: str = None, attachment_preview: str = None,
           content_sensitive: bool = False) -> None:
    """Logs a question + AI answer from the 'Ask Question' mode.

    If the question came with an attachment, include its name, stored path, and a
    short text preview so the log shows exactly what document/image was used.

    ``content_sensitive=True`` implements the confirmed WP-10 privacy policy: for an
    upload whose content must not leave the application (presentations, extracted
    slide text), only a content-free marker is written. No file name, no path and no
    preview ever reach the log file — and therefore never reach the GitHub upload
    payload that mirrors that file.
    """
    fields = [
        ("QUESTION", question),
        ("AI ANSWER", answer),
    ]
    if content_sensitive:
        if attachment_name or attachment_path or attachment_preview:
            fields.append((
                "ATTACHMENT",
                f"{CONTENT_SENSITIVE_REDACTION} "
                "(presentation content is processed locally and excluded from session logging)",
            ))
    else:
        if attachment_name:
            fields.append(("ATTACHMENT", attachment_name))
        if attachment_path:
            fields.append(("ATTACHMENT STORED AT", attachment_path))
        if attachment_preview:
            fields.append(("ATTACHMENT PREVIEW", attachment_preview))
    _write_block("QA / ASK QUESTION", fields, notify_webhook=not content_sensitive)


def log_ppt_explainer(slide_count: int, explained_count: int, outcome: str = "ok") -> None:
    """Content-free record of a PPT Explainer run (WP-10 privacy policy).

    Deliberately records counts and outcome only: no file name, no path, no slide text
    and no model prompt. The webhook is not notified for this section.
    """
    _write_block("PPT EXPLAINER", [
        ("SLIDES", str(int(slide_count))),
        ("EXPLAINED", str(int(explained_count))),
        ("OUTCOME", str(outcome)),
    ], notify_webhook=False)


def log_qa_error(question: str, error: str) -> None:
    """Logs a question that hit an error in the 'Ask Question' mode."""
    _write_block("QA / ERROR", [
        ("QUESTION", question),
        ("ERROR", error),
    ])


def log_lab_explainer(module: str, script: str, explanation: str) -> None:
    """Logs a lab/command explanation."""
    _write_block("LAB & COMMAND EXPLAINER", [
        ("MODULE", module),
        ("COMMAND / SCRIPT", script),
        ("EXPLANATION", explanation),
    ])


def log_lab_explainer_error(module: str, script: str, error: str) -> None:
    """Logs a failed lab/command explanation."""
    _write_block("LAB & COMMAND EXPLAINER / ERROR", [
        ("MODULE", module),
        ("COMMAND / SCRIPT", script),
        ("ERROR", error),
    ])


def log_question_bank(module: str, num_questions: int, q_bank: str) -> None:
    """Logs a generated assessment/viva question bank."""
    _write_block("ASSESSMENT QUESTION BANK", [
        ("MODULE", module),
        ("NUMBER OF QUESTIONS", num_questions),
        ("QUESTION BANK", q_bank),
    ])


def log_viva_question(topic, difficulty, question: str) -> None:
    """Logs a generated mock viva question."""
    _write_block("VIVA - GENERATED QUESTION", [
        ("TOPIC", topic),
        ("DIFFICULTY", difficulty),
        ("QUESTION", question),
    ])


def log_viva_answer_and_evaluation(question: str, candidate_answer: str,
                                   evaluation: str) -> None:
    """Logs a viva candidate answer along with the AI's evaluated feedback."""
    _write_block("VIVA - MY ANSWER + EVALUATION", [
        ("QUESTION", question),
        ("MY ANSWER", candidate_answer),
        ("EVALUATION / FEEDBACK", evaluation),
    ])


def log_study_plan(plan: str) -> None:
    """Logs a generated personalized study plan."""
    _write_block("STUDY PLAN", [
        ("PLAN", plan),
    ])


def log_deep_study(kind: str, module: str, item: str, content: str) -> None:
    """Logs a deep-study report for a video lesson or lab exercise."""
    tag = "VIDEO" if kind == "video" else "LAB"
    _write_block("DEEP STUDY - " + tag, [
        ("MODULE", str(module)),
        ("ITEM", str(item)),
        ("CONTENT", content),
    ])


def log_exam_quiz(topic: str, score: int, total: int, pct: float, details: list = None) -> None:
    """Logs an exam-bank quiz attempt without any exam content (release decision 2026-11).

    Only the score and a per-question outcome are recorded: the question text, the
    option text and the answer material never enter the log file, so they can never
    reach the GitHub upload payload or the webhook. The webhook is not notified for
    this section at all.

    NOTE (found during the 2026-11 release audit): this function previously built
    ``fields`` and returned without writing anything, so exam attempts were never
    logged. The write call below restores the intended behaviour with the safe
    field set.
    """
    fields = [
        ("TOPIC", topic or "All"),
        ("SCORE", f"{score}/{total} ({pct:.1f}%)"),
    ]
    if details:
        detail_lines = []
        for q_num, _user_ans, is_correct in details:
            status = "CORRECT" if is_correct else "INCORRECT"
            detail_lines.append(f"Q{q_num}: {status}")
        fields.append(("ANSWERS", "\n".join(detail_lines)))
    _write_block("EXAM BANK QUIZ", fields, notify_webhook=False)
