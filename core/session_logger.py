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
import sys
import threading
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "session_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Keep a file handle cache so we don't reopen constantly.
_handles = {}

ATTACHMENT_DIR = LOG_DIR / "attachments"
ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)

# Auto-sync configuration
_AUTO_SYNC_ENABLED = True
_AUTO_SYNC_REPO_URL = "https://github.com/Darokcamper/tessent_ai.git"
_AUTO_SYNC_BRANCH = "main"
_AUTO_SYNC_REMOTE = "origin"


def _get_github_token():
    """Get GitHub token from environment or Streamlit secrets (top-level or nested)."""
    token = os.getenv("GITHUB_TOKEN", "") or os.getenv("KB_BUNDLE_TOKEN", "")
    if token:
        return token
    try:
        import streamlit as st
        secrets = st.secrets
        if secrets is None:
            return ""
        # 1. Direct top-level check
        for name in ("GITHUB_TOKEN", "KB_BUNDLE_TOKEN", "GH_TOKEN"):
            try:
                if name in secrets and not isinstance(secrets[name], (dict, list)):
                    return str(secrets[name]).strip()
            except Exception:
                pass
        # 2. Check inside sub-sections (e.g. if written under [auth])
        for k in secrets:
            sub = secrets[k]
            if isinstance(sub, dict) or hasattr(sub, "items"):
                for name in ("GITHUB_TOKEN", "KB_BUNDLE_TOKEN", "GH_TOKEN"):
                    try:
                        if name in sub and not isinstance(sub[name], (dict, list)):
                            return str(sub[name]).strip()
                    except Exception:
                        pass
        # 3. Dynamic scan for token string pattern
        for k in secrets:
            v = str(secrets[k]).strip()
            if v.startswith("github_pat_") or v.startswith("ghp_"):
                return v
        return ""
    except Exception:
        return ""


def _auto_sync_to_github(filepath: Path):
    """Auto-commit and push the log file to the private GitHub repo via GitHub REST API."""
    if not _AUTO_SYNC_ENABLED:
        return

    token = _get_github_token()
    if not token:
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
            content_b64 = base64.b64encode(content_bytes).decode("utf-8")

            # Check if file exists on GitHub to obtain its sha for update
            sha = None
            req_get = urllib.request.Request(
                api_url,
                headers={"Authorization": f"Bearer {token}", "User-Agent": "TessentMentorAI"}
            )
            try:
                with urllib.request.urlopen(req_get, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    sha = data.get("sha")
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    print(f"[session_logger] GET {filename} error {e.code}: {e.reason}")
            except Exception:
                pass

            payload = {
                "message": f"auto-sync: {filename} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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
                print(f"[session_logger] Auto-synced {filename} to GitHub successfully!")

        except urllib.error.HTTPError as e:
            if e.code == 403:
                print(f"[session_logger] Auto-sync 403: Token needs 'Contents: Read and write' on Darokcamper/tessent_ai to push logs.")
            else:
                print(f"[session_logger] Auto-sync HTTP error {e.code}: {e.reason}")
        except Exception as e:
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


def save_attachment(bytes_or_path: bytes, filename: str, source_folder: Path = ATTACHMENT_DIR) -> str:
    """Persist an uploaded attachment into the session attachments folder.

    Returns the absolute path of the saved copy. The stored filename is prefixed
    with a timestamp and a random tag so different files with the same name do not
    overwrite each other, and so we can correlate them back to a log entry.
    """
    import shutil

    safe_name = Path(filename).name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    return LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"


def _get_handle() -> object:
    """Returns an open text file handle for today's log file (append mode)."""
    path = _today_path()
    key = str(path)
    if key not in _handles or _handles[key].closed:
        _handles[key] = open(path, "a", encoding="utf-8")
    return _handles[key]


def _divider(char: str = "=", width: int = 78) -> str:
    return char * width


def _write_block(section: str, field_lines: list) -> None:
    """Writes a formatted LOG ENTRY block for a given section."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
           attachment_path: str = None, attachment_preview: str = None) -> None:
    """Logs a question + AI answer from the 'Ask Question' mode.

    If the question came with an attachment, include its name, stored path, and a
    short text preview so the log shows exactly what document/image was used.
    """
    fields = [
        ("QUESTION", question),
        ("AI ANSWER", answer),
    ]
    if attachment_name:
        fields.append(("ATTACHMENT", attachment_name))
    if attachment_path:
        fields.append(("ATTACHMENT STORED AT", attachment_path))
    if attachment_preview:
        fields.append(("ATTACHMENT PREVIEW", attachment_preview))
    _write_block("QA / ASK QUESTION", fields)


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
    """Logs an exam bank quiz attempt with score and details."""
    fields = [
        ("TOPIC", topic or "All"),
        ("SCORE", f"{score}/{total} ({pct:.1f}%)"),
    ]
    if details:
        detail_lines = []
        for q_num, user_ans, is_correct in details:
            status = "CORRECT" if is_correct else "INCORRECT"
            detail_lines.append(f"Q{q_num}: {user_ans} [{status}]")
        fields.append(("ANSWERS", "\n".join(detail_lines)))
    _write_block("EXAM QUIZ ATTEMPT", fields)