import os
import re
import json
import uuid
from pathlib import Path
import streamlit as st

from agents.router_agent import route_question
from agents.scan_agent import ask_scan
from agents.atpg_agent import ask_atpg
from agents.edt_agent import ask_edt
from agents.mbist_agent import ask_mbist
from agents.jtag_agent import ask_jtag
from agents.ijtag_agent import ask_ijtag
from agents.wrapper_agent import ask_wrapper
from agents.occ_agent import ask_occ
from agents.sta_agent import ask_sta
from agents.boundaryscan_agent import ask_boundaryscan
from agents.gls_agent import ask_gls
from agents.linux_agent import ask_linux
from agents.tcl_agent import ask_tcl
from agents.general_agent import ask_general

from agents.interviewer_agent import ask_interview_question
from agents.evaluator_agent import evaluate_answer as evaluate_viva_answer
from agents.planner_agent import generate_study_plan
from agents.lab_explainer_agent import explain_lab_command
from agents.question_bank_agent import generate_assessment_question_bank
from agents.deep_study_agent import (
    list_video_modules, list_lab_exercises,
    list_video_lessons, explain_video_lesson_detail, lesson_slug,
    explain_video_lesson, explain_lab_exercise, explain_video_lesson_detail_v2,
    explain_lab_exercise_v2, list_lab_files_at_once,
    generate_lab_ppt_from_report, generate_lesson_ppt_from_report,
    extract_rapid_viva_items, check_viva_attempt, learning_path_position,
)
from agents.crossref_agent import (crossref_lab, where_used, format_crossref_table,
                                   related_labs_for_display, module_for_lab_display)
from agents.exam_bank_agent import (
    evaluate_answer as evaluate_exam_answer,
    default_include_unverified as default_exam_include_unverified,
    get_corpus_status as get_exam_corpus_status,
    get_filter_options as get_exam_filter_options,
    get_review_topics as get_exam_review_topics,
    load_questions as load_exam_questions,
    select_questions as select_exam_questions,
    search_questions as search_exam_questions,
    coverage_report as exam_coverage_report,
    export_questions_markdown as export_exam_markdown,
)
from core.rag_builder import upload_and_index_pdf, index_text_file
from core.command_search import search_commands, get_available_filter_options
from core.command_indexer import load_command_index
from agents.attachment_agent import ask_with_text_attachment, ask_with_image
from core.file_reader import document_text_to_string, is_text_ext, is_image_ext
from core.auth import (
    require_auth, auth_badge_sidebar, is_owner, owner_email,
    owner_verified, require_owner,
)
from core.batch_exporter import BatchExporter, ExportMode
from core.usage_panel import render_usage_panel
try:
    from core.llm import get_gemini_keys as _get_gemini_keys
except ImportError:
    # Older deployed core/llm.py may only expose GEMINI_KEYS list.
    from core.llm import GEMINI_KEYS as _LLM_STATIC_KEYS  # noqa: F401

    def _get_gemini_keys():  # type: ignore[no-redef]
        try:
            return list(_LLM_STATIC_KEYS or [])
        except Exception:
            return []
try:
    from core.session_logger import (
        log_qa, log_qa_error, save_attachment,
        log_lab_explainer, log_lab_explainer_error,
        log_question_bank,
        log_viva_question, log_viva_answer_and_evaluation,
        log_study_plan, log_deep_study, log_exam_quiz,
    )
    from core.chat_transcript import format_chat_transcript
    from core.chat_state import drop_last_answer, truncate_for_edit
except ImportError:
    try:
        import sys, importlib
        if "core.session_logger" in sys.modules:
            importlib.reload(sys.modules["core.session_logger"])
        from core.session_logger import (
            log_qa, log_qa_error, save_attachment,
            log_lab_explainer, log_lab_explainer_error,
            log_question_bank,
            log_viva_question, log_viva_answer_and_evaluation,
            log_study_plan, log_deep_study, log_exam_quiz,
        )
        from core.chat_transcript import format_chat_transcript
        from core.chat_state import drop_last_answer, truncate_for_edit
    except Exception:
        from core.session_logger import (
            log_qa, log_qa_error, save_attachment,
            log_lab_explainer, log_lab_explainer_error,
            log_question_bank,
            log_viva_question, log_viva_answer_and_evaluation,
            log_study_plan,
        )
        from core.chat_transcript import format_chat_transcript
        from core.chat_state import drop_last_answer, truncate_for_edit
        def log_deep_study(*args, **kwargs): pass
        def log_exam_quiz(*args, **kwargs): pass

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Tessent & DFT Expert AI", page_icon="🧠", layout="wide")

# =====================================
# AUTH GATE (Google sign-in / password + 2FA / open)
# Layered: [auth] OIDC in secrets -> Google sign-in; APP_PASSWORD -> email +
# password (+ optional TOTP 2FA); nothing set -> open (local dev default).
# See AUTH_SETUP.md.
# =====================================
if not require_auth():
    st.stop()

# =====================================
# PRIVATE KNOWLEDGE BASE RESTORE (public-repo deploy)
# If this deployment has no local knowledge base (e.g. the public code
# repo on Streamlit Cloud), fetch the private bundle once from a GitHub
# release. See core/bootstrap.py; no-op when data is already present.
# 
# IMPORTANT: This MUST run AFTER auth because st.secrets is only populated
# during a user session, not at import/build time.
# =====================================
from core.bootstrap import restore_done, restore_knowledge

if not restore_done():
    restore_knowledge()

# =====================================
# SESSION STATE
# =====================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = []

if "evaluations" not in st.session_state:
    st.session_state.evaluations = []

# =====================================
# AGENT MAP
# =====================================
agent_map = {
    "SCAN": ask_scan,
    "ATPG": ask_atpg,
    "EDT": ask_edt,
    "LINUX": ask_linux,
    "TCL": ask_tcl,
    "MBIST": ask_mbist,
    "JTAG": ask_jtag,
    "IJTAG": ask_ijtag,
    "WRAPPER": ask_wrapper,
    "OCC": ask_occ,
    "STA": ask_sta,
    "BOUNDARYSCAN": ask_boundaryscan,
    "GLS": ask_gls,
}

# Helper to extract score from evaluation text
def extract_score(eval_text):
    match = re.search(r'(?:Score|score)\s*(?::|=)?\s*(\d+)', eval_text)
    if match:
        return int(match.group(1))
    return None

# =====================================
# SIDEBAR
# =====================================
with st.sidebar:
    auth_badge_sidebar()
    # ── Privacy disclosure (release decision 2026-11-06) ────────────────────
    # Session activity is recorded and uploaded to the owner's private repository,
    # so users must be told. Rendered after require_auth(), i.e. only for signed-in
    # users, and it states what is deliberately NOT stored.
    st.caption(
        "ℹ️ **Privacy:** your questions, answers and generated explanations are "
        "recorded and stored in the site owner's private repository for review. "
        "Uploaded presentation content, file names and exam-bank material are never logged."
    )
    st.title("🧠 Tessent Mentor AI")
    st.caption("Tessent & DFT Expert Assistant")
    
    # Navigation entries are filtered per identity. The Exam Bank is owner-only
    # (release decision 2026-11): non-owners never see the entry, and — because
    # hiding a widget is not a security boundary — the page branch itself
    # re-authorizes server-side before any Exam Bank data is read.
    _nav_pages = [
        "📤 Upload Manuals & Labs",
        "📖 Ask Question & Manual Citation",
        "🧪 Lab & Command Explainer",
        "🔍 Commands Explorer (Siemens DFT)",
        "❓ Assessment Question Generator",
        "🎤 Interactive Viva Practice",
        "📚 DFT Study Planner",
        "🔬 Deep Study (Videos & Labs)",
        "📝 Exam Question Bank (Real Questions)",
        "📦 Batch Export (All Reports)",
        "🧾 PPT Explainer",
    ]
    if not owner_verified():
        _nav_pages.remove("📝 Exam Question Bank (Real Questions)")

    mode = st.radio(
        "Navigation Mode",
        _nav_pages,
    )

    # ── WP-B: session usage & model routing (capability, calls, tokens, retries) ──
    # Opaque per-session key for the in-process budget counters. It is never derived
    # from user data, never logged, and never persisted.
    if "_vm_session_id" not in st.session_state:
        st.session_state["_vm_session_id"] = uuid.uuid4().hex[:12]
    render_usage_panel(st, st.session_state["_vm_session_id"])

    # Helper to enforce Strict Guest Mode (BYOK) across the app
    def ensure_llm_access() -> bool:
        if is_owner():
            if not bool(_get_gemini_keys()) and not bool(st.session_state.get("guest_api_key")):
                st.error("🔑 No Gemini API key configured. Add one in the sidebar, or set "
                         "GEMINI_API_KEY_1..6 (or GEMINI_API_KEY) in Settings -> Secrets.")
                return False
            return True
        else:
            guest_key = str(st.session_state.get("guest_api_key", "")).strip()
            if not guest_key:
                st.warning("🔑 **Guest Mode (BYOK)**: You must enter your own free Gemini API key in the sidebar to ask questions. "
                           "(Get your free key at [Google AI Studio](https://aistudio.google.com/app/apikey))")
                return False
            return True

    # Owner vs Guest Mode check
    backend_keys_ok = bool(_get_gemini_keys())

    if is_owner() and backend_keys_ok:
        st.caption("👑 **Owner Mode**: Backend key rotation active")
    elif is_owner():
        st.caption("👑 **Owner Mode** - no backend Gemini key is set. Enter one below or set "
                   "GEMINI_API_KEY_1..6 in Settings -> Secrets.")
        _key_input = st.text_input(
            "🔑 Gemini API Key",
            type="password",
            value=st.session_state.get("guest_api_key", ""),
            help="Get your free key at https://aistudio.google.com/app/apikey",
        )
        if _key_input:
            st.session_state["guest_api_key"] = _key_input.strip()
    else:
        st.info("👤 **Guest Mode (BYOK)**: Enter your Gemini API key below:")
        _key_input = st.text_input(
            "🔑 Gemini API Key (Required)",
            type="password",
            value=st.session_state.get("guest_api_key", ""),
            help="Get your free key at https://aistudio.google.com/app/apikey",
        )
        if _key_input:
            st.session_state["guest_api_key"] = _key_input.strip()
            st.success("✅ Personal API key active")
        else:
            st.session_state["guest_api_key"] = ""
            st.warning("⚠️ Enter your free Gemini API key to ask questions.")


    st.markdown("---")

    # Show missing secrets diagnostic panel
    try:
        from core.bootstrap import missing_secrets, restore_done
        missing = missing_secrets()
        if missing:
            st.markdown("### ⚠️ Missing Secrets")
            st.caption("Add these in Streamlit Cloud → Settings → Secrets:")
            for name, desc in missing:
                st.markdown(f"- **{name}**: {desc}")
            if not restore_done():
                st.caption("📚 Knowledge base not present — RAG will degrade gracefully")
    except Exception:
        pass

    st.markdown("---")

    # Always-visible logging status so the user knows every interaction is saved.
    from core.session_logger import LOG_DIR, _AUTO_SYNC_ENABLED, _get_github_token
    st.markdown("### 📝 Session Logging")
    st.caption("✅ Active — everything you ask & every answer is saved.")
    st.caption(f"Folder: `{LOG_DIR}`")
    # Which daily file is being written, and on whose clock. A container runs in UTC, so
    # without this the previous day's file looks like it is still being written.
    try:
        from core.session_logger import log_clock_info
        _clock = log_clock_info()
        st.caption(
            f"Writing to `{_clock['active_file']}` · clock {_clock['now']} "
            f"({_clock['timezone']})"
        )
    except Exception:
        pass
    
    # Show sync status
    token = _get_github_token()
    if _AUTO_SYNC_ENABLED and token:
        st.caption("🔄 **Auto-sync: ON** — logs sync to GitHub after each question")
        # Last attempt, so the owner can see the result in the app instead of hunting
        # through the Cloud log or GitHub.
        try:
            from core.session_logger import last_sync_status
            _sync = last_sync_status()
        except Exception:
            _sync = {}
        if _sync.get("status"):
            _status = _sync["status"]
            _entries = _sync.get("entries")
            _count = f" · {_entries} entries" if _entries is not None else ""
            _at = _sync.get("at") or ""
            if _status == "synced":
                st.caption(f"✅ **Last sync: {_at}**{_count} (merged with the remote copy)")
            elif _status == "up_to_date":
                st.caption(f"✅ **Last sync: {_at}**{_count} — remote already had every entry")
            elif _status == "blocked":
                st.caption(f"⛔ **Last sync: {_at}** — blocked by the content-sensitivity guard")
                st.caption(f"   _{_sync.get('detail', '')}_")
            elif _status == "error":
                st.caption(f"⚠️ **Last sync: {_at}** failed — {_sync.get('detail', '')}")
            else:
                st.caption(f"⏸️ **Auto-sync idle** — {_sync.get('detail', '')}")
        else:
            st.caption("_No sync attempt yet in this process — ask a question to start._")
    elif _AUTO_SYNC_ENABLED:
        st.caption("⚠️ **Auto-sync: OFF** — set GITHUB_TOKEN secret to enable")
    
    try:
        log_files = sorted(LOG_DIR.glob("*.log"))
        if log_files:
            st.caption(f"Files: `{', '.join(f.name for f in log_files[-3:])}`")
            # Download button for the latest log file
            latest_log = log_files[-1]
            with open(latest_log, "r", encoding="utf-8") as f:
                log_content = f.read()
            st.download_button(
                label=f"⬇️ Download {latest_log.name}",
                data=log_content,
                file_name=latest_log.name,
                mime="text/plain",
                help="Download the latest session log to your local machine",
            )
    except Exception:
        pass

    st.markdown("---")

    if mode == "📖 Ask Question & Manual Citation":
        st.markdown("### Core DFT Domains")
        st.markdown("- 🟢 **SCAN** (Tessent Scan)")
        st.markdown("- 🔵 **ATPG** (Tessent FastScan)")
        st.markdown("- 🟣 **EDT** (Tessent TestKompress)")
        st.markdown("- 🐧 **LINUX / TSHELL** (Environment & TCL Flow)")
        
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("show_export", None)
            st.rerun()

        if st.button("💾 Export Chat", use_container_width=True):
            st.session_state["show_export"] = True

        if st.session_state.get("show_export") and st.session_state.messages:
            transcript_md = format_chat_transcript(st.session_state.messages)
            st.download_button(
                label="⬇️ Download Conversation (.md)",
                data=transcript_md,
                file_name="tessent_mentor_conversation.md",
                mime="text/markdown",
                use_container_width=True,
            )
            
    elif mode == "🎤 Interactive Viva Practice":
        if st.button("🗑 Clear Viva Session", use_container_width=True):
            st.session_state.pop("interview_question", None)
            st.session_state.pop("evaluation", None)
            st.session_state["asked_questions"] = []
            st.session_state["evaluations"] = []
            st.rerun()
            
    elif mode == "📚 DFT Study Planner":
        if st.button("🗑 Reset Study History", use_container_width=True):
            st.session_state["asked_questions"] = []
            st.session_state["evaluations"] = []
            st.session_state.pop("cached_study_plan", None)
            st.success("Study history cleared!")
            st.rerun()

# =====================================
# 0. UPLOAD MANUALS & LABS MODE
# =====================================
if mode == "📤 Upload Manuals & Labs":
    st.title("📤 Upload Tessent Manuals & Lab PDFs")
    st.caption("Upload your official Tessent Reference Manuals or Lab PDFs here. They will be automatically extracted, cataloged, and indexed into the RAG vector store for instant retrieval.")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Select PDF File to Upload", type=["pdf"])
    with col2:
        topic_choice = st.selectbox(
            "Assign Target Manual Category",
            [
                "Shell User Manual",
                "Shell Reference Manual",
                "ATPG & Scan User Manual",
                "Library User Manual",
                "Scan & ATPG Lab Manual"
            ]
        )

    if uploaded_file is not None:
        st.info(f"📄 Ready to process: `{uploaded_file.name}` ({uploaded_file.size / (1024*1024):.2f} MB)")
        
        if st.button("🚀 Upload & Index Manual Now", use_container_width=True):
            with st.spinner("Saving PDF, extracting text pages, and updating vector index..."):
                res = upload_and_index_pdf(uploaded_file.getvalue(), uploaded_file.name, topic_choice)
                st.success(res)

    st.markdown("---")
    st.markdown("### 📁 Manual Directory Upload Guide")
    st.markdown("""
    You can also copy your PDF manual files directly into your project knowledge directory:
    - **Shell User Manual:** `knowledge/01_Tessent_Shell_User_Manual/`
    - **Shell Reference Manual:** `knowledge/02_Tessent_Shell_Reference_Manual/`
    - **ATPG & Scan User Manual:** `knowledge/03_Scan_and_ATPG_User_Manual/`
    - **Library User Manual:** `knowledge/04_Library_User_Manual/`
    - **Scan & ATPG Lab Manual:** `knowledge/05_Scan_and_ATPG_Lab_Manual/`

    After copying PDFs directly into those folders, run the re-indexing command:
    ```bash
    python core/rag_builder.py build
    ```
    """)


# =====================================
# 1. ASK QUESTION & MANUAL CITATION MODE
# =====================================
elif mode == "📖 Ask Question & Manual Citation":
    st.title("📖 Tessent Reference Manual Assistant")
    st.caption("Ask questions grounded strictly in official Tessent reference manuals with exact page & section citations.")

    # Display Old Chat (WP-9: per-message copy, edit and regenerate)
    _messages = st.session_state.messages
    _last_assistant_index = max(
        (i for i, m in enumerate(_messages) if m.get("role") == "assistant"), default=None
    )
    for _idx, msg in enumerate(_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("role") == "assistant":
                # One-action copy: st.code renders Streamlit's own copy button, so no
                # custom frontend or dependency is needed and the rendered Markdown is
                # unchanged.
                with st.expander("📋 Copy answer", expanded=False):
                    st.code(msg["content"], language=None)
                if _idx == _last_assistant_index:
                    if st.button("🔄 Regenerate this answer", key=f"regen_{_idx}"):
                        _remaining, _question_for_retry = drop_last_answer(_messages)
                        st.session_state.messages = _remaining
                        st.session_state["pending_question"] = _question_for_retry
                        st.rerun()
            elif msg.get("role") == "user":
                if st.button("✏️ Edit this question", key=f"edit_{_idx}"):
                    st.session_state["edit_index"] = _idx
                    st.session_state["edit_text"] = msg["content"]
                    st.rerun()

    if st.session_state.get("edit_index") is not None:
        _edit_idx = st.session_state["edit_index"]
        with st.form(key="edit_question_form", clear_on_submit=False):
            _edited = st.text_area(
                "Edit your question and resubmit (this replaces it and everything after it):",
                value=st.session_state.get("edit_text", ""),
                height=140,
                key="edit_question_area",
            )
            _col_save, _col_cancel = st.columns(2)
            _save = _col_save.form_submit_button("✅ Resubmit", use_container_width=True)
            _cancel = _col_cancel.form_submit_button("✖ Cancel", use_container_width=True)
        if _cancel:
            st.session_state.pop("edit_index", None)
            st.session_state.pop("edit_text", None)
            st.rerun()
        if _save and _edited.strip():
            st.session_state.messages = truncate_for_edit(st.session_state.messages, _edit_idx)
            st.session_state.pop("edit_index", None)
            st.session_state.pop("edit_text", None)
            st.session_state["pending_question"] = _edited.strip()
            st.rerun()

    # Optional attachment (document or image) to go with the question
    st.markdown("---")
    att_col1, att_col2 = st.columns([3, 1])
    with att_col1:
        uploaded_file = st.file_uploader(
            "📎 Attach a file with your question (PDF / DOCX / PPTX / TXT / image)",
            type=["pdf", "docx", "pptx", "ppt", "txt", "png", "jpg", "jpeg", "gif", "webp", "bmp"],
        )
    with att_col2:
        add_to_kb = st.checkbox(
            "Also add to knowledge base",
            value=False,
            help="Index the attached document permanently so future questions can retrieve it. Images are not indexed.",
        )
    if uploaded_file is not None:
        fnote = uploaded_file.name
        if is_image_ext(fnote):
            st.caption(f"🖼️ Attached image: `{fnote}` — will be analyzed with your question.")
        else:
            st.caption(f"📄 Attached document: `{fnote}` — will be read and used with your question.")
    st.markdown("---")

    # Chat Input (WP-9: a pending question from Edit / Regenerate takes precedence so
    # the retried turn runs through exactly the same pipeline as a typed question)
    question = st.chat_input("Ask a Tessent Scan, ATPG, EDT, or TShell question...")
    _pending_question = st.session_state.pop("pending_question", None)
    if _pending_question:
        question = _pending_question

    if question:
        if not ensure_llm_access():
            st.stop()

        history = st.session_state.messages
        user_content = question
        if uploaded_file is not None:
            user_content = f"{question}\n\n[📎 Attached: {uploaded_file.name}]"
        st.session_state.messages.append({"role": "user", "content": user_content})
        with st.chat_message("user"):
            st.markdown(user_content)

        with st.chat_message("assistant"):
            with st.spinner("Searching Tessent Reference Manuals..."):
                try:
                    followup_words = ["it", "this", "that", "they", "those", "why"]
                    q = question.lower()

                    if len(question.split()) <= 5 and any(word in q for word in followup_words):
                        topic = st.session_state.get("last_topic", "GENERAL")
                    else:
                        topic = route_question(question)

                    st.session_state["last_topic"] = topic

                    topic_colors = {
                        "SCAN": "🟢",
                        "ATPG": "🔵",
                        "EDT": "🟣",
                        "LINUX": "🐧",
                        "TCL": "⚙️",
                        "MBIST": "🟠",
                        "JTAG": "🟡",
                        "STA": "🟩"
                    }

                    st.success(f"{topic_colors.get(topic, '⚫')} Expert Router Selected: {topic}")

                    # ----- Handle optional attachment, else plain text -----
                    answer = None
                    # Attachment log metadata (filename, stored path, preview)
                    log_att_name = None
                    log_att_path = None
                    log_att_preview = None

                    if uploaded_file is not None:
                        fname = uploaded_file.name
                        fbytes = uploaded_file.getvalue()
                        # Always persist a copy next to the session logs so we know,
                        # later, exactly what document/image a given answer used.
                        att_stored = save_attachment(fbytes, fname)
                        log_att_name = fname
                        log_att_path = att_stored if att_stored else None

                        if is_image_ext(fname):
                            if add_to_kb:
                                st.info("ℹ️ Images are used for this question and are not added to the knowledge base.")
                            log_att_preview = f"[Image: {fname}, {len(fbytes)} bytes]"
                            answer = ask_with_image(question, fbytes, fname, history=history[:-1], _topic=topic)
                        elif is_text_ext(fname):
                            if add_to_kb:
                                if fname.lower().endswith(".pdf"):
                                    idx_msg = upload_and_index_pdf(fbytes, fname, topic)
                                else:
                                    idx_msg = index_text_file(fbytes, fname, topic)
                                with st.expander("🗂 Index result"):
                                    st.write(idx_msg)
                            att_text = document_text_to_string(fbytes, fname)
                            log_att_preview = att_text[:400] + ("..." if len(att_text) > 400 else "")
                            answer = ask_with_text_attachment(question, att_text, history=history[:-1], _topic=topic)
                        else:
                            raise ValueError(
                                f"Unsupported file type for '{fname}'. Supported: PDF, DOCX, PPTX, TXT, and image files."
                            )
                        st.markdown(answer)
                    else:
                        agent = agent_map.get(topic, ask_general)
                        try:
                            stream_gen = agent(question, history[:-1], stream=True)
                            answer = st.write_stream(stream_gen)
                        except (TypeError, Exception):
                            answer = agent(question, history[:-1])
                            st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    # WP-1 fallback: when the answer abstained and no concept matched, offer
                    # clearly-labelled suggestions so the user has a way forward. The abstention
                    # text above stays authoritative; nothing here is presented as an answer.
                    try:
                        _abstained = any(
                            marker in (answer or "").lower()
                            for marker in ("do not have the exact answer", "do not have this information",
                                           "does not contain enough detail", "not in my verified source")
                        )
                        if _abstained:
                            from core.concept_thesaurus import suggest_commands as _suggest_concepts
                            from core.retrieval_fallback import (
                                render_suggestions_markdown as _render_suggestions,
                                suggest_commands_for_question as _suggest_commands,
                            )

                            _concept_cmds, _ = _suggest_concepts(question)
                            _suggestions = _suggest_commands(question, concept_commands=_concept_cmds)
                            _block = _render_suggestions(_suggestions)
                            if _block:
                                with st.expander("🔎 Suggestions to refine your question", expanded=False):
                                    st.markdown(_block)
                    except Exception:
                        pass
                    if log_att_name:
                        st.caption(
                            f"📎 Answer used attachment `{log_att_name}` — saved copy: `{log_att_path}`"
                            if log_att_path else f"📎 Answer used attachment `{log_att_name}`"
                        )
                    log_qa(
                        question,
                        answer,
                        attachment_name=log_att_name,
                        attachment_path=log_att_path,
                        attachment_preview=log_att_preview,
                    )

                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    log_qa_error(question, error_msg)

# =====================================
# 2. LAB & COMMAND EXPLAINER MODE
# =====================================
elif mode == "🧪 Lab & Command Explainer":
    st.title("🧪 Tessent Lab & Command Explainer")
    st.caption("Analyze Tessent lab scripts step-by-step. Get internal DB mechanics, flow placement, output files, missing command impact, and key interview/viva questions.")

    col1, col2 = st.columns([1, 3])
    with col1:
        topic = st.selectbox("Select Module", ["SCAN", "ATPG", "EDT", "LINUX / TSHELL"])
    with col2:
        preset_cmd = st.selectbox(
            "Quick Command Presets (or type custom TCL below)",
            [
                "Custom Input",
                "read_verilog / set_current_design / add_clocks / check_design_rules",
                "set_system_mode analysis / add_scan_chains / check_design_rules",
                "create_patterns -type transition / run_atpg / write_patterns",
                "set_edt_options -channels 2 / create_dft_structures / write_dft_setup"
            ]
        )

    default_script = ""
    if preset_cmd != "Custom Input":
        default_script = preset_cmd

    script_input = st.text_area("Paste Lab TCL Commands or Script Sequence:", value=default_script, height=180, placeholder="e.g. read_verilog top.v\nset_current_design top\nadd_clocks 0 clk\ncheck_design_rules")

    if st.button("🚀 Explain Lab Command Sequence", use_container_width=True):
        if not ensure_llm_access():
            st.stop()
        if script_input:
            with st.spinner("Analyzing command flow against Tessent Reference Manuals..."):
                try:
                    explanation = explain_lab_command(script_input, topic)
                    st.markdown(explanation)
                    log_lab_explainer(topic, script_input, explanation)
                except Exception as e:
                    err = f"Error: {e}"
                    st.error(err)
                    log_lab_explainer_error(topic, script_input, err)
        else:
            st.error("Please enter a command or script sequence.")

    # WP-2: the cross-reference used to live only in Deep Study → Lab tab, so users working
    # in this lab-oriented mode saw no cross-reference at all (and concluded it "only works
    # for ATPG Core"). It is now reachable here, over every course group.
    with st.expander("🔗 Lab ↔ Manual Cross-Reference (all modules)", expanded=False):
        st.caption(
            "Deterministic command extraction per lab exercise — every Tessent command found in "
            "the lab material is matched to its manual definition with page citations. Same "
            "engine as Deep Study; no LLM, no hallucination."
        )
        _xr_all_labs = list_lab_exercises()
        _xr_pick = st.selectbox("Cross-reference which lab?", _xr_all_labs, key="labxr_pick")
        _xr_mod = module_for_lab_display(_xr_pick)
        st.caption(f"Resolved course group: **{_xr_mod}**")
        _xr_key = "labxr_rows_" + _xr_mod + "_" + _xr_pick
        if st.button("🔍 Build Cross-Reference", key="labxr_btn", use_container_width=True):
            with st.spinner("Extracting commands and matching manual definitions..."):
                try:
                    st.session_state[_xr_key] = crossref_lab(_xr_pick, module=_xr_mod)
                except Exception as exc:
                    st.session_state[_xr_key] = "ERROR: " + str(exc)

        _xr_rows = st.session_state.get(_xr_key)
        if isinstance(_xr_rows, str):
            st.error(_xr_rows)
        elif _xr_rows is not None:
            if _xr_rows:
                st.markdown(format_crossref_table(_xr_rows, lab_display=_xr_pick))
            else:
                st.info(
                    "No exercise-level command evidence was found for this lab. "
                    "This lab is one of the known gaps (the workbook section is absent, the "
                    "scope text contains no command, or no scope text exists at all) — no "
                    "mapping is invented to fill it."
                )

# =====================================
# 2B. COMMANDS EXPLORER MODE (STAGE 2.3)
# =====================================
elif mode == "🔍 Commands Explorer (Siemens DFT)":
    st.title("🔍 Commands Explorer (Siemens DFT)")
    st.caption("Deterministic, corpus-grounded index of verified Siemens Tessent Shell commands, syntax, options, and workbook references.")

    filter_opts = get_available_filter_options()

    # WP-3: genuine type-ahead. Streamlit 1.58.0 filters the option list live as the
    # user types (filter_mode="prefix"), and accept_new_options lets the same widget
    # submit free text, so wildcard/substring/alias queries still go through the
    # deterministic search_commands() engine below.
    _index_for_options = load_command_index()
    _command_names = set(_index_for_options.get("commands", {}))
    _alias_names = set(_index_for_options.get("alias_map", {}))
    _query_options = sorted(_command_names | _alias_names)

    col_q, col_m, col_mode, col_flow = st.columns([3, 1, 1, 2])
    with col_q:
        picked_query = st.selectbox(
            "Search Command / Alias / Keyword",
            options=_query_options,
            index=None,
            filter_mode="prefix",
            accept_new_options=True,
            placeholder="e.g. write (type-ahead) · write* · scan · read_cell_lib",
            help=(
                "Type-ahead prefix filtering over every indexed command. Press Enter on "
                "free text to run the deterministic search: 'write*' for prefix-only "
                "matching, 'scan' for a substring, or an alias such as read_cell_lib."
            ),
            key="cmd_explorer_query",
        )
    search_query = (picked_query or "").strip()
    with col_m:
        sel_module = st.selectbox("Module", ["ALL"] + filter_opts["modules"])
    with col_mode:
        sel_mode = st.selectbox("Mode", ["ALL"] + filter_opts["modes"])
    with col_flow:
        sel_flow = st.selectbox("Flow Step", ["ALL"] + filter_opts["flow_steps"])

    inc_unverified = st.checkbox("Include unverified allowlist candidates (6 legacy)", value=False)

    results = search_commands(
        query=search_query,
        module=sel_module,
        mode=sel_mode,
        flow_step=sel_flow,
        include_unverified=inc_unverified,
    )

    st.caption(f"Showing **{len(results)}** command(s)")

    # WP-3: explain every match in one pass. Deterministic and source-grounded (index fields
    # only), so a wildcard like write* can be answered without inventing definitions.
    if len(results) > 1:
        _bulk_col1, _bulk_col2 = st.columns([2, 1])
        with _bulk_col1:
            _bulk_clicked = st.button(
                f"📖 Explain all {len(results)} matching commands",
                key="cmd_bulk_explain_btn",
                help="Renders purpose, syntax, arguments, examples and the manual citation for "
                     "every match, straight from the verified command index.",
            )
        with _bulk_col2:
            _bulk_limit = st.number_input("Max commands", min_value=1, max_value=60, value=min(32, len(results)),
                                          key="cmd_bulk_limit")
        if _bulk_clicked:
            try:
                from core.command_bulk_explain import bulk_explain, render_bulk_markdown

                _bulk = bulk_explain(search_query or "", limit=int(_bulk_limit))
                _md = render_bulk_markdown(_bulk)
                st.session_state["cmd_bulk_markdown"] = _md
                st.session_state["cmd_bulk_summary"] = (
                    _bulk["matched"], _bulk["explained"], len(_bulk["undocumented"]))
            except Exception as _bulk_error:
                st.error(f"Bulk explanation failed: {type(_bulk_error).__name__}")
        if st.session_state.get("cmd_bulk_markdown"):
            _matched, _explained, _undocumented = st.session_state.get(
                "cmd_bulk_summary", (0, 0, 0))
            st.caption(f"Bulk explanation: **{_explained}** of **{_matched}** rendered · "
                       f"{_undocumented} with incomplete documentation (labelled in the output).")
            with st.expander("📄 Bulk explanation", expanded=True):
                st.markdown(st.session_state["cmd_bulk_markdown"])
                st.download_button(
                    "⬇️ Download bulk explanation (.md)",
                    data=st.session_state["cmd_bulk_markdown"],
                    file_name=f"bulk_explain_{(search_query or 'all').replace('*', 'star')}.md",
                    mime="text/markdown",
                    key="cmd_bulk_download",
                )

    if not results:
        st.info("ℹ️ No commands match your search query and filters.")
    else:
        col_list, col_detail = st.columns([1, 2])
        with col_list:
            cmd_labels = []
            for r in results:
                status_icon = "✅" if r.get("status") == "VERIFIED_CANONICAL" else "⚠️"
                alias_tag = f" [alias: {r['matched_alias']}]" if r.get("matched_alias") else ""
                cmd_labels.append(f"{status_icon} {r.get('canonical_name', r.get('command'))}{alias_tag}")

            selected_idx = st.selectbox(
                "Select Command to Inspect",
                range(len(results)),
                format_func=lambda i: cmd_labels[i],
                key="cmd_explorer_select",
            )
            selected = results[selected_idx]

        with col_detail:
            cname = selected.get("canonical_name", selected.get("command"))
            st.subheader(f"`{cname}`")

            if selected.get("status") == "UNVERIFIED_IN_CORPUS":
                st.warning("⚠️ **UNVERIFIED IN CORPUS** — Legacy allowlist candidate with zero definitions in Siemens manuals and zero occurrences in audited workbooks.")
                if selected.get("note"):
                    st.markdown(f"**Note:** {selected['note']}")
            else:
                col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                with col_b1:
                    st.metric("Module", selected.get("module") or "N/A")
                with col_b2:
                    st.metric("Mode", selected.get("mode") or "N/A")
                with col_b3:
                    st.metric("Context", selected.get("context") or "N/A")
                with col_b4:
                    st.metric("Difficulty", selected.get("difficulty", "N/A").capitalize() if selected.get("difficulty") else "N/A")

                if selected.get("flow_step"):
                    st.markdown(f"🎯 **Flow Step (curated):** `{selected['flow_step']}`")
                    st.caption(
                        "Curated taxonomy label — a human classification of where this command "
                        "sits in the flow. The cited page supports the command; it does not "
                        "necessarily state this flow position verbatim "
                        "(see FLOW_STEP_AUDIT_REPORT.md)."
                    )
                else:
                    st.caption(
                        "🎯 **Flow Step:** not recorded in the available sources for this command."
                    )

                if selected.get("module_inferred"):
                    st.caption(
                        "🧭 Module inferred from the manual's own `Context:` line: "
                        f"`{selected.get('module_evidence') or 'no context recorded'}`"
                    )

                if selected.get("matched_alias"):
                    rel = selected.get("alias_metadata", {}).get("relationship", "alias")
                    st.info(f"🔗 Match resolved from alias **`{selected['matched_alias']}`** (Relationship: `{rel}`)")

                if selected.get("aliases"):
                    alias_badges = ", ".join(f"`{a}`" for a in selected["aliases"])
                    st.caption(f"Known Aliases: {alias_badges}")

                st.markdown(f"**Purpose:** {selected.get('purpose') or 'Not stated in the available manual entry.'}")

                arguments = selected.get("arguments") or []
                if arguments:
                    shown = ", ".join(f"`{arg}`" for arg in arguments[:40])
                    if len(arguments) > 40:
                        shown += f" … (+{len(arguments) - 40} more)"
                    st.markdown(f"**Arguments / Options:** {shown}")
                    if selected.get("arguments_note"):
                        st.caption(selected["arguments_note"])
                else:
                    st.markdown(
                        "**Arguments / Options:** "
                        + (
                            "the manual entry does not list arguments or options."
                            if selected.get("arguments_unavailable")
                            else "none captured for this command."
                        )
                    )

                st.markdown("**Authoritative Syntax:**")
                st.code(selected.get("syntax") or f"{cname}  # syntax not recorded in the available sources", language="tcl")

                if selected.get("examples"):
                    with st.expander("📝 Examples from the manual", expanded=False):
                        st.code(selected["examples"], language="tcl")
                else:
                    st.caption("📝 No example is recorded for this command in the available sources.")

                evidence = selected.get("source_evidence")
                if evidence:
                    st.markdown(
                        f"📖 **Authoritative Citation:** `{evidence.get('file')}`, **Page {evidence.get('page')}** ({evidence.get('section', 'Command Reference')})"
                    )

                wbs = selected.get("workbook_references", [])
                if wbs:
                    with st.expander(f"📚 Audited Workbook Occurrences ({len(wbs)})", expanded=False):
                        for ref in wbs:
                            st.markdown(f"- **{ref.get('module')}**: `{ref.get('item_id')}` ({ref.get('source_reference')})")
elif mode == "❓ Assessment Question Generator":
    st.title("❓ DFT Practice & Viva Question Generator")
    st.caption("Generate a targeted question bank for any core DFT/Tessent module, with benchmark model answers and manual citations.")

    col1, col2 = st.columns(2)
    with col1:
        target_module = st.selectbox("Select DFT Module", ["SCAN", "ATPG", "EDT", "LINUX / TSHELL"])
    with col2:
        num_q = st.slider("Number of Questions", min_value=5, max_value=20, value=15)

    if st.button("🎲 Generate Question Bank", use_container_width=True):
        if not ensure_llm_access():
            st.stop()
        with st.spinner(f"Building {num_q}-question bank for {target_module}..."):
            q_bank = generate_assessment_question_bank(target_module, num_questions=num_q)
            st.markdown(q_bank)
            log_question_bank(target_module, num_q, q_bank)

# =====================================
# 4. INTERACTIVE VIVA PRACTICE MODE
# =====================================
elif mode == "🎤 Interactive Viva Practice":
    st.title("🎤 Tessent Interactive Viva & Interview Practice")
    st.caption("Practice DFT interview and viva questions. Your answers are graded against Tessent reference manual criteria with scores and detailed feedback.")

    col1, col2 = st.columns(2)
    with col1:
        topic = st.selectbox(
            "Select Viva Topic",
            ["SCAN", "ATPG", "EDT", "LINUX", "TCL", "MBIST", "JTAG", "STA"]
        )
    with col2:
        difficulty = st.selectbox("Difficulty Level", ["Beginner (Basic Commands)", "Intermediate (Lab DRCs)", "Advanced (POC Tricky Scenarios)"])

    if st.button("Generate Viva Question", use_container_width=True):
        if not ensure_llm_access():
            st.stop()
        with st.spinner("Generating mock viva question..."):
            question = ask_interview_question(topic, difficulty, st.session_state["asked_questions"])
            st.session_state["interview_question"] = question
            st.session_state["asked_questions"].append(question)
            st.session_state.pop("evaluation", None)
            log_viva_question(topic, difficulty, question)

    if "interview_question" in st.session_state:
        st.markdown("### 🎙️ Interview Question")
        st.info(st.session_state["interview_question"])

        candidate_answer = st.text_area("Type your explanation / answer below:", height=180, placeholder="Explain what the command does, why it's placed here, what file is generated, or what happens if skipped...")

        if st.button("Submit Answer for Evaluation", use_container_width=True):
            if not ensure_llm_access():
                st.stop()
            if candidate_answer.strip():
                with st.spinner("Grading response against Tessent Reference Manual criteria..."):
                    result = evaluate_viva_answer(st.session_state["interview_question"], candidate_answer)
                    st.session_state["evaluation"] = result
                    st.session_state.evaluations.append(result)
                    log_viva_answer_and_evaluation(
                        st.session_state["interview_question"], candidate_answer, result
                    )
            else:
                st.warning("Please type an answer before submitting.")

    if "evaluation" in st.session_state:
        st.markdown("### 📋 Evaluation Report")
        st.markdown(st.session_state["evaluation"])

# =====================================
# 5. DFT STUDY PLANNER MODE
# =====================================
elif mode == "📚 DFT Study Planner":
    st.title("📚 Personalized DFT Study Planner")
    st.caption("Track your practice metrics, visualize score trends, and get a customized DFT study and revision schedule.")

    if st.session_state.evaluations:
        scores = []
        for e in st.session_state.evaluations:
            score = extract_score(e)
            if score is not None:
                scores.append(score)

        if scores:
            col1, col2, col3 = st.columns(3)
            avg_score = sum(scores) / len(scores)
            
            with col1:
                st.metric("Average Viva Score", f"{avg_score:.2f} / 10")
            with col2:
                st.metric("Viva Questions Attempted", len(scores))
            with col3:
                readiness = "Needs Work (< 6)"
                if avg_score >= 8:
                    readiness = "Interview Ready (8–10)"
                elif avg_score >= 6:
                    readiness = "Good Level (6–7.9)"
                st.metric("Interview Readiness", readiness)

            st.markdown("### 📈 Viva Score History")
            st.line_chart(scores)
        else:
            st.info("Start practicing viva questions in **Interactive Viva Practice** mode to see your score trends here!")

        st.markdown("---")
        st.markdown("### 📋 Your Personalized Study Guide")

        if "cached_study_plan" in st.session_state:
            st.markdown(st.session_state["cached_study_plan"])
            if st.button("🔄 Regenerate Study Plan", use_container_width=True):
                if not ensure_llm_access():
                    st.stop()
                with st.spinner("Regenerating 7-day revision plan..."):
                    study_plan = generate_study_plan(st.session_state.asked_questions, st.session_state.evaluations)
                    st.session_state["cached_study_plan"] = study_plan
                    log_study_plan(study_plan)
                    st.rerun()
        else:
            st.info("Click the button below to generate your personalized 7-day study plan based on your viva practice scores.")
            if st.button("📝 Generate 7-Day Study Plan", use_container_width=True):
                if not ensure_llm_access():
                    st.stop()
                with st.spinner("Generating 7-day revision plan..."):
                    study_plan = generate_study_plan(st.session_state.asked_questions, st.session_state.evaluations)
                    st.session_state["cached_study_plan"] = study_plan
                    log_study_plan(study_plan)
                    st.rerun()
    else:
        st.info("No practice interview history found. Practice some viva questions in **Interactive Viva Practice** mode to unlock your customized study plan!")

elif mode == "🔬 Deep Study (Videos & Labs)":
    st.header("🔬 Deep Study — Video Lessons & Lab Exercises")
    st.caption(
        "Pick a training video or a lab exercise and get a detailed, "
        "manual-cited study report, ending with rapid viva questions."
    )

    video_tab, lab_tab = st.tabs(["🎬 Video Lessons", "🧪 Lab Exercises"])

    with video_tab:
        video_modules = list_video_modules()
        if not video_modules:
            st.warning("No video transcripts found under knowledge/txt.")
        else:
            nums = sorted(video_modules.keys())
            labels = [video_modules[n] for n in nums]
            sel_video = st.selectbox("Choose a video module:", labels, key="ds_video_sel")
            sel_num = nums[labels.index(sel_video)]

            lesson_titles = list_video_lessons(sel_num)
            lesson_options = ["Whole module (all lessons)"] + lesson_titles

            # Learning-path strip: ordered course sequence + position.
            # The Next button uses an on_click callback (set before the
            # selectbox is instantiated below) — never mutates ds_lesson_sel
            # after widget creation in the same run.
            def _ds_advance_lesson():
                target = st.session_state.get("ds_path_next_target")
                if target is not None:
                    st.session_state["ds_lesson_sel"] = target

            path_pos = learning_path_position(sel_num, lessons=lesson_titles)
            if path_pos is None:
                st.caption(
                    "Learning path unavailable for this module — "
                    "no lesson data was found."
                )
            else:
                crumbs = []
                for i, cname in enumerate(path_pos["path"], 1):
                    if cname == path_pos["course_name"]:
                        crumbs.append("**[%s]**" % cname)
                    else:
                        crumbs.append("%d. %s" % (i, cname))
                st.caption("🗺️ Learning path: " + " → ".join(crumbs))
                cur_title = st.session_state.get("ds_lesson_sel")
                if cur_title and cur_title != "Whole module (all lessons)":
                    cur_pos = learning_path_position(
                        sel_num, cur_title, lessons=lesson_titles
                    )
                    if cur_pos is not None and cur_pos["lesson_index"] is not None:
                        st.caption(
                            "Lesson %d of %d in %s (Course %d of %d)"
                            % (
                                cur_pos["lesson_index"],
                                cur_pos["lesson_total"],
                                cur_pos["course_name"],
                                cur_pos["course_index"],
                                cur_pos["course_total"],
                            )
                        )
                        if cur_pos["next_lesson"] is not None:
                            st.session_state["ds_path_next_target"] = cur_pos[
                                "next_lesson"
                            ]
                            st.button(
                                "Next lesson: %s" % cur_pos["next_lesson"],
                                key="ds_path_next_btn",
                                on_click=_ds_advance_lesson,
                                use_container_width=True,
                            )
                        else:
                            st.caption(
                                "You are on the last lesson of this module — "
                                "end of the available path."
                            )

            sel_lesson = st.selectbox(
                "Choose an individual video lesson:",
                lesson_options,
                key="ds_lesson_sel",
            )

            if sel_lesson == "Whole module (all lessons)":
                if st.button("🎥 Generate Deep Study Report", key="ds_video_btn", use_container_width=True):
                    if not ensure_llm_access():
                        st.stop()
                    with st.spinner("Reading transcript, slides and manuals... (this can take a minute)"):
                        v_rep = explain_video_lesson(sel_num)
                        st.session_state["ds_video_report_" + str(sel_num)] = v_rep
                        log_deep_study("video", str(sel_num), "Whole Module", v_rep)
                cached_video = st.session_state.get("ds_video_report_" + str(sel_num))
                if cached_video:
                    # Layman's explanation expander
                    if "## Layman's Explanation" in cached_video:
                        with st.expander("🧑‍🏫 Layman's Explanation (Beginner Friendly)", expanded=True):
                            layman_start = cached_video.index("## Layman's Explanation")
                            layman_end = cached_video.find("## ", layman_start + 1)
                            if layman_end == -1:
                                layman_end = len(cached_video)
                            st.markdown(cached_video[layman_start:layman_end])
                    st.markdown(cached_video)
                    st.download_button(
                        "⬇️ Download report (.md)",
                        data=cached_video,
                        file_name="deep_study_video_module_" + str(sel_num) + ".md",
                        mime="text/markdown",
                        key="ds_video_dl_" + str(sel_num),
                    )
            else:
                slug = lesson_slug(sel_lesson)
                cache_key = "ds_lesson_report_" + str(sel_num) + "_" + slug
                ppt_key = "ds_lesson_ppt_" + str(sel_num) + "_" + slug
                dia_key = "ds_lesson_dia_" + str(sel_num) + "_" + slug

                # Diagrams toggle (vision analysis is slow — opt-in)
                include_diagrams = st.checkbox(
                    "🔍 Include diagram/screenshot AI analysis (slower, richer report)",
                    value=True,
                    key="ds_lesson_diagrams_toggle",
                )

                if st.button("🎥 Generate Lesson Report", key="ds_lesson_btn", use_container_width=True):
                    if not ensure_llm_access():
                        st.stop()
                    with st.spinner("Reading this lesson's transcript, slides and manuals... (up to a minute)"):
                        report, ppt_path, dia = explain_video_lesson_detail_v2(
                            sel_num, sel_lesson, generate_ppt=False, include_diagrams=include_diagrams
                        )
                        st.session_state[cache_key] = report
                        st.session_state[ppt_key] = ppt_path
                        st.session_state[dia_key] = dia
                        log_deep_study("video", str(sel_num), sel_lesson, report)
                cached_lesson = st.session_state.get(cache_key)
                if cached_lesson:
                    # Layman's explanation expander
                    if "## Layman's Explanation" in cached_lesson:
                        with st.expander("🧑‍🏫 Layman's Explanation (Beginner Friendly)", expanded=True):
                            layman_start = cached_lesson.index("## Layman's Explanation")
                            layman_end = cached_lesson.find("## ", layman_start + 1)
                            if layman_end == -1:
                                layman_end = len(cached_lesson)
                            st.markdown(cached_lesson[layman_start:layman_end])
                    # Full report
                    st.markdown(cached_lesson)
                    # PPT generation button
                    if st.button("📊 Generate PPT for SPOC Review", key="ds_ppt_btn_" + slug, use_container_width=True):
                        with st.spinner("Generating PowerPoint deck from cached report..."):
                            ppt_path = st.session_state.get(ppt_key)
                            if not ppt_path:
                                _, ppt_path = generate_lesson_ppt_from_report(
                                    sel_num, sel_lesson, cached_lesson,
                                    diagram_descriptions=st.session_state.get(dia_key, ""),
                                )
                                st.session_state[ppt_key] = ppt_path
                            if ppt_path and ppt_path.endswith(".pptx"):
                                st.success(f"✅ PPT generated: {ppt_path}")
                                with open(ppt_path, "rb") as f:
                                    st.download_button(
                                        "⬇️ Download PPT",
                                        data=f.read(),
                                        file_name=Path(ppt_path).name,
                                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                        key="ds_ppt_dl_" + slug,
                                    )
                            else:
                                st.warning(f"PPT generation issue: {ppt_path}")
                    st.download_button(
                        "⬇️ Download report (.md)",
                        data=cached_lesson,
                        file_name="deep_study_lesson_" + slug + ".md",
                        mime="text/markdown",
                        key="ds_lesson_dl_" + slug,
                    )
                    # Rapid Viva self-check: interactive per-lesson assessment
                    # parsed deterministically from the report's own section.
                    viva_items = extract_rapid_viva_items(cached_lesson)
                    if viva_items:
                        with st.expander(
                            "📝 Rapid Viva Self-Check (%d questions)" % len(viva_items),
                            expanded=False,
                        ):
                            st.caption(
                                "Type your answer per question and check it against "
                                "the report's model answer. Nothing is sent anywhere; "
                                "checking is local keyword overlap."
                            )
                            n_show = min(len(viva_items), 10)
                            for idx in range(n_show):
                                item = viva_items[idx]
                                st.markdown("**Q%d. %s**" % (idx + 1, item["question"]))
                                attempt_key = (
                                    "ds_viva_attempt_" + str(sel_num) + "_" + slug
                                    + "_" + str(idx)
                                )
                                attempt = st.text_area(
                                    "Your answer:",
                                    key=attempt_key,
                                    height=68,
                                )
                                if st.button(
                                    "Check answer",
                                    key="ds_viva_check_" + str(sel_num) + "_" + slug
                                    + "_" + str(idx),
                                ):
                                    verdict, detail = check_viva_attempt(
                                        attempt, item["answer"]
                                    )
                                    if verdict == "full":
                                        st.success("✅ Full credit. " + detail)
                                    elif verdict == "partial":
                                        st.warning("🟡 Partial credit. " + detail)
                                    else:
                                        st.info("⬜ No credit yet. " + detail)
                                with st.expander("Model answer", expanded=False):
                                    st.markdown(item["answer"])
                    else:
                        st.caption(
                            "Rapid Viva Self-Check unavailable for this report — "
                            "no answerable rapid-fire section was found."
                        )

    with lab_tab:
        lab_list = list_lab_exercises()
        if not lab_list:
            st.warning("No lab exercises found in the knowledge base.")
        else:
            # WP-5: name the coverage gap per course instead of leaving the list looking
            # incomplete. Counts come from the same list this tab offers.
            try:
                from collections import Counter as _Counter

                _lab_counts = _Counter()
                for _lab in lab_list:
                    _lab_counts[module_for_lab_display(_lab)] += 1
                _counts_text = ", ".join(f"{course}: {count}"
                                         for course, count in sorted(_lab_counts.items()))
                st.caption(f"Lab exercises available — {_counts_text} ({len(lab_list)} total).")
                if "TSHELL" in str(_lab_counts.keys()):
                    _shell_labs = _lab_counts.get("TSHELL_LINUX", 0)
                    if _shell_labs:
                        st.caption(
                            f"Tessent Shell has {_shell_labs} lab exercises; the separate "
                            "\"Tessent Shell\" theory group in Batch Export has no lab source in "
                            "the corpus, so no Shell lab is invented to fill it."
                        )
            except Exception:
                pass
            sel_lab = st.selectbox("Choose a lab exercise:", lab_list, key="ds_lab_sel")
            # Related labs (suggested prerequisites): deterministic ranking by
            # shared verified commands (crossref_agent). Lazy — only computed on
            # click so no corpus scan happens during normal browsing.
            rel_key = "ds_lab_related_" + sel_lab.replace(" / ", "_")
            if st.button(
                "Show related labs (suggested prerequisites)",
                key="ds_lab_related_btn",
                use_container_width=True,
            ):
                with st.spinner("Finding labs that share this exercise's commands..."):
                    try:
                        st.session_state[rel_key] = related_labs_for_display(sel_lab)
                    except Exception as exc:
                        st.session_state[rel_key] = "ERROR: " + str(exc)

            rel_cached = st.session_state.get(rel_key)
            if isinstance(rel_cached, str):
                st.error(rel_cached)
            elif rel_cached:
                st.markdown("**Related labs (suggested prerequisites)**")
                st.caption(
                    "Ranked by number of shared Tessent commands found in the "
                    "verified lab corpus. A related lab is not a mandate — it is "
                    "traceable evidence of what to review first."
                )
                for entry in rel_cached:
                    st.markdown(
                        "- **%s** — shares %d command(s): %s"
                        % (
                            entry["lab"],
                            entry["count"],
                            ", ".join("`%s`" % c for c in entry["shared"]),
                        )
                    )
            elif rel_cached is not None:
                st.info(
                    "No other lab in the verified corpus shares commands with "
                    "this exercise — nothing supportable to suggest."
                )
            
            # Show all files at once
            lab_name = sel_lab.split(" / ")[0]
            lab_files = list_lab_files_at_once(lab_name)
            
            if lab_files["exercises"]:
                with st.expander(f"📁 All Files in {lab_name} ({lab_files['total_screenshots']} screenshots)", expanded=True):
                    for ex_name, ex_data in sorted(lab_files["exercises"].items()):
                        st.markdown(f"**{ex_name}/** ({len(ex_data['screenshots'])} screenshots)")
                        if ex_data["tcl_files"]:
                            st.markdown(f"  - TCL: {', '.join(f.name for f in ex_data['tcl_files'])}")
                        if ex_data["text_files"]:
                            st.markdown(f"  - Text: {', '.join(f.name for f in ex_data['text_files'])}")
                        # Show screenshot thumbnails
                        if ex_data["screenshots"]:
                            cols = st.columns(min(4, len(ex_data["screenshots"])))
                            for idx, img in enumerate(ex_data["screenshots"][:8]):  # max 8 previews
                                with cols[idx % 4]:
                                    st.image(str(img), width=120, caption=img.name[:20])
                            if len(ex_data["screenshots"]) > 8:
                                st.caption(f"... and {len(ex_data['screenshots']) - 8} more")
                        st.markdown("---")
            
            lab_cache_key = "ds_lab_report_v2_" + sel_lab.replace(" / ", "_")
            lab_ppt_key = "ds_lab_ppt_" + sel_lab.replace(" / ", "_")
            lab_dia_key = "ds_lab_dia_" + sel_lab.replace(" / ", "_")
            
            # Diagrams toggle (vision analysis is slow — opt-in)
            include_diagrams = st.checkbox(
                "🔍 Include diagram/screenshot AI analysis (slower, richer report)",
                value=True,
                key="ds_lab_diagrams_toggle",
            )
            
            if st.button("🧪 Generate Deep Study Report", key="ds_lab_btn", use_container_width=True):
                if not ensure_llm_access():
                    st.stop()
                with st.spinner("Reading ALL lab files: slides, TCL, screenshots, manuals... (this can take 1-2 minutes)"):
                    report, ppt_path, dia = explain_lab_exercise_v2(
                        sel_lab, generate_ppt=False, include_diagrams=include_diagrams
                    )
                    st.session_state[lab_cache_key] = report
                    st.session_state[lab_ppt_key] = ppt_path
                    st.session_state[lab_dia_key] = dia
                    log_deep_study("lab", sel_lab, "Exercise", report)
            
            cached_lab = st.session_state.get(lab_cache_key)
            if cached_lab:
                # Layman's explanation expander
                if "## Layman's Explanation" in cached_lab:
                    with st.expander("🧑‍🏫 Layman's Explanation (Beginner Friendly)", expanded=True):
                        layman_start = cached_lab.index("## Layman's Explanation")
                        layman_end = cached_lab.index("## ", layman_start + 1)
                        st.markdown(cached_lab[layman_start:layman_end])
                
                # Full report with hyperlinks
                st.markdown(cached_lab)
                
                # PPT generation button
                if st.button("📊 Generate PPT for SPOC Review", key="ds_lab_ppt_btn", use_container_width=True):
                    with st.spinner("Generating PowerPoint deck from cached report..."):
                        ppt_path = st.session_state.get(lab_ppt_key)
                        if not ppt_path:
                            _, ppt_path = generate_lab_ppt_from_report(
                                sel_lab, cached_lab,
                                diagram_descriptions=st.session_state.get(lab_dia_key, ""),
                            )
                            st.session_state[lab_ppt_key] = ppt_path
                        if ppt_path and ppt_path.endswith(".pptx"):
                            st.success(f"✅ PPT generated: {ppt_path}")
                            with open(ppt_path, "rb") as f:
                                st.download_button(
                                    "⬇️ Download PPT",
                                    data=f.read(),
                                    file_name=Path(ppt_path).name,
                                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                    key="ds_lab_ppt_dl",
                                )
                        else:
                            st.warning(f"PPT generation issue: {ppt_path}")
                
                st.download_button(
                    "⬇️ Download report (.md)",
                    data=cached_lab,
                    file_name="deep_study_lab_" + sel_lab.replace(" / ", "_") + ".md",
                    mime="text/markdown",
                    key="ds_lab_dl",
                )
                # Rapid Viva self-check (same shared helpers as video
                # lessons): interactive lab assessment parsed
                # deterministically from the report's own Rapid Viva
                # section; a missing section abstains with a caption.
                lab_viva_items = extract_rapid_viva_items(cached_lab)
                lab_slug = sel_lab.replace(" / ", "_")
                if lab_viva_items:
                    with st.expander(
                        "Rapid Viva Self-Check (%d questions)" % len(lab_viva_items),
                        expanded=False,
                    ):
                        st.caption(
                            "Type your answer per question and check it against "
                            "the report's model answer. Nothing is sent anywhere; "
                            "checking is local keyword overlap."
                        )
                        n_lab_show = min(len(lab_viva_items), 10)
                        for lab_idx in range(n_lab_show):
                            lab_item = lab_viva_items[lab_idx]
                            st.markdown(
                                "**Q%d. %s**" % (lab_idx + 1, lab_item["question"])
                            )
                            lab_attempt = st.text_area(
                                "Your answer:",
                                key="ds_lab_viva_attempt_" + lab_slug + "_" + str(lab_idx),
                                height=68,
                            )
                            if st.button(
                                "Check answer",
                                key="ds_lab_viva_check_" + lab_slug + "_" + str(lab_idx),
                            ):
                                lab_verdict, lab_detail = check_viva_attempt(
                                    lab_attempt, lab_item["answer"]
                                )
                                if lab_verdict == "full":
                                    st.success("✅ Full credit. " + lab_detail)
                                elif lab_verdict == "partial":
                                    st.warning("🟡 Partial credit. " + lab_detail)
                                else:
                                    st.info("⬜ No credit yet. " + lab_detail)
                            with st.expander("Model answer", expanded=False):
                                st.markdown(lab_item["answer"])
                else:
                    st.caption(
                        "Rapid Viva Self-Check unavailable for this report — "
                        "no answerable rapid-fire section was found."
                    )

    with st.expander("🔗 Lab ↔ Manual Cross-Reference", expanded=False):
        st.caption(
            "Deterministic command extraction per lab: every Tessent command found in the lab's "
            "material is matched to its full manual definition (usage/options) with page citations, "
            "plus a reverse lookup of where else the command is used. No LLM, no hallucination."
        )
        st.caption(
            "⚠️ **Relationship types:** Only `DIRECTLY OBSERVED` relationships are currently "
            "emitted (both commands confirmed in the same lab text). `INFERRED` and `VALIDATED` "
            "types (PHASE_1_PLAN_APPROVED.md §9.1) require dofile/flow-step sequence files not "
            "present in the current corpus and are reserved for future corpus enrichment."
        )

        xr_lab = st.selectbox(
            "Cross-reference which lab?",
            list_lab_exercises(),
            key="xr_lab_sel",
        )

        # Derive the module key from the lab display string.  The mapping is
        # owned by agents.crossref_agent.module_for_lab_display(), which
        # follows the canonical taxonomy (knowledge/canonical_taxonomy.json)
        # so the UI and the agent cannot drift apart.  The "ATPG Core" course
        # spans two canonical domains, so it is resolved per lab there.
        xr_module = module_for_lab_display(xr_lab)

        # Session-state key includes both module and lab so cached rows from a
        # previous module/lab selection are never silently reused.
        _xr_cache_key = "xr_rows_" + xr_module + "_" + xr_lab

        if st.button("🔍 Build Cross-Reference", key="xr_btn", use_container_width=True):
            with st.spinner("Extracting commands and matching manual definitions..."):
                try:
                    rows = crossref_lab(xr_lab, module=xr_module)
                    st.session_state[_xr_cache_key] = rows
                except Exception as exc:
                    st.session_state[_xr_cache_key] = "ERROR: " + str(exc)

        xr_cached = st.session_state.get(_xr_cache_key)
        if xr_cached is not None:
            if isinstance(xr_cached, str):
                st.error(xr_cached)
            elif not xr_cached:
                # WP-2: an empty result used to render nothing at all, which reads as a
                # broken panel. Name the reason instead of leaving it blank.
                st.info(
                    "No exercise-level command evidence was found for this lab. This lab is one "
                    "of the known gaps (workbook section absent, no command in the scope text, or "
                    "no scope text at all) — no mapping is invented to fill it."
                )
            else:
                # Pass lab_display=xr_lab so build_relationships() can perform
                # the actual text co-occurrence check and emit DIRECTLY OBSERVED
                # relationships.  Without this kwarg the relationship section is
                # permanently suppressed.
                _xr_md = format_crossref_table(xr_cached, lab_display=xr_lab)
                st.markdown(_xr_md)
                st.download_button(
                    "⬇️ Download cross-reference (.md)",
                    data=_xr_md,
                    file_name="crossref_" + xr_lab.replace(" / ", "_") + ".md",
                    mime="text/markdown",
                    key="xr_dl_" + xr_lab,
                )

        st.markdown("---")
        st.markdown("**Reverse lookup:** which labs use a command?")
        xr_cmd = st.text_input(
            "Command name (e.g. add_scan_mode, check_design_rules):",
            key="xr_cmd_input",
        )
        if xr_cmd.strip():
            used = where_used(xr_cmd.strip())
            if used:
                st.success("Used in: " + ", ".join(used))
            else:
                st.info("Not found in any lab slide material.")

elif mode == "📝 Exam Question Bank (Real Questions)":
    # ── Owner-only surface: fail-closed, enforced before any data is touched ──
    # Selecting the mode directly (or priming session state) must not bypass
    # this: authorization is decided here, from the authenticated identity.
    if not require_owner("The Exam Question Bank"):
        st.stop()

    def _exam_owner_recheck(action: str) -> None:
        """Re-authorize immediately before each sensitive Exam Bank operation."""
        if not owner_verified():
            st.error(f"🔒 {action} is restricted to the owner account.")
            st.stop()

    _exam_owner_recheck("Loading exam questions")
    st.header("📝 Exam Question Bank — Real Level 1 Exam Questions")
    st.caption(
        "Practice with parsed Level 1 questions. Verified questions require independent "
        "approved-corpus evidence for both the question and its answer."
    )

    questions = load_exam_questions()
    if not questions:
        st.warning("No exam questions found. The exam file may not be parsed yet.")
    else:
        filters = get_exam_filter_options(questions)
        status = get_exam_corpus_status(questions)
        st.caption(
            f"{status['verified']} of {status['total']} records currently have verified corpus evidence "
            f"(question and answer). {status['grounded']} records have question provenance grounded in the "
            "approved exam corpus; their answer keys are still awaiting a reviewed evidence mapping."
        )
        include_unverified = st.checkbox(
            "Include unverified material for review (not scored)",
            value=default_exam_include_unverified(questions),
            key="exam_include_unverified",
        )

        # WP-6: make the four statuses self-explanatory instead of implying an error.
        with st.expander("ℹ️ What the statuses mean (and why nothing is verified)", expanded=False):
            st.markdown(
                "- **Verified (0 of 50)** — the *answer key* is backed by an independently "
                "approved corpus page. None has been adjudicated yet because the Phase 4A "
                "human decisions (Q12, Q9, Q15, Q19, Q34) are **deferred by governance**.\n"
                "- **Grounded (49)** — the *question* text is traced to the approved exam "
                "corpus; the answer is the parse of the source exam and is **not** verified.\n"
                "- **Unmapped (1)** — the record could not be tied to a corpus page at all "
                "(the Q12 fill-in-the-blank; its decision is deferred).\n"
                "- **Unverified** — everything above. Use it for practice only; scores are "
                "not authoritative and no answer is presented as verified.\n\n"
                "The validator, the literal-answer evidence rule and the deferred decisions are "
                "unchanged by this build. Nothing here marks an answer as correct."
            )

        # WP-7: free-text search (text / option / question number) + coverage & export.
        exam_search = st.text_input(
            "🔎 Search questions (text, option, or number — e.g. lockup, 'clock sequential', 12, q47)",
            value="",
            key="exam_search_query",
        )
        if exam_search.strip():
            _exam_owner_recheck("Exam Bank search")
            questions = search_exam_questions(questions, exam_search)
            st.caption(f"**{len(questions)}** record(s) match `{exam_search.strip()}`.")

        with st.expander("📊 Coverage report & export", expanded=False):
            _exam_owner_recheck("The coverage report")
            _coverage = exam_coverage_report(questions)
            st.markdown(
                f"- Total records: **{_coverage['total']}**\n"
                f"- Verified / scorable: **{_coverage['verified']}**\n"
                f"- Status: `{_coverage['status_counts']}`\n"
                f"- Records without question provenance: `{_coverage['unmapped_numbers'] or 'none'}`"
            )
            if _coverage["reason_counts"]:
                st.caption("Blocking validation reasons (records affected):")
                for _reason, _count in list(_coverage["reason_counts"].items())[:8]:
                    st.caption(f"- {_count} × {_reason}")
            _include_keys = st.checkbox(
                "Include answers for verified records only",
                value=False,
                key="exam_export_keys",
                help="Unverified answer keys are never exported, even when this is enabled.",
            )
            _exam_owner_recheck("Exam Bank export")
            _export_md = export_exam_markdown(questions, include_answers=_include_keys)
            _exam_owner_recheck("Exam Bank export download")
            st.download_button(
                "⬇️ Export question bank (.md)",
                data=_export_md,
                file_name="exam_question_bank.md",
                mime="text/markdown",
                key="exam_export_download",
            )
        col1, col2, col3 = st.columns(3)
        with col1:
            sel_module = st.selectbox("Module:", ["All"] + filters["modules"], key="exam_module_sel")
        with col2:
            sel_topic = st.selectbox("Topic:", ["All"] + filters["topics"], key="exam_topic_sel")
        with col3:
            sel_chapter = st.selectbox("Chapter:", ["All"] + filters["chapters"], key="exam_chapter_sel")
        review_topics = get_exam_review_topics(questions)
        sel_review_topic = st.selectbox(
            "Prior exam grouping (review label only, not a canonical mapping):",
            ["All"] + review_topics,
            key="exam_review_topic_sel",
        )
        unclassified_only = st.checkbox("Only unclassified records", value=False, key="exam_unclassified_only")
        randomized = st.checkbox("Randomize order with a reproducible seed", value=False, key="exam_randomized")
        seed = None
        if randomized:
            seed = int(st.number_input("Selection seed", min_value=0, value=0, step=1, key="exam_seed"))
        eligible = select_exam_questions(
            questions,
            module=None if sel_module == "All" else sel_module,
            topic=None if sel_topic == "All" else sel_topic,
            chapter=None if sel_chapter == "All" else sel_chapter,
            review_topic=None if sel_review_topic == "All" else sel_review_topic,
            include_unverified=include_unverified,
            unclassified_only=unclassified_only,
            seed=seed,
        )
        if not eligible:
            if include_unverified:
                st.info("No records match these filters.")
            else:
                st.info(
                    "No verified questions are available for these filters. "
                    "Enable review material to inspect unverified records without authoritative scoring."
                )
        else:
            num_q = st.slider(
                "Number of questions:",
                min_value=1,
                max_value=len(eligible),
                value=min(10, len(eligible)),
                key="exam_num_q",
            )
            quiz_questions = eligible[:num_q]
            quiz_has_verified = any(q["validation_status"] == "verified" for q in quiz_questions)
            if quiz_has_verified:
                st.markdown(f"### 📋 Practice Quiz ({len(quiz_questions)} questions)")
                st.markdown("Select your answer for each question, then click **Submit** to check verified answers.")
            else:
                st.markdown(f"### 🟡 Review Quiz ({len(quiz_questions)} questions, not scored)")
                st.markdown(
                    "These records have no verified answer evidence. Answer them for your own review — "
                    "**Submit** will not produce a score and will not reveal the stored answer key."
                )

            user_answers = {}
            for q in quiz_questions:
                st.markdown("---")
                st.markdown(f"**Q{q['num']}:** {q['text']}")
                st.caption(f"Status: {q['validation_status'].replace('_', ' ')}")
                if q["validation_status"] == "verified":
                    source = q["question_source"]
                    st.caption(
                        f"Question evidence: [Source: {source['source_path']}, "
                        f"Page {source['page']}, Section {source['section']}]"
                    )
                else:
                    st.caption(
                        "🟡 Review material only — the answer key is not verified against the "
                        "approved corpus, so this question is not scorable."
                    )
                    source = q.get("question_source")
                    if source:
                        st.caption(
                            f"Question provenance: [Source: {source['source_path']}, "
                            f"Page {source['page']}, Section {source['section']}]"
                        )
                    if q.get("validation_reasons"):
                        st.caption("Not verified: " + "; ".join(q["validation_reasons"]))
                if len(q.get("correct", [])) > 1 and q["validation_status"] == "verified":
                    user_answers[q["question_id"]] = st.multiselect(
                        f"Your answers for Q{q['num']}:", q["options"], key=f"exam_q_{q['question_id']}"
                    )
                else:
                    user_answers[q["question_id"]] = st.radio(
                        f"Your answer for Q{q['num']}:", q["options"], key=f"exam_q_{q['question_id']}", index=None
                    )

            if st.button("✅ Submit Quiz", key="exam_submit", use_container_width=True):
                results = []
                for q in quiz_questions:
                    answer = user_answers.get(q["question_id"])
                    if answer is None or answer == []:
                        continue
                    results.append((q, answer, evaluate_exam_answer(q, answer)))
                if not results:
                    st.warning("Please answer at least one question before submitting.")
                else:
                    scored = [(q, answer, result) for q, answer, result in results if result["scorable"]]
                    if scored:
                        score = sum(result["correct"] for _, _, result in scored)
                        total = len(scored)
                        pct = (score / total) * 100
                        log_exam_quiz(sel_topic, score, total, pct, [(q["num"], answer, result["correct"]) for q, answer, result in scored])
                        st.markdown(f"### 📊 Verified score: {score}/{total} ({pct:.1f}%)")
                    else:
                        st.info("No answered questions had verified answer evidence, so no authoritative score was calculated.")
                    st.markdown("### 📋 Review Answers")
                    for q, answer, result in results:
                        st.markdown("---")
                        st.markdown(f"**Q{q['num']}:** {q['text']}")
                        st.markdown(f"Your answer: *{answer}*")
                        st.markdown(result["feedback"])
                        if result.get("answer_source"):
                            source = result["answer_source"]
                            st.caption(
                                f"Answer evidence: [Source: {source['source_path']}, "
                                f"Page {source['page']}, Section {source['section']}]"
                            )
                        if result.get("explanation"):
                            st.markdown(result["explanation"])

elif mode == "📦 Batch Export (All Reports)":
    st.header("📦 Batch Export")
    st.subheader("Canonical taxonomy export (offline)")
    st.caption(
        "Deterministic Markdown export from `canonical_taxonomy.json`. This mode never calls an LLM, "
        "rebuilds the index, or changes the knowledge base."
    )
    try:
        _canonical_exporter = BatchExporter(Path(__file__).resolve().parent)
        _export_records = _canonical_exporter.records()

        # WP-4: Module -> Submodule -> Chapter/Lesson -> Theory/Lab, with live counts.
        _COURSE_ORDER = [
            "ATPG Core Topics",
            "ATPG with TestKompress",
            "Tessent Shell Introduction",
            "ATPG Advanced Topics",
            "Tessent Shell",
            "Library User Manual",
            "Other corpus material",
        ]

        def _course_rank(record):
            course = record.node.get("course") or "Other corpus material"
            try:
                return _COURSE_ORDER.index(course)
            except ValueError:
                return len(_COURSE_ORDER)

        _module_options = ["All modules"] + [c for c in _COURSE_ORDER
                                             if any((r.node.get("course") or "Other corpus material") == c for r in _export_records)]
        _scope_module = st.selectbox("Module", _module_options, key="canonical_export_module")
        _module_records = [
            r for r in _export_records
            if _scope_module == "All modules" or (r.node.get("course") or "Other corpus material") == _scope_module
        ]

        _kind_labels = ["Theory + Lab", "Theory only", "Lab only", "All content"]
        _scope_kind = st.selectbox("Content type", _kind_labels, key="canonical_export_kind")
        if _scope_kind == "Theory only":
            _kind_records = [r for r in _module_records if r.node.get("classification") == "Theory"]
        elif _scope_kind == "Lab only":
            _kind_records = [r for r in _module_records if r.node.get("classification") == "Lab"]
        elif _scope_kind == "Theory + Lab":
            _kind_records = [r for r in _module_records if r.node.get("classification") in ("Theory", "Lab")]
        else:
            _kind_records = list(_module_records)

        _submodule_options = sorted({str(r.node.get("submodule", "")) for r in _kind_records})
        _selected_submodules = st.multiselect(
            "Submodules (leave empty for every submodule)",
            _submodule_options, default=[], key="canonical_export_submodules",
        )
        _sub_records = [
            r for r in _kind_records
            if not _selected_submodules or str(r.node.get("submodule", "")) in _selected_submodules
        ]

        _chapter_options = sorted({str(r.node.get("chapter", "")) for r in _sub_records})
        _selected_chapters = st.multiselect(
            "Chapters / lessons (leave empty for every one)",
            _chapter_options, default=[], key="canonical_export_chapters",
        )
        _scoped_records = [
            r for r in _sub_records
            if not _selected_chapters or str(r.node.get("chapter", "")) in _selected_chapters
        ]
        # Curriculum order, not alphabetical domain order.
        _scoped_records.sort(key=lambda r: (
            _course_rank(r), str(r.node.get("submodule", "")), str(r.node.get("chapter", "")), str(r.node.get("topic", "")),
        ))

        st.caption(
            f"Selected scope: **{len(_scoped_records)}** node(s) — theory "
            f"{sum(1 for r in _scoped_records if r.node.get('classification') == 'Theory')}, lab "
            f"{sum(1 for r in _scoped_records if r.node.get('classification') == 'Lab')}, "
            f"unclassified {sum(1 for r in _scoped_records if not r.node.get('classification'))}."
        )
        st.info("ℹ️ Chapter-level export is currently unavailable because reliable chapter boundaries cannot be established from available sources.")
        st.caption("Policy A is active: only explicit Theory/Lab metadata is eligible. Unclassified nodes are not inferred.")

        _record_labels = {
            record.key: f"{record.node.get('course') or 'Other'} / {record.node.get('submodule')} / "
                         f"{record.node.get('chapter')} / {record.node.get('topic')}"
            for record in _scoped_records
        }
        _selected_keys = st.multiselect(
            "Fine-tune individual nodes (leave empty to export the whole scope)",
            list(_record_labels), format_func=_record_labels.get, default=[], key="canonical_export_nodes",
        )
        if not _selected_keys:
            _selected_keys = [record.key for record in _scoped_records]

        if st.button("Prepare canonical Markdown export", key="canonical_export_run", use_container_width=True):
            _canonical_result = _canonical_exporter.export(
                ExportMode.MULTIPLE,
                selected_keys=_selected_keys,
            )
            st.session_state["canonical_export_result"] = _canonical_result
        _canonical_result = st.session_state.get("canonical_export_result")
        if _canonical_result:
            st.json(_canonical_result.summary)
            for _diagnostic in _canonical_result.diagnostics:
                st.warning(_diagnostic)
            st.download_button(
                "Download canonical export (.md)",
                data=_canonical_result.markdown,
                file_name=f"canonical_taxonomy_{_canonical_result.summary['selected']}_nodes.md",
                mime="text/markdown",
                key="canonical_export_download",
            )
    except (OSError, ValueError, json.JSONDecodeError) as _export_error:
        st.error(f"Canonical Batch Export is unavailable: {_export_error}")

    st.divider()
    st.subheader("Legacy Deep Study report export")
    st.caption(
        "Generate all Deep Study reports (video lessons + lab exercises) at once, "
        "saved as .md files on disk. No need to click through each one in the UI."
    )

    # WP-4: counts are computed from the corpus, never hard-coded. The historical
    # "70 lessons / 14 labs" labels were ATPG-Core-only figures that misdescribed the
    # four-module curriculum.
    _ds_modules = list_video_modules()
    _ds_lessons_total = sum(len(list_video_lessons(key)) for key in _ds_modules)
    _ds_labs = list_lab_exercises()
    st.caption(
        f"Available across all courses: **{len(_ds_modules)} modules**, "
        f"**{_ds_lessons_total} video lessons**, **{len(_ds_labs)} lab exercises**."
    )

    _selected_ds_modules = st.multiselect(
        "Limit to modules (leave empty for all)",
        sorted(_ds_modules.values()),
        default=[],
        key="batch_modules",
    )

    _scope_labels = {
        "Video lessons": f"All video lessons ({_ds_lessons_total})",
        "Lab exercises": f"All lab exercises ({len(_ds_labs)})",
        "Everything": f"Everything ({_ds_lessons_total + len(_ds_labs)} reports)",
    }
    export_type = st.radio(
        "What to export:",
        list(_scope_labels),
        format_func=_scope_labels.get,
        key="batch_type",
    )

    st.warning(
        "⚠️ This will make real LLM calls for each report. "
        "Each report takes ~30–90 seconds. Exporting everything may take 1–2 hours. "
        "Reports already cached in memory will be reused."
    )

    if st.button("🚀 Start Batch Export", key="batch_start", use_container_width=True):
        # Determine what to export (module filter honoured for lessons; labs carry their
        # course in the bracket prefix used by list_lab_exercises()).
        def _module_allowed(display_name, module_label):
            if not _selected_ds_modules:
                return True
            course = module_label.split("]", 1)[0].lstrip("[")
            return any(course in selected for selected in _selected_ds_modules)

        items = []
        if export_type in ("Video lessons", "Everything"):
            for num in sorted(_ds_modules.keys()):
                if not _module_allowed(_ds_modules[num], _ds_modules[num]):
                    continue
                for lesson in list_video_lessons(num):
                    items.append(("lesson", num, lesson))
        if export_type in ("Lab exercises", "Everything"):
            for lab in _ds_labs:
                if not _module_allowed(lab, lab):
                    continue
                items.append(("lab", lab, None))

        if not items:
            st.warning("No items match the selected modules and scope.")
            st.stop()

        st.info(f"Generating {len(items)} reports...")

        # Progress bar
        progress = st.progress(0)
        status = st.empty()
        output_dir = st.text_input("Output folder:", value="reports", key="batch_output_dir")

        import os
        os.makedirs(output_dir, exist_ok=True)

        success = 0
        failed = 0
        for i, (kind, identifier, extra) in enumerate(items):
            progress.progress((i + 1) / len(items))
            try:
                if kind == "lesson":
                    report = explain_video_lesson_detail(identifier, extra)
                    fname = f"lesson_{identifier}_{lesson_slug(extra)}.md"
                else:
                    report = explain_lab_exercise(identifier)
                    fname = f"lab_{identifier.replace(' ', '_').replace('/', '_')}.md"

                with open(os.path.join(output_dir, fname), "w", encoding="utf-8") as f:
                    f.write(report)
                success += 1
                status.text(f"✅ {i+1}/{len(items)}: {fname}")
            except Exception as e:
                failed += 1
                status.text(f"❌ {i+1}/{len(items)}: {e}")

        st.success(f"Done! {success} reports saved to '{output_dir}/'. {failed} failed.")
        if failed == 0:
            st.balloons()

# =====================================
# 11. PPT EXPLAINER (WP-10)
# =====================================
elif mode == "🧾 PPT Explainer":
    import hashlib

    from core.ppt_explainer import (PptExplainerError, estimate_deck_usage, explain_deck,
                                    explain_slide, extract_deck, parse_slide_selection)

    try:
        from core.assess_mode import assess_deck, recommended_scope
        from core.assessment_export import build_document, to_docx, to_markdown
        _ASSESS_AVAILABLE = True
    except Exception:  # pragma: no cover - the explainer must survive a partial install
        _ASSESS_AVAILABLE = False

    try:
        from core.session_logger import log_ppt_explainer
    except Exception:  # pragma: no cover - logging must never break the UI
        def log_ppt_explainer(*args, **kwargs):
            return None

    st.header("🧾 PPT Explainer")
    st.caption(
        "Upload a .pptx deck and get slide-by-slide technical explanations grounded in the "
        "Tessent knowledge base. Uploaded presentations are processed locally and are never "
        "written to the session log or its upload payload."
    )

    _ppt_file = st.file_uploader(
        "Upload a PowerPoint presentation (.pptx only)",
        type=["pptx"],
        key="ppt_explainer_upload",
        help="Legacy binary .ppt files are rejected: save the deck as .pptx first.",
    )

    if _ppt_file is None:
        st.info(
            "Supported: `.pptx`. Slides are extracted in order; you can explain one slide at a "
            "time or the whole deck. Model use is optional — extraction and knowledge-base "
            "lookup still work without it."
        )
        st.stop()

    _ppt_bytes = _ppt_file.getvalue()

    # WP-10 privacy: never persist the upload. persist=False writes nothing under
    # session_logs/attachments/ and returns an empty string.
    _stored = save_attachment(_ppt_bytes, _ppt_file.name, persist=False)
    if _stored:
        st.warning("Unexpected: a content-sensitive upload reported a stored path. Nothing was logged.")

    _digest = hashlib.sha256(_ppt_bytes).hexdigest()[:16]
    _deck_cache = st.session_state.setdefault("ppt_decks", {})
    _explain_cache = st.session_state.setdefault("ppt_explanations", {})

    if _digest not in _deck_cache:
        try:
            _deck_cache[_digest] = extract_deck(_ppt_bytes, _ppt_file.name)
        except PptExplainerError as _ppt_error:
            st.error(f"Could not read this presentation: {_ppt_error}")
            st.stop()

    _deck = _deck_cache[_digest]
    if not _deck.slides:
        st.warning("This presentation contains no slides.")
        st.stop()

    st.success(f"Extracted **{_deck.slide_count}** slide(s), in order.")
    _empty_slides = [slide.number for slide in _deck.slides if slide.is_empty]
    if _empty_slides:
        st.caption(f"Slides with no extractable text, table or notes: {_empty_slides}")
    if _deck.warnings:
        with st.expander(f"⚠️ Extraction notes ({len(_deck.warnings)})", expanded=False):
            for _warning in _deck.warnings[:40]:
                st.caption(f"- {_warning}")

    _pictures_total = sum(slide.picture_count for slide in _deck.slides)
    _ctl_col1, _ctl_col2 = st.columns([2, 1])
    with _ctl_col1:
        _interpret_images = st.checkbox(
            f"🖼️ Describe the deck's pictures with the vision model "
            f"({_pictures_total} picture(s) found)",
            value=bool(_pictures_total),
            disabled=not _pictures_total,
            key=f"ppt_interpret_images_{_digest}",
            help=(
                "Pictures are the deck's own material: descriptions are marked unverified, "
                "never cited as manual evidence, and never written to disk or logs. "
                "Turning this off avoids every vision call."
            ),
        )
    with _ctl_col2:
        if st.button("🧹 Clear this deck from the session", key=f"ppt_clear_{_digest}",
                     use_container_width=True):
            _deck_cache.pop(_digest, None)
            for _key in [k for k in st.session_state if str(k).startswith("ppt_deck_result_")]:
                st.session_state.pop(_key, None)
            for _key in [k for k in _explain_cache if isinstance(k, tuple) and k[0] == _digest]:
                _explain_cache.pop(_key, None)
            st.session_state.pop(f"ppt_slide_select_{_digest}", None)
            st.info("Deck removed from this browser session. Nothing was written to disk.")
            st.rerun()

    _estimate = estimate_deck_usage(_deck, interpret_images=_interpret_images)
    with st.expander("💵 Estimated work for this deck (before running)", expanded=False):
        st.markdown(
            f"**Whole deck ({_estimate['slides']} slides)**\n"
            f"- Estimated explanation calls: **{_estimate['estimated_explanation_calls']}** "
            f"(one per slide + one synthesis)\n"
            f"- Estimated vision calls: **{_estimate['estimated_vision_calls']}**\n"
            f"- Estimated total calls: **{_estimate['estimated_total_calls']}**"
        )
        st.caption(
            "These are deck-wide figures. The run below only covers the slides you select, "
            "so its actual counts are expected to be far lower — the estimate for your "
            "selection is shown next to the run buttons."
        )

    st.markdown("#### Slides")
    for _slide in _deck.slides:
        _status = "🟡 empty" if _slide.is_empty else "📄 text"
        if _slide.picture_count:
            _status += f" · 🖼️ {_slide.picture_count}"
        st.caption(f"- {_slide.display_label()} — {_status}")


    # ── Slide selection: whole deck, or a custom set such as "1-5, 8, 12" ─────────
    _slide_spec = st.text_input(
        "Slides to explain",
        value="",
        key=f"ppt_slide_spec_{_digest}",
        placeholder="blank = whole deck, or e.g. 1-5, 8, 12",
        help=(
            "Leave blank (or type 'all') for the whole deck. Otherwise give slide numbers "
            "and/or ranges separated by commas."
        ),
    )
    _selection_error = None
    _selected_numbers = list(range(1, _deck.slide_count + 1))
    try:
        _selected_numbers = parse_slide_selection(_slide_spec, _deck.slide_count)
    except PptExplainerError as _sel_error:
        _selection_error = str(_sel_error)
        _selected_numbers = []
    _is_subset = bool(_slide_spec.strip()) and _slide_spec.strip().lower() not in {
        "all", "*", "every", "whole",
    }
    if _selection_error:
        st.error(_selection_error)
    elif _is_subset and _selected_numbers:
        _preview = ", ".join(str(number) for number in _selected_numbers[:12])
        _more = "" if len(_selected_numbers) <= 12 else f" ... (+{len(_selected_numbers) - 12} more)"
        st.caption(f"{len(_selected_numbers)} slide(s) selected: {_preview}{_more}")

    _selection_estimate = estimate_deck_usage(
        _deck, interpret_images=_interpret_images,
        slides=_selected_numbers if (_is_subset and _selected_numbers) else None,
    )
    if _selection_estimate.get("scope") == "selection":
        st.caption(
            f"Estimated work for this selection: "
            f"**{_selection_estimate['selected_explanation_calls']}** explanation call(s) + "
            f"**{_selection_estimate['selected_vision_calls']}** vision call(s) = "
            f"**{_selection_estimate['selected_total_calls']}** total "
            f"(whole deck would be {_selection_estimate['estimated_total_calls']})."
        )

    _labels = {slide.number: slide.display_label() for slide in _deck.slides}
    _nav_col1, _nav_col2, _nav_col3 = st.columns([2, 1, 1])
    with _nav_col1:
        _selected_slide = st.selectbox(
            "Navigate to a slide",
            list(_labels),
            format_func=_labels.get,
            key=f"ppt_slide_select_{_digest}",
        )
    with _nav_col2:
        if st.button("⬅️ Previous", disabled=_selected_slide <= 1,
                     key=f"ppt_prev_{_digest}", use_container_width=True):
            st.session_state[f"ppt_slide_select_{_digest}"] = max(1, _selected_slide - 1)
            st.rerun()
    with _nav_col3:
        if st.button("Next ➡️", disabled=_selected_slide >= _deck.slide_count,
                     key=f"ppt_next_{_digest}", use_container_width=True):
            st.session_state[f"ppt_slide_select_{_digest}"] = min(_deck.slide_count, _selected_slide + 1)
            st.rerun()

    _slide_obj = _deck.slides[_selected_slide - 1]
    with st.expander(f"📄 Extracted content — {_labels[_selected_slide]}", expanded=True):
        if _slide_obj.text:
            st.markdown("\n".join(f"- {line}" for line in _slide_obj.text.splitlines() if line.strip()))
        else:
            st.caption("No extractable text on this slide.")
        for _table in _slide_obj.tables:
            st.table(_table[:20])
        if _slide_obj.notes:
            st.caption("Speaker notes:")
            st.markdown(_slide_obj.notes)
        for _warning in _slide_obj.warnings:
            st.caption(f"⚠️ {_warning}")

    def _render_slide_result(_result):
        st.markdown(f"#### {_result['label']}")
        _coverage = _result.get("coverage")
        if _coverage == "grounded":
            st.caption(f"Knowledge-base support: {len(_result['evidence'])} relevant chunk(s) retrieved.")
        elif _coverage == "weak_evidence":
            _relevance = _result.get("relevance") or {}
            st.caption(
                "Knowledge-base support: **weak** — nothing retrieved shares the slide's key "
                f"terms ({_relevance.get('matched_terms', 0)} of "
                f"{_relevance.get('slide_terms', 0)}). Treat the explanation as unverified."
            )
        elif _coverage == "no_evidence":
            st.caption(
                "Knowledge-base support: none retrieved for this slide. The explanation is "
                "limited to what the slide itself shows."
            )
        else:
            st.caption("Knowledge-base support: reference lookup was skipped.")

        if _result.get("explanation"):
            st.markdown(_result["explanation"])
        elif _result.get("empty"):
            st.info("This slide has no extractable content to explain.")
        else:
            st.info("No explanation generated yet for this slide.")

        _check = _result.get("citation_check") or {}
        if _check.get("total"):
            try:
                from core.citation_check import summary_line as _citation_summary

                _check_line = _citation_summary(_check)
            except Exception:
                _check_line = ""
            if _check_line:
                if _check.get("counts", {}).get("verified") == _check.get("total"):
                    st.caption(f"✅ {_check_line}")
                elif _check.get("pages_missing"):
                    st.warning(f"⚠️ {_check_line}")
                else:
                    st.caption(f"🔎 {_check_line}")

        if _result.get("model_error"):
            st.warning(
                f"Model unavailable ({_result['model_error']}). Extraction and references are "
                "still available; retry when the model is reachable."
            )

        if _result.get("citations"):
            with st.expander(f"📖 Reference citations ({len(_result['citations'])})", expanded=False):
                for _citation in _result["citations"]:
                    st.caption(f"- {_citation}")
                st.caption(
                    "Slide content is shown separately above; citations come only from the "
                    "knowledge base."
                )
        elif _coverage == "no_evidence":
            st.caption("No knowledge-base citations: the sources do not cover this slide.")

        _sources = _result.get("sources") or []
        if _sources:
            _kinds_summary = _result.get("source_kinds") or {}
            if _kinds_summary.get("line"):
                st.caption(f"🧩 Sources combined — {_kinds_summary['line']}")
            _manual_sources = [s for s in _sources if s.get("kind") == "manual"]
            _visual_sources = [s for s in _sources if s.get("kind") == "visual"]
            _kind_names = {
                "manual": "Manual",
                "video_slide": "Video slide (frame on screen)",
                "video_transcript": "Video transcript (spoken)",
                "lab_workbook": "Lab workbook",
                "course_exam": "Course exam",
                "reference": "Knowledge base",
            }
            with st.expander(
                f"Where this explanation comes from — {len(_manual_sources)} manual page(s)"
                + (f", {len(_kinds_summary.get('kinds') or [])} kind(s) of source"
                   if _kinds_summary.get("kinds") else ""),
                expanded=False,
            ):
                for _source in _sources:
                    _kind = _source.get("source_kind") or _source.get("kind") or "reference"
                    if _source.get("kind") == "visual":
                        st.markdown(f"- **[Deck picture, unverified]** {_source['label']}")
                    elif _source.get("kind") == "slide":
                        st.markdown(f"- **[Slide]** {_source['label']}")
                    else:
                        _origin = (" (linked to this slide)" if _source.get("origin")
                                   in {"correlated", "deck_picture"} else "")
                        st.markdown(
                            f"- **[{_kind_names.get(_kind, _kind)}]** {_source['label']}"
                            + _origin
                        )
                if _result.get("picture_terms_used"):
                    st.caption(
                        "This slide had little or no text, so its picture descriptions were "
                        "used to search the knowledge base — that is why pages were found."
                    )
                if _visual_sources:
                    st.caption(
                        "Picture descriptions carry no page number and are not themselves "
                        "evidence; where a picture is linked to a video-slide or transcript "
                        "page, that page is what supports the claim."
                    )

        _images = _result.get("images") or {}
        if _images.get("observations"):
            with st.expander(
                f"🖼️ Visual observations ({len(_images['observations'])}) — unverified",
                expanded=False,
            ):
                for _obs in _images["observations"]:
                    _obs_head = f"figure {_obs.get('ordinal', '?')} of this slide"
                    if _obs.get("observed_title"):
                        _obs_head += f" — its own heading reads “{_obs['observed_title']}”"
                    _linked = _obs.get("linked_pages") or []
                    if _linked:
                        _where = "; ".join(
                            f"{item['source']} page {item['page']}" for item in _linked)
                        _obs_head += f" · linked to {_where}"
                    st.markdown(f"- **{_obs_head}**: {_obs['description']}")
                if _images.get("mismatched_titles"):
                    st.warning(
                        "A picture on this slide carries its own heading which differs from "
                        "the slide title ("
                        + "; ".join(f"“{title}”" for title in _images["mismatched_titles"])
                        + "). Such pictures are frames from the source video embedded in the "
                        "slide, so they show neighbouring topics — not this slide's own text."
                    )
                st.caption(
                    "Described from the deck's own pictures. This is not manual evidence: it "
                    "carries no page number and is never used as the source of a technical claim. "
                    "Video-player furniture is filtered out."
                )
        elif _images.get("error"):
            st.caption(
                f"Picture description unavailable ({_images['error']}); the text explanation is "
                "unaffected."
            )
        elif _result.get("picture_count") and _images.get("available") is False:
            st.caption("Picture description was skipped for this slide.")

    _run_col1, _run_col2 = st.columns(2)
    with _run_col1:
        _explain_one = st.button("🔍 Explain this slide", key=f"ppt_explain_one_{_digest}",
                                 use_container_width=True)
    with _run_col2:
        _deck_button_label = (f"📚 Explain {len(_selected_numbers)} selected slide(s)"
                              if _is_subset else "📚 Explain the whole deck")
        _explain_all = st.button(_deck_button_label, key=f"ppt_explain_all_{_digest}",
                                 use_container_width=True,
                                 disabled=_selection_error is not None)

    if _explain_one:
        if not ensure_llm_access():
            st.stop()
        with st.spinner(f"Explaining slide {_selected_slide}..."):
            _explain_cache[(_digest, _selected_slide)] = explain_slide(
                _deck, _selected_slide, interpret_images=_interpret_images)
        log_ppt_explainer(_deck.slide_count, 1, "single_slide")

    _cached_single = _explain_cache.get((_digest, _selected_slide))
    if _cached_single:
        _render_slide_result(_cached_single)

    if _explain_all:
        if not ensure_llm_access():
            st.stop()
        _progress = st.progress(0)
        _status = st.empty()

        def _on_progress(done, total):
            _progress.progress(done / max(1, total))
            _status.text(f"Explained {done}/{total} slide(s)...")

        with st.spinner("Explaining the selected slides in order... this can take a while."):
            _deck_result = explain_deck(_deck, interpret_images=_interpret_images,
                                        on_progress=_on_progress,
                                        slides=_selected_numbers if _is_subset else None)
        for _item in _deck_result["slides"]:
            _explain_cache[(_digest, _item["slide_number"])] = _item
        st.session_state[f"ppt_deck_result_{_digest}"] = _deck_result
        log_ppt_explainer(_deck.slide_count, _deck_result["explained_count"],
                          "subset" if _is_subset else "whole_deck")
        _progress.progress(1.0)
        _status.text("Deck explained.")

    _deck_result = st.session_state.get(f"ppt_deck_result_{_digest}")
    if _deck_result:
        st.markdown("---")
        _subset = bool(_deck_result.get("selected_slides")) and (
            _deck_result.get("explained_slides") != _deck_result.get("slide_count"))
        st.markdown("### 📚 Slide explanation" + (" (custom set)" if _subset else " (whole deck)"))
        if _subset:
            _chosen = ", ".join(str(number) for number in _deck_result["selected_slides"][:20])
            st.caption(
                f"{_deck_result['explained_count']} of {_deck_result['explained_slides']} "
                f"selected slide(s) explained. Selected: {_chosen}"
            )
        else:
            st.caption(
                f"{_deck_result['explained_count']} of {_deck_result['slide_count']} slide(s) explained."
            )
        if _deck_result.get("synthesis"):
            with st.expander("🧭 Presentation-level synthesis", expanded=True):
                st.markdown(_deck_result["synthesis"])
        elif _deck_result.get("synthesis_error"):
            st.warning(
                f"Presentation synthesis unavailable ({_deck_result['synthesis_error']}). "
                "Per-slide explanations are below."
            )
        for _item in _deck_result["slides"]:
            with st.expander(_item["label"], expanded=False):
                _render_slide_result(_item)

        _usage = _deck_result.get("usage") or {}
        if _usage:
            with st.expander("📈 Actual usage for this deck run", expanded=False):
                _actual = _usage.get("actual") or {}
                _planned = _usage.get("estimated") or {}
                st.markdown(
                    f"- Explanation calls: **{_actual.get('explanation_calls', 0)}** "
                    f"(estimated {_planned.get('estimated_explanation_calls', 0)})\n"
                    f"- Vision calls: **{_actual.get('vision_calls', 0)}** "
                    f"(estimated {_planned.get('estimated_vision_calls', 0)})"
                )
                if _actual.get("failed_vision_calls"):
                    st.caption(
                        f"{_actual['failed_vision_calls']} picture(s) could not be described; "
                        "the text explanations are unaffected."
                    )
                _budget = _usage.get("budget")
                if isinstance(_budget, dict) and _budget.get("max_calls") is not None:
                    st.caption(
                        f"Session budget: {_budget.get('calls', 0)} call(s) used of "
                        f"{_budget['max_calls']}, {_budget.get('remaining_calls')} remaining."
                    )
                st.caption(_actual.get("token_note") or "Token usage is unavailable for this call path.")

        _export_lines = [f"# PPT Explainer — {_deck.slide_count} slide(s)", ""]
        for _item in _deck_result["slides"]:
            _export_lines.append(f"## {_item['label']}")
            if _item.get("explanation"):
                _export_lines.append(_item["explanation"])
            if _item.get("citations"):
                _export_lines.append("")
                _export_lines.append("References:")
                _export_lines.extend(f"- {_citation}" for _citation in _item["citations"])
            if _item.get("coverage") == "no_evidence":
                _export_lines.append("_No knowledge-base evidence was retrieved for this slide._")
            _export_lines.append("")
        if _deck_result.get("synthesis"):
            _export_lines.extend(["## Presentation synthesis", _deck_result["synthesis"], ""])
        st.download_button(
            "⬇️ Download explanations (.md)",
            data="\n".join(_export_lines),
            file_name="ppt_explanations.md",
            mime="text/markdown",
            key=f"ppt_export_{_digest}",
        )

    # ── Fresher assessment mode (WP-11) ──────────────────────────────────────
    if _ASSESS_AVAILABLE:
        st.markdown("---")
        st.markdown("### 🎓 Fresher assessment")
        st.caption(
            "Turns the same grounded pipeline into a trainer's sheet: three must-knows, the "
            "mistake freshers usually make, one check question with its expected answer, a "
            "quiz, and a provenance appendix listing every manual page used. Exports to .docx."
        )
        _scope = recommended_scope(_deck)
        st.markdown(f"**Recommended scope:** {_scope['note']}")
        if _scope.get("skipped_labs"):
            st.caption("LAB exercise slides skipped: "
                       + ", ".join(str(number) for number in _scope["skipped_labs"]))
        for _flow in _scope["flows"]:
            if _flow["slides"]:
                st.caption(f"- {_flow['name']}: slides "
                           + ", ".join(str(number) for number in _flow["slides"]))

        _a_col1, _a_col2, _a_col3 = st.columns([1, 1, 2])
        with _a_col1:
            _assess_recommended = st.checkbox("Recommended scope", value=True,
                                              key=f"assess_rec_{_digest}")
        with _a_col2:
            _assess_quiz_count = st.number_input("Quiz questions", min_value=0, max_value=25,
                                                 value=10, key=f"assess_quiz_{_digest}")
        with _a_col3:
            _assess_title = st.text_input("Sheet title", value="Fresher assessment",
                                          key=f"assess_title_{_digest}")

        _assess_slides = _scope["slides"] if _assess_recommended else list(_selected_numbers)
        _assess_blocked = (not _assess_recommended) and _selection_error is not None
        if _assess_blocked:
            st.error("Fix the slide selection above first.")
        else:
            st.caption(f"{len(_assess_slides)} slide(s) will be assessed"
                       + (f": {', '.join(str(n) for n in _assess_slides)}"
                          if _assess_slides else ""))

        if st.button(f"🎓 Build the assessment ({len(_assess_slides)} slides + quiz)",
                     key=f"assess_run_{_digest}", use_container_width=True,
                     disabled=_assess_blocked or not _assess_slides):
            if ensure_llm_access():
                _assess_progress = st.progress(0)
                _assess_status = st.empty()

                def _on_assess_progress(done, total):
                    _assess_progress.progress(done / max(1, total))
                    _assess_status.text(f"Assessing {done}/{total} slide(s)...")

                try:
                    with st.spinner("Building the assessment sheet..."):
                        st.session_state[f"ppt_assess_{_digest}"] = assess_deck(
                            _deck,
                            slides=None if _assess_recommended else _assess_slides,
                            interpret_images=False,
                            quiz_count=int(_assess_quiz_count),
                            on_progress=_on_assess_progress,
                        )
                    _assess_progress.progress(1.0)
                    _assess_status.text("Assessment built.")
                except Exception as _assess_error:
                    st.error(f"Assessment failed ({type(_assess_error).__name__}). The "
                             "per-slide explanations above are unaffected.")

        _assess_result = st.session_state.get(f"ppt_assess_{_digest}")
        if _assess_result:
            _assess_coverage = _assess_result.get("coverage") or {}
            st.markdown(f"**Coverage:** {_assess_coverage.get('statement', '')}")

            for _a_item in _assess_result.get("assessments") or []:
                with st.expander(_a_item["label"], expanded=False):
                    if _a_item.get("must_know"):
                        st.markdown("**Must be able to state**")
                        for _point in _a_item["must_know"]:
                            st.markdown(f"- {_point}")
                    if _a_item.get("misconception"):
                        st.markdown(f"**Typical mistake:** {_a_item['misconception']}")
                    if _a_item.get("check_question"):
                        st.markdown(f"**Check question:** {_a_item['check_question']}")
                    if _a_item.get("expected_answer"):
                        st.markdown(f"**Expected answer:** {_a_item['expected_answer']}")
                    if _a_item.get("not_covered"):
                        st.warning(f"Not covered by the knowledge base: {_a_item['not_covered']}")
                    if _a_item.get("parse_error"):
                        st.warning(f"{_a_item['parse_error']} - the reply is shown raw.")
                        st.code(_a_item.get("raw_text") or "", language="text")
                    _a_check = _a_item.get("citation_check") or {}
                    if _a_check.get("total"):
                        try:
                            from core.citation_check import summary_line as _citation_summary

                            st.caption("🔎 " + _citation_summary(_a_check))
                        except Exception:
                            pass

            _assess_quiz = (_assess_result.get("quiz") or {}).get("questions") or []
            if _assess_quiz:
                st.markdown("#### End-of-session quiz")
                for _q_index, _q in enumerate(_assess_quiz, start=1):
                    st.markdown(f"{_q_index}. {_q['question']}")
                with st.expander("Answer key", expanded=False):
                    for _q_index, _q in enumerate(_assess_quiz, start=1):
                        _where = f" (slide {_q['slide']})" if _q.get("slide") else ""
                        st.markdown(f"{_q_index}. {_q['answer']}{_where}")
            elif (_assess_result.get("quiz") or {}).get("skipped"):
                st.caption(f"Quiz not built: {_assess_result['quiz']['skipped']}")

            try:
                # The uploaded file name is deliberately NOT put into the handout.
                _handout = build_document(
                    _assess_result,
                    title=_assess_title,
                    deck_label=f"{_deck.slide_count}-slide presentation",
                )
                _dl_col1, _dl_col2 = st.columns(2)
                with _dl_col1:
                    st.download_button(
                        "⬇️ Download assessment (.docx)", data=to_docx(_handout),
                        file_name="fresher_assessment.docx",
                        mime="application/vnd.openxmlformats-officedocument."
                             "wordprocessingml.document",
                        key=f"assess_docx_{_digest}", use_container_width=True,
                    )
                with _dl_col2:
                    st.download_button(
                        "⬇️ Download assessment (.md)", data=to_markdown(_handout),
                        file_name="fresher_assessment.md", mime="text/markdown",
                        key=f"assess_md_{_digest}", use_container_width=True,
                    )
            except Exception as _export_error:
                st.warning(f"Export unavailable ({type(_export_error).__name__}); the "
                           "assessment is still readable above.")

            _assess_usage = _assess_result.get("usage") or {}
            st.caption(
                f"Calls used: {_assess_usage.get('assessment_calls', 0)} assessment + "
                f"{_assess_usage.get('quiz_calls', 0)} quiz + "
                f"{_assess_usage.get('vision_calls', 0)} vision."
            )

