import streamlit as st
import re

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
from agents.evaluator_agent import evaluate_answer
from agents.planner_agent import generate_study_plan
from agents.lab_explainer_agent import explain_lab_command
from agents.question_bank_agent import generate_assessment_question_bank
from agents.deep_study_agent import (
    list_video_modules, list_lab_exercises,
    list_video_lessons, explain_video_lesson_detail, lesson_slug,
    explain_video_lesson, explain_lab_exercise, explain_video_lesson_detail_v2,
    explain_lab_exercise_v2, list_lab_files_at_once,
    generate_lab_ppt_from_report, generate_lesson_ppt_from_report,
)
from agents.crossref_agent import crossref_lab, where_used, format_crossref_table
from agents.exam_bank_agent import load_questions, get_questions_by_topic, get_topics, format_question, format_quiz, evaluate_answer
from core.rag_builder import upload_and_index_pdf, index_text_file
from agents.attachment_agent import ask_with_text_attachment, ask_with_image
from core.file_reader import document_text_to_string, is_text_ext, is_image_ext
from core.auth import require_auth, auth_badge_sidebar
from core.session_logger import (
    log_qa, log_qa_error, save_attachment,
    log_lab_explainer, log_lab_explainer_error,
    log_question_bank,
    log_viva_question, log_viva_answer_and_evaluation,
    log_study_plan,
)

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Tessent Mentor AI - Assessment Edition", page_icon="🧠", layout="wide")

# =====================================
# PRIVATE KNOWLEDGE BASE RESTORE (public-repo deploy)
# If this deployment has no local knowledge base (e.g. the public code
# repo on Streamlit Cloud), fetch the private bundle once from a GitHub
# release. See core/bootstrap.py; no-op when data is already present.
# =====================================
from core.bootstrap import restore_done, restore_knowledge

if not restore_done():
    restore_knowledge()

# =====================================
# AUTH GATE (password + optional email allowlist)
# Set APP_PASSWORD / APP_ALLOWED_EMAILS in .env or Streamlit secrets.
# If APP_PASSWORD is unset, the app runs open (local dev default).
# =====================================
if not require_auth():
    st.stop()

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
    st.title("🧠 Tessent Mentor AI")
    st.caption("1-Week Assessment Preparation Edition")
    
    mode = st.radio(
        "Navigation Mode",
        [
            "📤 Upload Manuals & Labs",
            "📖 Ask Question & Manual Citation",
            "🧪 Lab & Command Explainer",
            "❓ Assessment Question Generator",
            "🎤 Interactive Viva Practice",
            "📚 DFT Study Planner",
            "🔬 Deep Study (Videos & Labs)",
            "📝 Exam Question Bank (Real Questions)",
            "📦 Batch Export (All Reports)",
        ]
    )

    st.markdown("---")

    # Always-visible logging status so the user knows every interaction is saved.
    from core.session_logger import LOG_DIR
    st.markdown("### 📝 Session Logging")
    st.caption("✅ Active — everything you ask & every answer is saved.")
    st.caption(f"Folder: `{LOG_DIR}`")
    try:
        log_files = sorted(LOG_DIR.glob("*.log"))
        if log_files:
            st.caption(f"Files: `{', '.join(f.name for f in log_files[-3:])}`")
    except Exception:
        pass

    st.markdown("---")

    if mode == "📖 Ask Question & Manual Citation":
        st.markdown("### Core Assessment Modules")
        st.markdown("- 🟢 **SCAN** (Tessent Scan)")
        st.markdown("- 🔵 **ATPG** (Tessent FastScan)")
        st.markdown("- 🟣 **EDT** (Tessent TestKompress)")
        st.markdown("- 🐧 **LINUX / TSHELL** (Environment & TCL Flow)")
        
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
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
    You can also copy your PDF manual files directly into your project directory on your computer:
    - **Shell User Manual:** `C:\\Wipro\\vlsi-mentor-ai\\knowledge\\01_Tessent_Shell_User_Manual\\`
    - **Shell Reference Manual:** `C:\\Wipro\\vlsi-mentor-ai\\knowledge\\02_Tessent_Shell_Reference_Manual\\`
    - **ATPG & Scan User Manual:** `C:\\Wipro\\vlsi-mentor-ai\\knowledge\\03_Scan_and_ATPG_User_Manual\\`
    - **Library User Manual:** `C:\\Wipro\\vlsi-mentor-ai\\knowledge\\04_Library_User_Manual\\`
    - **Scan & ATPG Lab Manual:** `C:\\Wipro\\vlsi-mentor-ai\\knowledge\\05_Scan_and_ATPG_Lab_Manual\\`

    After copying PDFs directly into those folders, run the re-indexing command in PowerShell:
    ```powershell
    .\\venv\\Scripts\\python.exe core/rag_builder.py build
    ```
    """)


# =====================================
# 1. ASK QUESTION & MANUAL CITATION MODE
# =====================================
elif mode == "📖 Ask Question & Manual Citation":
    st.title("📖 Tessent Reference Manual Assistant")
    st.caption("Ask questions grounded strictly in official Tessent reference manuals with exact page & section citations.")

    # Display Old Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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

    # Chat Input
    question = st.chat_input("Ask a Tessent Scan, ATPG, EDT, or TShell question...")

    if question:
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
                    else:
                        agent = agent_map.get(topic, ask_general)
                        answer = agent(question, history[:-1])

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
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
    st.title("🧪 Tessent Lab & Command Explainer (POC Prep)")
    st.caption("Analyze lab scripts step-by-step. Get internal DB mechanics, flow placement, output files, missing command impact, and top POC viva questions.")

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

# =====================================
# 3. ASSESSMENT QUESTION GENERATOR MODE
# =====================================
elif mode == "❓ Assessment Question Generator":
    st.title("❓ Tessent Assessment & Viva Bank Generator")
    st.caption("Generate a targeted 20-question viva bank for any core module, with benchmark model answers and manual citations.")

    col1, col2 = st.columns(2)
    with col1:
        target_module = st.selectbox("Select Core Assessment Module", ["SCAN", "ATPG", "EDT", "LINUX / TSHELL"])
    with col2:
        num_q = st.slider("Number of Viva Questions", min_value=5, max_value=20, value=15)

    if st.button("🎲 Generate Viva Question Bank", use_container_width=True):
        with st.spinner(f"Building {num_q}-question Assessment Bank for {target_module}..."):
            q_bank = generate_assessment_question_bank(target_module, num_questions=num_q)
            st.markdown(q_bank)
            log_question_bank(target_module, num_q, q_bank)

# =====================================
# 4. INTERACTIVE VIVA PRACTICE MODE
# =====================================
elif mode == "🎤 Interactive Viva Practice":
    st.title("🎤 Tessent Mock Assessment & Viva Simulator")
    st.caption("Practice mock viva questions. Your answers will be graded against Tessent reference manual criteria to highlight your score and weak spots.")

    col1, col2 = st.columns(2)
    with col1:
        topic = st.selectbox(
            "Select Viva Topic",
            ["SCAN", "ATPG", "EDT", "LINUX", "TCL", "MBIST", "JTAG", "STA"]
        )
    with col2:
        difficulty = st.selectbox("Difficulty Level", ["Beginner (Basic Commands)", "Intermediate (Lab DRCs)", "Advanced (POC Tricky Scenarios)"])

    if st.button("Generate Viva Question", use_container_width=True):
        with st.spinner("Generating mock viva question..."):
            question = ask_interview_question(topic, difficulty, st.session_state["asked_questions"])
            st.session_state["interview_question"] = question
            st.session_state["asked_questions"].append(question)
            st.session_state.pop("evaluation", None)
            log_viva_question(topic, difficulty, question)

    if "interview_question" in st.session_state:
        st.markdown("### 🎙️ POC Assessor Question")
        st.info(st.session_state["interview_question"])

        candidate_answer = st.text_area("Type your explanation / answer below:", height=180, placeholder="Explain what the command does, why it's placed here, what file is generated, or what happens if skipped...")

        if st.button("Submit Answer for Evaluation", use_container_width=True):
            if candidate_answer.strip():
                with st.spinner("Grading response against Tessent Reference Manual criteria..."):
                    result = evaluate_answer(st.session_state["interview_question"], candidate_answer)
                    st.session_state["evaluation"] = result
                    st.session_state.evaluations.append(result)
                    log_viva_answer_and_evaluation(
                        st.session_state["interview_question"], candidate_answer, result
                    )
            else:
                st.warning("Please type an answer before submitting.")

    if "evaluation" in st.session_state:
        st.markdown("### 📋 Assessor Evaluation Report")
        st.markdown(st.session_state["evaluation"])

# =====================================
# 5. DFT STUDY PLANNER MODE
# =====================================
elif mode == "📚 DFT Study Planner":
    st.title("📚 Personalized 7-Day Assessment Study Guide")
    st.caption("Track your practice metrics, visualize score trends, and follow your customized revision schedule.")

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
                    readiness = "Assessment Ready (8–10)"
                elif avg_score >= 6:
                    readiness = "Passing Level (6–7.9)"
                st.metric("Assessment Readiness", readiness)

            st.markdown("### 📈 Viva Score History")
            st.line_chart(scores)
        else:
            st.info("Start practicing viva questions in **Interactive Viva Practice** mode to see your score trends here!")

        st.markdown("---")
        st.markdown("### 📋 Your Personalized 7-Day Study Guide")

        if "cached_study_plan" in st.session_state:
            st.markdown(st.session_state["cached_study_plan"])
            if st.button("🔄 Regenerate 7-Day Assessment Plan", use_container_width=True):
                with st.spinner("Regenerating 7-day revision plan..."):
                    study_plan = generate_study_plan(st.session_state.asked_questions, st.session_state.evaluations)
                    st.session_state["cached_study_plan"] = study_plan
                    log_study_plan(study_plan)
                    st.rerun()
        else:
            st.info("Click the button below to generate your personalized 7-day study plan based on your viva practice scores.")
            if st.button("📝 Generate 7-Day Study Plan", use_container_width=True):
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
            sel_lesson = st.selectbox(
                "Choose an individual video lesson:",
                lesson_options,
                key="ds_lesson_sel",
            )

            if sel_lesson == "Whole module (all lessons)":
                if st.button("🎥 Generate Deep Study Report", key="ds_video_btn", use_container_width=True):
                    with st.spinner("Reading transcript, slides and manuals... (this can take a minute)"):
                        st.session_state["ds_video_report_" + str(sel_num)] = explain_video_lesson(sel_num)
                cached_video = st.session_state.get("ds_video_report_" + str(sel_num))
                if cached_video:
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
                    with st.spinner("Reading this lesson's transcript, slides and manuals... (up to a minute)"):
                        report, ppt_path, dia = explain_video_lesson_detail_v2(
                            sel_num, sel_lesson, generate_ppt=False, include_diagrams=include_diagrams
                        )
                        st.session_state[cache_key] = report
                        st.session_state[ppt_key] = ppt_path
                        st.session_state[dia_key] = dia
                cached_lesson = st.session_state.get(cache_key)
                if cached_lesson:
                    # Layman's explanation expander
                    if "## Layman's Explanation" in cached_lesson:
                        with st.expander("🧑‍🏫 Layman's Explanation (Beginner Friendly)", expanded=True):
                            layman_start = cached_lesson.index("## Layman's Explanation")
                            layman_end = cached_lesson.index("## ", layman_start + 1)
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

    with lab_tab:
        lab_list = list_lab_exercises()
        if not lab_list:
            st.warning("No lab exercises found under knowledge/Tessent Atpg Core Topics/10.Labs.")
        else:
            sel_lab = st.selectbox("Choose a lab exercise:", lab_list, key="ds_lab_sel")
            
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
                with st.spinner("Reading ALL lab files: slides, TCL, screenshots, manuals... (this can take 1-2 minutes)"):
                    report, ppt_path, dia = explain_lab_exercise_v2(
                        sel_lab, generate_ppt=False, include_diagrams=include_diagrams
                    )
                    st.session_state[lab_cache_key] = report
                    st.session_state[lab_ppt_key] = ppt_path
                    st.session_state[lab_dia_key] = dia
            
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

    with st.expander("🔗 Lab ↔ Manual Cross-Reference", expanded=False):
        st.caption(
            "Deterministic command extraction per lab: every Tessent command found in the lab's "
            "material is matched to its full manual definition (usage/options) with page citations, "
            "plus a reverse lookup of where else the command is used. No LLM, no hallucination."
        )
        xr_lab = st.selectbox(
            "Cross-reference which lab?",
            list_lab_exercises(),
            key="xr_lab_sel",
        )
        if st.button("🔍 Build Cross-Reference", key="xr_btn", use_container_width=True):
            with st.spinner("Extracting commands and matching manual definitions..."):
                try:
                    rows = crossref_lab(xr_lab)
                    st.session_state["xr_rows_" + xr_lab] = rows
                except Exception as exc:
                    st.session_state["xr_rows_" + xr_lab] = "ERROR: " + str(exc)
        xr_cached = st.session_state.get("xr_rows_" + xr_lab)
        if xr_cached:
            if isinstance(xr_cached, str):
                st.error(xr_cached)
            else:
                st.markdown(format_crossref_table(xr_cached))
                st.download_button(
                    "⬇️ Download cross-reference (.md)",
                    data=format_crossref_table(xr_cached),
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
    st.header("📝 Exam Question Bank — Real Level 1 Exam Questions")
    st.caption(
        "Practice with actual exam questions parsed from the training platform. "
        "These are the real questions from the Level 1 Exam — not AI-generated."
    )

    questions = load_questions()
    if not questions:
        st.warning("No exam questions found. The exam file may not be parsed yet.")
    else:
        topics = get_topics()
        col1, col2 = st.columns(2)
        with col1:
            sel_topic = st.selectbox(
                "Filter by topic:",
                ["All"] + topics,
                key="exam_topic_sel",
            )
        with col2:
            num_q = st.slider(
                "Number of questions:",
                min_value=5,
                max_value=min(48, len(questions)),
                value=10,
                key="exam_num_q",
            )

        topic_filter = None if sel_topic == "All" else sel_topic
        quiz_questions = get_questions_by_topic(topic=topic_filter, count=num_q, shuffle=True)

        st.markdown(f"### 📋 Practice Quiz ({len(quiz_questions)} questions)")
        st.markdown("Select your answer for each question, then click **Submit** to check.")

        # Quiz form
        user_answers = {}
        for i, q in enumerate(quiz_questions):
            st.markdown(f"---")
            st.markdown(f"**Q{q['num']}:** {q['text']}")
            if q.get('options'):
                options = q['options']
                if q.get('correct'):
                    options = options + q['correct']
                user_answers[q['num']] = st.radio(
                    f"Your answer for Q{q['num']}:",
                    options,
                    key=f"exam_q_{q['num']}",
                    index=None,
                )

        if st.button("✅ Submit Quiz", key="exam_submit", use_container_width=True):
            if not any(v is not None for v in user_answers.values()):
                st.warning("Please answer at least one question before submitting.")
            else:
                score = 0
                total = 0
                results = []
                for q in quiz_questions:
                    ua = user_answers.get(q['num'])
                    if ua is None:
                        continue
                    total += 1
                    result = evaluate_answer(q, ua)
                    if result['correct']:
                        score += 1
                    results.append((q, ua, result))

                if total > 0:
                    pct = (score / total) * 100
                    st.markdown(f"### 📊 Score: {score}/{total} ({pct:.1f}%)")
                    if pct >= 80:
                        st.balloons()
                        st.success("🎉 Excellent! You're ready for the real exam!")
                    elif pct >= 60:
                        st.success("👍 Passing level! Keep practicing to improve.")
                    else:
                        st.warning("📚 Needs more practice. Review the Deep Study reports for weak topics.")

                    st.markdown("### 📋 Review Answers")
                    for q, ua, result in results:
                        st.markdown(f"---")
                        st.markdown(f"**Q{q['num']}:** {q['text']}")
                        st.markdown(f"Your answer: *{ua}*")
                        st.markdown(result['feedback'])

elif mode == "📦 Batch Export (All Reports)":
    st.header("📦 Batch Export — Generate All Reports to Files")
    st.caption(
        "Generate all Deep Study reports (video lessons + lab exercises) at once, "
        "saved as .md files on disk. No need to click through each one in the UI."
    )

    export_type = st.radio(
        "What to export:",
        ["All Video Lessons (70 lessons)", "All Lab Exercises (14 labs)", "Everything (lessons + labs)"],
        key="batch_type",
    )

    st.warning(
        "⚠️ This will make real LLM calls for each report. "
        "Each report takes ~30–90 seconds. Exporting everything may take 1–2 hours. "
        "Reports already cached in memory will be reused."
    )

    if st.button("🚀 Start Batch Export", key="batch_start", use_container_width=True):
        # Determine what to export
        if export_type == "All Video Lessons (70 lessons)":
            mods = list_video_modules()
            items = []
            for num in sorted(mods.keys()):
                lessons = list_video_lessons(num)
                for lesson in lessons:
                    items.append(("lesson", num, lesson))
        elif export_type == "All Lab Exercises (14 labs)":
            labs = list_lab_exercises()
            items = [("lab", lab, None) for lab in labs]
        else:  # Everything
            mods = list_video_modules()
            items = []
            for num in sorted(mods.keys()):
                lessons = list_video_lessons(num)
                for lesson in lessons:
                    items.append(("lesson", num, lesson))
            labs = list_lab_exercises()
            items.extend([("lab", lab, None) for lab in labs])

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

