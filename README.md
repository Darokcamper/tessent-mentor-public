# 🧠 Tessent Mentor AI — Assessment Edition

Tessent Mentor AI is a production-quality, multi-agent educational platform for preparing for Tessent / VLSI Design-for-Test (DFT) assessments and interviews. Every answer is grounded strictly in the official Tessent reference manuals using a page-level citation RAG pipeline, and every question & answer is logged to disk.

---

## 🚀 Key Features

1. **Manual-Grounded Q&A with Exact Citations** — A 14-domain expert router (SCAN, ATPG, EDT, MBIST, JTAG, IJTAG, WRAPPER, OCC, BOUNDARYSCAN, STA, GLS, TCL, LINUX, GENERAL) answers strictly from retrieved manual excerpts with `[Source: file, Page N]` citations. If the manuals don't cover a topic, the agent says so instead of guessing.
2. **File & Image Attachment** — Attach a PDF, DOCX, PPTX, TXT, or image (waveform/screenshot/log) directly with a question. Documents are read for context (still grounded strictly in the manuals), and images are analyzed by the Gemini vision model. Optionally index attached documents permanently into the knowledge base.
2. **Lab & Command Explainer** — Explains Tessent shell commands and lab scripts from the Shell Reference / lab manuals.
3. **Assessment Question Generator** — Generates module-wise practice question banks.
4. **Interactive Viva Practice** — Mock interview questions evaluated per answer (score + strengths/weaknesses).
5. **DFT Study Planner** — Analyzes viva performance and generates a personalized study plan.
6. **Robust OCR Pipeline** — `ocrmypdf` for scanned PDFs, plus RapidOCR for lecture slide screenshots.
7. **Resilient LLM Layer** — Rotates up to 6 Gemini API keys with model fallback and exponential backoff to survive rate limits; normalizes multi-part responses to plain strings to prevent agent crashes.
8. **Session Logging** — Every Q&A, viva answer, lab explanation, and study plan is saved to `session_logs/`.

---

## 📁 Folder Structure

```text
vlsi-mentor-ai/
│
├── agents/                     # Domain expert & coordinator agents
│   ├── base_agent.py           # ReAct expert core: RAG context injection, citations, tools
│   ├── expert_agent.py         # ask_expert() dispatcher to all domain experts
│   ├── router_agent.py         # LLM router classifying questions into 14 domains
│   ├── attachment_agent.py     # Handles file/image attachments over the expert flow
│   ├── attachment_context.py   # Shared holder for the current attachment text
│   ├── *_agent.py              # SCAN, ATPG, EDT, MBIST, JTAG, IJTAG, WRAPPER, OCC,
│   │                           # BOUNDARYSCAN, STA, GLS, TCL, LINUX, GENERAL experts
│   ├── interviewer_agent.py    # Mock viva question generator
│   ├── evaluator_agent.py      # Answer evaluation & scoring
│   ├── planner_agent.py        # Personalized study planner
│   ├── lab_explainer_agent.py  # Lab / command explainer
│   └── question_bank_agent.py  # Assessment question bank generator
│
├── core/                       # Core infrastructure
│   ├── llm.py                  # Rotating multi-key Gemini LLM (content normalization + vision)
│   ├── rag_builder.py          # PDF/DOCX/PPTX/TXT extraction, FAISS indexing, retrieval
│   ├── core_topics_ingest.py   # RapidOCR ingest of lecture slides + .srt subtitles
│   ├── file_reader.py          # Text extraction for attached PDF/DOCX/PPTX/TXT files
│   ├── memory.py               # Short-term chat history formatting
│   ├── streaming.py            # Streamed response utilities
│   └── session_logger.py       # Disk logging of all interactions
│
├── knowledge/                  # Knowledge base (committed)
│   ├── 01_Tessent_Shell_User_Manual/
│   ├── 02_Tessent_Shell_Reference_Manual/
│   ├── 03_Scan_and_ATPG_User_Manual/
│   ├── 04_Library_User_Manual/
│   ├── 05_Scan_and_ATPG_Lab_Manual/
│   └── Tessent Atpg Core Topics/   # 900+ lecture screenshots + .srt subtitles
│
├── tests/                      # Unit tests (RAG, agents, graph)
├── ui.py                       # Streamlit web UI (single entry point)
├── _run_ocr.bat                # OCR all Core Topics slides + rebuild index (Windows)
├── _build_idx.bat              # Rebuild FAISS index (Windows)
├── Dockerfile / docker-compose.yml
└── requirements.txt
```

Generated (git-ignored) artifacts land in `knowledge/images/`, `knowledge/ocr_pdfs/`, and `session_logs/`. The FAISS index (`knowledge/vectorstore/`) and extracted text caches (`knowledge/txt/`) are committed so deployments boot instantly.

---

## ⚙️ Setup and Installation

### 1. Prerequisites
- Python 3.13 (3.11+ works; 3.13 matches the Docker image and Streamlit Cloud deploy)
- Tesseract OCR (used by `ocrmypdf` for scanned PDFs)
  - **Windows**: Install via [UB-Mannheim installers](https://github.com/UB-Mannheim/tesseract/wiki) and ensure it is on your PATH.
  - **Linux/Docker**: installed automatically via the Dockerfile / `packages.txt`.

### 2. Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the project root with one or more Gemini keys (up to 6 are rotated):
```env
GEMINI_API_KEY_1=your_key_1
GEMINI_API_KEY_2=your_key_2
...
```
A single `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) also works.

> **Public/Streamlit Cloud:** the same keys work when set in the app's **Secrets**
> panel (`Settings → Secrets`) instead of a local `.env` — `core/llm.py` checks
> env vars, then `st.secrets`. This is how the public deploy supplies backend
> keys for owner-mode inference (see the `toml` example below).

---

## 🗄️ Building the RAG Knowledge Base

Extract text from the reference PDFs / slides and build the FAISS index:
```bash
venv\Scripts\python.exe core\rag_builder.py build
```
- Digital PDFs are parsed directly; scanned PDFs trigger the `ocrmypdf` engine automatically.
- Extracted page text is cached in `knowledge/txt/`.
- The index and chunk metadata are saved to `knowledge/vectorstore/`.

Test retrieval alone:
```bash
venv\Scripts\python.exe core\rag_builder.py test "Why are lockup latches needed?"
```

### Tessent ATPG Core Topics ingest
`core/core_topics_ingest.py` OCRs the 900+ lecture screenshots in `knowledge/Tessent Atpg Core Topics/` (via RapidOCR) and parses the `.srt` subtitles of all lectures into page-structured text files used by the RAG pipeline:

```bash
# OCR all slide screenshots (slow: can take 30-60+ minutes)
venv\Scripts\python.exe -m core.core_topics_ingest --slides

# Subtitles only (fast)
venv\Scripts\python.exe -m core.core_topics_ingest --subtitles

# Quick test with 3 images per topic
venv\Scripts\python.exe -m core.core_topics_ingest --slides --sample 3
```

Or just run `_run_ocr.bat` (Windows), which OCRs the slides and rebuilds the index.

Outputs in `knowledge/txt/`:
- `atpg_video_subs__*.txt` — extracted subtitles per lecture
- `atpg_video_slides__*.txt` — OCR'd slide text per lecture
- `atpg_video_quiz.txt` — combined subtitles + slides + knowledge-check questions

---

## 💻 Running the Application

```bash
venv\Scripts\streamlit run ui.py
```
Open `http://localhost:8501`. Sidebar modes:
- 📤 **Upload Manuals & Labs** — index a new PDF on the fly
- 📖 **Ask Question & Manual Citation** — grounded Q&A across 14 DFT domains (optionally attach a document or image)
- 🧪 **Lab & Command Explainer** — explain Tessent commands / lab scripts
- ❓ **Assessment Question Generator** — module-wise question banks
- 🎤 **Interactive Viva Practice** — graded mock interview
- 📚 **DFT Study Planner** — performance analytics + study plan

### Docker
```bash
docker compose up --build
```

### 🔐 Access Control (password / email gate)

The app ships with a lightweight sign-in gate (`core/auth.py`). Behavior:

| Config | Behavior |
|---|---|
| `APP_PASSWORD` unset/empty | **Open access** (local dev default) |
| `APP_PASSWORD` set, `APP_ALLOWED_EMAILS` empty | Password-only sign-in |
| Both set | User must enter a listed email **and** a valid password |

- `APP_PASSWORD` accepts a comma-separated list, e.g. `APP_PASSWORD=trainer123,guest456` (multiple shared passwords, e.g. trainer vs. trainee).
- `APP_ALLOWED_EMAILS` is a comma-separated allowlist, e.g. `alice@x.com,bob@y.com`. Case-insensitive.
- Session state is per-browser; a **Sign out** button appears in the sidebar.

Set the values in `.env` (local / Docker via `env_file`) or Streamlit secrets (cloud). See `.env.example`.

---

## ☁️ Hosting Options

> **Python version**: local venv, Dockerfile (`python:3.13-slim`), and the recommended
> Streamlit Cloud "Advanced settings → Python version" are all **3.13**. The pinned
> dependencies (numpy 2.3, pandas 3.0, torch 2.12) all publish cp313 wheels (verified).
> `packages.txt` installs tesseract-ocr + ghostscript on Streamlit Cloud for scanned-PDF OCR.

| Option | Auth support | Cost | Notes |
|---|---|---|---|
| **Streamlit Community Cloud** (share.streamlit.io) | ✅ Streamlit's built-in options + this app gate | Free | Deploy straight from the GitHub repo; add `APP_PASSWORD`, `APP_ALLOWED_EMAILS` and Gemini keys in *Settings → Secrets*. Easiest path. |
| **Render / Railway** (Docker) | ✅ This gate | Free tier / ~$5 mo | Deploy the `Dockerfile`; set env vars in dashboard. Container sleeps on free tiers (cold starts). |
| **Fly.io / AWS / GCP / Azure** (Docker) | ✅ This gate | Pay-as-you-go | Full control; best for always-on company use. |
| **Internal company server** (Docker) | ✅ This gate | Hardware only | `docker compose up -d` behind the intranet; works fully offline except Gemini API calls. |

**Recommended for your use case** (small cohort of trainees): push to GitHub → Streamlit Community Cloud → set secrets. The app gate restricts entry, and the repo stays private.

### Streamlit Community Cloud checklist

**Option 1 - public code repo + private data bundle (recommended)**

The repo you deploy from is a **public** repo that contains ONLY code; all
proprietary content stays in a private release asset that the app downloads
at boot. Build steps (run these on your machine):

```bash
# 1. build the private data bundle (dist/knowledge_bundle.zip, ~425 MB)
python tools/pack_knowledge.py

# 2. regenerate the exam bank data from the exam screenshots via Gemini vision
python tools/regen_exam_json.py

# 3. re-pack so the fresh exam data is included
python tools/pack_knowledge.py --force

# 4. export a clean public repo folder (code only, no knowledge data)
python tools/make_public_repo.py --dest C:\path\to\tessent-mentor-public
```

Then:
1. Create the **public** GitHub repo from that folder (`git init`, commit, push).
2. On the **private** repo (`Darokcamper/tessent_ai`): Releases -> Draft a new
   release -> tag `v1` -> upload `dist/knowledge_bundle.zip` as an asset -> Publish.
3. Create a **fine-grained PAT** (read-only, Contents: Read-only, access to only
   the private repo). GitHub -> Settings -> Developer settings -> Fine-grained tokens.
4. Deploy on share.streamlit.io: *New app* -> pick the **public** repo -> `ui.py` ->
   *Advanced settings* -> **Python 3.13**.
5. *Settings -> Secrets* (TOML):
```toml
APP_PASSWORD = "trainer123"
APP_ALLOWED_EMAILS = "alice@x.com,bob@y.com"
GEMINI_API_KEY_1 = "..."
KB_BUNDLE_URL = "https://github.com/YOUR_USER/tessent_ai/releases/download/v1/knowledge_bundle.zip"
KB_BUNDLE_TOKEN = "github_pat_..."
```
6. First boot: the app downloads + unpacks the bundle (~425 MB, 1-3 min),
   then serves everything normally. Subsequent reboots re-download only when
   Streamlit reclaims the container (the restore is idempotent).

**Option 2 - keep everything in the private repo (no extra work, but needs GitHub-App access)**

If the Streamlit GitHub App has been granted access to the private repo
(GitHub -> Settings -> Integrations -> Applications -> Streamlit -> Configure),
the committed index/caches boot instantly and no bundle is needed. The public
repo remains the reliable fallback when this is refused.


---

## 🧪 Testing
```bash
venv\Scripts\python.exe -m unittest discover -s tests
```
`tests/test_agents.py` requires valid API keys; `tests/test_graph.py` skips automatically when the `graphs/` package is not present.

---

## 🛡️ Security & Production Guidelines
- **API Keys**: load secrets via `os.getenv` / Streamlit secrets — never hardcode.
- **Git Ignore**: `.env`, venvs, cached text, vector stores, and logs are excluded from git.