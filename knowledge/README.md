# knowledge/

This directory is intentionally EMPTY in the public repo.

The private knowledge base (FAISS index, OCR'd manuals, lesson screenshots,
exam data) is fetched at first boot from a private GitHub release:

- set `KB_BUNDLE_URL` + `KB_BUNDLE_TOKEN` in Streamlit secrets (see .env.example)
- see `core/bootstrap.py` for the restore logic

To build the bundle locally: `python tools/pack_knowledge.py`
