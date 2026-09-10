"""Boot-time knowledge-base restore for data-light deployments.

The public code repo ships WITHOUT the private knowledge base. On startup
this module checks for knowledge/vectorstore/faiss_index.index and, if
missing, downloads and unpacks the private bundle from a GitHub release:

  KB_BUNDLE_URL   full URL of the bundle asset on a PRIVATE repo's release
  KB_BUNDLE_TOKEN a GitHub token with read access to that private repo

Everything is optional: if the vars are unset or the download fails, the app
still boots -- RAG retrieval and the Exam Bank degrade gracefully (the code
falls back to plain-text search / empty lists), and Deep Study works only
for the parts whose data is present.

Run standalone for testing:
  python core/bootstrap.py            (uses env/.env/secrets)
  python core/bootstrap.py --check    (report status only, no download)
"""
import io
import os
import sys
import zipfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "knowledge"
SENTINEL = KB_DIR / "vectorstore" / "faiss_index.index"
EXAM_FILE = PROJECT_ROOT / "_exam_parsed.json"
MARKER = KB_DIR / ".bootstrap_done"


def _get(name: str, default: str = "") -> str:
    """Fetch config from env vars, then Streamlit secrets (mirrors core/auth.py)."""
    val = os.getenv(name, "")
    if val:
        return val
    try:
        import streamlit as st

        return str(st.secrets.get(name, default))
    except Exception:
        return default


def _log(msg: str) -> None:
    print(f"[bootstrap] {msg}", flush=True)


def restore_done() -> bool:
    """True when the private knowledge base is already on disk."""
    return SENTINEL.exists() and EXAM_FILE.exists()


def restore_knowledge() -> bool:
    """Download + unpack the private bundle. Returns True when data is ready.

    - No KB_BUNDLE_URL configured  -> skip silently (local dev has data already).
    - Already restored (or marker) -> skip.
    - Download/extract failure     -> log loudly, return False; app still boots.
    """
    # Marker short-circuit: a completed restore must never re-download on the
    # next boot, even if some optional files (e.g. exam json) were not in the
    # bundle and restore_done() therefore reports "not fully ready".
    if MARKER.exists() or restore_done():
        return True
    kb_url = _get("KB_BUNDLE_URL")
    if not kb_url:
        _log("KB_BUNDLE_URL not set - skipping private knowledge restore")
        return False

    token = _get("KB_BUNDLE_TOKEN")
    _log(f"downloading bundle: {kb_url}")
    try:
        req = urllib.request.Request(kb_url)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
            req.add_header("Accept", "application/octet-stream")
            _log("token present (Bearer) - authenticated download")
        else:
            _log("WARNING: no KB_BUNDLE_TOKEN - download will likely 404 on private release")
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = resp.read()
    except Exception as exc:
        _log(f"FAILED to download bundle: {exc}")
        _log("app will boot WITHOUT the private knowledge base (RAG/Exam Bank degraded)")
        return False

    _log(f"downloaded {len(data) / 1e6:.1f} MB - extracting to {KB_DIR}")
    try:
        _extract_bundle(data)
    except Exception as exc:
        _log(f"FAILED to extract bundle: {exc}")
        return False

    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text("restored\n", encoding="utf-8")
    _log(f"restore complete (ready={restore_done()})")
    return restore_done()


def _extract_bundle(data: bytes) -> None:
    """Validate and unpack the bundle zip into knowledge/ (+ repo root files)."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt member in bundle: {bad}")
        names = zf.namelist()
        total = sum(i.file_size for i in zf.infolist())
        _log(f"unpacking {len(names)} entries ({total / 1e6:.1f} MB)")
        for info in zf.infolist():
            if info.is_dir():
                top = Path(info.filename).parts[:1]
                if top == ("knowledge",):
                    (KB_DIR / ".keep").parent.mkdir(parents=True, exist_ok=True)
                continue
            # strip a single top-level 'knowledge/' prefix if present
            parts = Path(info.filename).parts
            if parts and parts[0] == "knowledge":
                parts = parts[1:]
            if not parts:
                continue
            rel = Path(*parts)
            if ".." in rel.parts:
                raise RuntimeError(f"unsafe path in bundle: {info.filename}")
            dest = KB_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(info))
        # _exam_parsed.json lives at bundle root, not under knowledge/
        root_files = [
            i for i in zf.infolist()
            if not i.is_dir()
            and len(Path(i.filename).parts) == 1
            and Path(i.filename).parts[0] != "knowledge"
        ]
        for info in root_files:
            dest = PROJECT_ROOT / Path(info.filename).name
            dest.write_bytes(zf.read(info))
            _log(f"restored root file: {dest.name}")


def _from_env_file(path: Path) -> dict:
    """Minimal .env parser for standalone runs (no python-dotenv import)."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    return out


if __name__ == "__main__":
    check_only = "--check" in sys.argv
    env = _from_env_file(PROJECT_ROOT / ".env")
    for key in ("KB_BUNDLE_URL", "KB_BUNDLE_TOKEN"):
        os.environ.setdefault(key, env.get(key, ""))

    if check_only:
        print(f"KB_BUNDLE_URL set  : {bool(os.getenv('KB_BUNDLE_URL'))}")
        print(f"KB_BUNDLE_TOKEN set: {bool(os.getenv('KB_BUNDLE_TOKEN'))}")
        print(f"faiss index present: {SENTINEL.exists()}")
        print(f"exam json present  : {EXAM_FILE.exists()}")
        print(f"restore done       : {restore_done()}")
        sys.exit(0)

    ok = restore_knowledge()
    print(f"[bootstrap] final status: {'READY' if ok else 'DEGRADED'}")
