"""Boot-time knowledge-base restore for data-light deployments.

The public code repo ships WITHOUT the private knowledge base. On startup
this module checks for knowledge/vectorstore/faiss_index.index and, if
missing, downloads and unpacks the private bundle from a GitHub release:

  KB_BUNDLE_URL   full URL of the bundle asset on a PRIVATE repo's release
  KB_BUNDLE_TOKEN a GitHub token with read access to that private repo

IMPORTANT: fine-grained PATs require downloading via the GitHub API endpoint
(NOT the browser download URL), because urllib strips the Authorization header
on redirect to GitHub's CDN, and GitHub masks auth failures as 404 for private
repos. This module auto-detects the API endpoint from the release URL.

Everything is optional: if the vars are unset or the download fails, the app
still boots -- RAG retrieval and the Exam Bank degrade gracefully (the code
falls back to plain-text search / empty lists), and Deep Study works only
for the parts whose data is present.

Run standalone for testing:
  python core/bootstrap.py            (uses env/.env/secrets)
  python core/bootstrap.py --check    (report status only, no download)
"""
import io
import json
import os
import re
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

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


def _parse_release_url(kb_url: str):
    """Parse a GitHub release download URL into (owner, repo, tag, asset_name).

    Supports formats:
      https://github.com/{owner}/{repo}/releases/download/{tag}/{asset_name}
      https://github.com/{owner}/{repo}/releases/latest/download/{asset_name}
    """
    parsed = urlparse(kb_url)
    if parsed.netloc != "github.com":
        return None
    parts = parsed.path.strip("/").split("/")
    # Expected: owner, repo, "releases", "download", tag, asset_name...
    if len(parts) < 6:
        return None
    owner, repo = parts[0], parts[1]
    if parts[2] != "releases" or parts[3] != "download":
        return None
    tag = parts[4]
    asset_name = parts[5]
    if tag == "latest":
        tag = _resolve_latest_tag(owner, repo)
    return owner, repo, tag, asset_name


def _resolve_latest_tag(owner: str, repo: str) -> str:
    """Resolve 'latest' to the actual tag name via GitHub API."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    req = urllib.request.Request(api_url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    token = _get("KB_BUNDLE_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("tag_name", "")
    except Exception as exc:
        _log(f"failed to resolve latest tag: {exc}")
        return ""


def _get_asset_id(owner: str, repo: str, tag: str, asset_name: str):
    """Get the asset ID for a release asset via GitHub API."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    req = urllib.request.Request(api_url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    token = _get("KB_BUNDLE_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        for asset in data.get("assets", []):
            if asset.get("name") == asset_name:
                return asset.get("id")
    return None


def _download_via_api(owner: str, repo: str, asset_id: int, token: str):
    """Download a release asset via GitHub API (works with fine-grained PATs).

    The API endpoint serves content directly (no redirect), so the Bearer
    token stays intact.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/assets/{asset_id}"
    req = urllib.request.Request(api_url)
    req.add_header("Accept", "application/octet-stream")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    with urllib.request.urlopen(req, timeout=600) as resp:
        _log(f"API response status: {resp.status}")
        return resp.read()


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
    if token:
        _log(f"token type: {'fine-grained' if token.startswith('github_pat_') else 'classic'} (len={len(token)})")

    # Parse the release URL to get owner, repo, tag, and asset name
    parsed = _parse_release_url(kb_url)
    if not parsed:
        _log(f"ERROR: could not parse release URL: {kb_url}")
        return False

    owner, repo, tag, asset_name = parsed
    _log(f"parsed release: owner={owner}, repo={repo}, tag={tag}, asset={asset_name}")

    # Get the asset ID via the API
    if not token:
        _log("WARNING: no KB_BUNDLE_TOKEN - download will likely 404 on private release")
        return False

    _log("fetching asset ID via GitHub API...")
    asset_id = _get_asset_id(owner, repo, tag, asset_name)
    if not asset_id:
        _log(f"ERROR: asset '{asset_name}' not found in release {tag}")
        _log("app will boot WITHOUT the private knowledge base (RAG/Exam Bank degraded)")
        return False

    _log(f"found asset ID: {asset_id} - downloading via API...")

    try:
        data = _download_via_api(owner, repo, asset_id, token)
    except urllib.error.HTTPError as exc:
        _log(f"FAILED to download bundle: HTTP {exc.code} {exc.reason}")
        _log(f"response headers: {dict(exc.headers)}")
        try:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            _log(f"response body: {body}")
        except:
            pass
        _log("app will boot WITHOUT the private knowledge base (RAG/Exam Bank degraded)")
        return False
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
