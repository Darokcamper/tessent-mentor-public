import os
import time
import base64
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def _secret(name: str, default: str = "") -> str:
    """Fetch a config value from env vars, then Streamlit secrets (mirrors core/auth.py).

    In Streamlit Cloud, st.secrets may not be populated until after auth, or may
    not exist at all. We use direct indexing (most reliable) with fallback to .get().
    """
    val = os.getenv(name, "")
    if val:
        return val
    try:
        import streamlit as st
        secrets = st.secrets
        if secrets is None:
            return default
        # Try direct indexing first (most reliable across Streamlit versions)
        try:
            if isinstance(secrets, dict):
                if name in secrets:
                    return str(secrets[name])
            else:
                # Streamlit Secrets object - try __contains__ then __getitem__
                if name in secrets:
                    return str(secrets[name])
        except Exception:
            pass
        # Fallback to .get() method
        try:
            val = secrets.get(name)
            if val is not None:
                return str(val)
        except Exception:
            pass
        return default
    except Exception:
        return default


def _gemini_keys() -> list:
    """Collect Gemini keys from env/.env, then from Streamlit secrets.

    Priority: GEMINI_API_KEY_1..6 (rotated), then a single GEMINI_API_KEY or
    GOOGLE_API_KEY. Reading Streamlit secrets means owner-mode backend keys set
    in the Streamlit Cloud Secrets panel also work on public deployments, not
    just keys supplied via a local .env file.
    """
    keys = [_secret(f"GEMINI_API_KEY_{i}") for i in range(1, 7)]
    keys = [k for k in keys if k]
    if not keys:
        single = _secret("GEMINI_API_KEY") or _secret("GOOGLE_API_KEY")
        if single:
            keys = [single]
    # Debug: log how many keys we found (without exposing the keys themselves)
    try:
        import streamlit as st
        st.toast(f"LLM: {len(keys)} Gemini key(s) loaded (env/secrets/BYOK)", icon="🔑")
    except Exception:
        pass
    return keys


# Lazy-loaded Gemini keys cache. In Streamlit Cloud, st.secrets may not be
# populated at import time, so we defer loading until first access.
_GEMINI_KEYS_CACHE = None

def get_gemini_keys() -> list:
    """Lazily load and cache Gemini keys from env/.env or Streamlit secrets.
    
    This avoids import-time st.secrets access which can fail in Streamlit Cloud
    before the secrets panel is fully initialized.
    """
    global _GEMINI_KEYS_CACHE
    if _GEMINI_KEYS_CACHE is None:
        _GEMINI_KEYS_CACHE = _gemini_keys()
        if not _GEMINI_KEYS_CACHE:
            _GEMINI_KEYS_CACHE = []
    return _GEMINI_KEYS_CACHE

# Legacy alias for backward compatibility - code that imports GEMINI_KEYS directly
# will trigger lazy loading via __getattr__ on first access.
def __getattr__(name):
    if name == "GEMINI_KEYS":
        return get_gemini_keys()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

class RotatingGeminiLLM:
    """
    Production-grade LLM wrapper that rotates through 6 Gemini API keys and model fallbacks
    to prevent rate limits (429/TPD/RPM) and connection interruptions.
    Also supports Bring-Your-Own-Key (BYOK) for guest users.
    """
    def __init__(self, max_retries: int = 6, initial_backoff: float = 2.0):
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.key_idx = 0
        self.model_idx = 0

    def get_client(self, custom_key: str = None):
        # Check for user-provided key (BYOK)
        if not custom_key:
            try:
                import streamlit as st
                custom_key = str(st.session_state.get("guest_api_key", "")).strip()
            except Exception:
                custom_key = ""

        if custom_key:
            key = custom_key
        else:
            keys = get_gemini_keys()
            if keys:
                key = keys[self.key_idx % len(keys)]
            else:
                raise ValueError(
                    "No Gemini API key available. Please enter your free Gemini API key in the sidebar "
                    "(get one at https://aistudio.google.com/app/apikey)."
                )

        model = MODELS[self.model_idx % len(MODELS)]
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=key,
            temperature=0.3
        )

    def invoke(self, *args, **kwargs):
        retries = 0
        backoff = self.initial_backoff
        
        while True:
            try:
                client = self.get_client()
                raw_response = client.invoke(*args, **kwargs)
                # Normalize content to prevent "'list' object has no attribute 'strip'" crashes
                if hasattr(raw_response, 'content'):
                    raw_response.content = _normalize_content(raw_response.content)
                else:
                    raw_response = TextResponse(_normalize_content(raw_response))
                return raw_response
            except Exception as e:
                retries += 1
                # Rotate key and fallback model on error or rate limit
                if GEMINI_KEYS:
                    self.key_idx = (self.key_idx + 1) % len(GEMINI_KEYS)
                    if retries % len(GEMINI_KEYS) == 0:
                        self.model_idx = (self.model_idx + 1) % len(MODELS)
                else:
                    self.model_idx = (self.model_idx + 1) % len(MODELS)
                    
                if retries > self.max_retries:
                    raise e
                    
                msg = f"Retrying with {MODELS[self.model_idx]} in {backoff:.1f}s..."
                print(msg)
                try:
                    st.toast(msg, icon="🔄")
                except Exception:
                    pass
                time.sleep(backoff)
                backoff *= 1.2

    def stream(self, *args, **kwargs):
        retries = 0
        backoff = self.initial_backoff
        
        while True:
            try:
                client = self.get_client()
                return client.stream(*args, **kwargs)
            except Exception as e:
                retries += 1
                if GEMINI_KEYS:
                    self.key_idx = (self.key_idx + 1) % len(GEMINI_KEYS)
                    if retries % len(GEMINI_KEYS) == 0:
                        self.model_idx = (self.model_idx + 1) % len(MODELS)
                else:
                    self.model_idx = (self.model_idx + 1) % len(MODELS)
                    
                if retries > self.max_retries:
                    raise e
                    
                time.sleep(backoff)
                backoff *= 1.2

    def invoke_vision(self, prompt: str, image_bytes: bytes, image_mime: str = "image/png"):
        """Sends a text prompt together with an image to the multimodal Gemini model.

        Uses LangChain message content objects so ChatGoogleGenerativeAI can encode
        the image. Applies the same key rotation / model fallback / content
        normalization as invoke(). Returns an object with a .content plain string.
        """
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": "data:%s;base64,%s" % (image_mime, _b64(image_bytes))},
            },
        ]
        messages = [("user", content)]

        retries = 0
        backoff = self.initial_backoff
        while True:
            try:
                client = self.get_client()
                raw_response = client.invoke(messages)
                if hasattr(raw_response, "content"):
                    raw_response.content = _normalize_content(raw_response.content)
                else:
                    raw_response = TextResponse(_normalize_content(raw_response))
                return raw_response
            except Exception as e:
                retries += 1
                if GEMINI_KEYS:
                    self.key_idx = (self.key_idx + 1) % len(GEMINI_KEYS)
                    if retries % len(GEMINI_KEYS) == 0:
                        self.model_idx = (self.model_idx + 1) % len(MODELS)
                else:
                    self.model_idx = (self.model_idx + 1) % len(MODELS)
                if retries > self.max_retries:
                    raise e
                try:
                    st.toast(f"Rotating API Key for vision (retry {retries})...", icon="🔄")
                except Exception:
                    pass
                time.sleep(backoff)
                backoff *= 1.2

def _b64(data: bytes) -> str:
    """Return a base64-encoded string (utf-8) of raw bytes for inline data URLs."""
    return base64.b64encode(data).decode("utf-8")


# Export wrapped multi-key rotating LLM
class TextResponse:
    """
    Thin wrapper that normalizes the LangChain AIMessage content to a plain string.

    Gemini 2.5 Flash sometimes returns multi-part content (thinking blocks + text blocks),
    in which case response.content is a list of dicts. This wrapper ensures .content is
    always a string, preventing "'list' object has no attribute 'strip'" crashes in agents.
    """
    def __init__(self, content: str):
        self._content = content

    @property
    def content(self) -> str:
        return self._content

    def __str__(self) -> str:
        return self._content


def _normalize_content(content) -> str:
    """
    Normalize LLM response content to a plain string.
    
    - If str: return as-is
    - If list: extract and join text fields from dicts and plain strings
    - Otherwise: str() fallback
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content) if content is not None else ""


# Export wrapped multi-key rotating LLM
llm = RotatingGeminiLLM()