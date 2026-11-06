import os
import time
import base64
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from core.model_routing import (
    BudgetExceeded,
    capability_for_agent,
    compute_backoff,
    is_rate_limit_error,
    parse_retry_after,
    record_call,
    resolve_model,
    response_tokens,
    retry_hint_from_exception,
    session_budget,
)

load_dotenv()


def _secret(name: str, default: str = "") -> str:
    """Fetch a config value from env vars, then Streamlit secrets (mirrors core/auth.py).

    Searches both top-level secrets AND any nested sections (such as when variables
    are placed after [auth] in secrets.toml).
    """
    val = os.getenv(name, "")
    if val:
        return val
    try:
        import streamlit as st
        secrets = st.secrets
        if secrets is None:
            return default
        # 1. Direct top-level check
        try:
            if name in secrets:
                v = secrets[name]
                if not isinstance(v, (dict, list)):
                    return str(v)
        except Exception:
            pass
        # 2. Case-insensitive top-level check
        try:
            for k in secrets:
                if str(k).lower() == name.lower():
                    v = secrets[k]
                    if not isinstance(v, (dict, list)):
                        return str(v)
        except Exception:
            pass
        # 3. Search nested sections (e.g. if variables were written under [auth])
        try:
            for k in secrets:
                sub = secrets[k]
                if isinstance(sub, dict) or hasattr(sub, "items"):
                    try:
                        if name in sub:
                            v = sub[name]
                            if not isinstance(v, (dict, list)):
                                return str(v)
                        for sk in sub:
                            if str(sk).lower() == name.lower():
                                v = sub[sk]
                                if not isinstance(v, (dict, list)):
                                    return str(v)
                    except Exception:
                        pass
        except Exception:
            pass
        # 4. Fallback: .get() method
        try:
            val = secrets.get(name)
            if val is not None and not isinstance(val, (dict, list)):
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
    if not keys:
        # Fallback: dynamically scan st.secrets for any Gemini/Google API key
        try:
            import streamlit as st
            secrets = st.secrets
            if secrets is not None:
                for k in secrets:
                    k_upper = str(k).upper()
                    if "GEMINI" in k_upper or ("GOOGLE" in k_upper and "KEY" in k_upper):
                        val = str(secrets[k]).strip()
                        if val and val not in keys:
                            keys.append(val)
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
    
    IMPORTANT: We only cache a NON-EMPTY result. If no keys are found (e.g.
    because st.secrets was not yet accessible at import time), we leave the
    cache as None so that the next call retries — this ensures that once the
    Streamlit session is fully initialized and secrets are available, the keys
    are picked up automatically without needing an app restart.
    """
    global _GEMINI_KEYS_CACHE
    if _GEMINI_KEYS_CACHE:  # Only trust a non-empty cached result
        return _GEMINI_KEYS_CACHE
    keys = _gemini_keys()
    if keys:
        _GEMINI_KEYS_CACHE = keys  # Cache only on success
    return keys or []


# Back-compat constant alias. NOTE: plain module constants bind at IMPORT time,
# which breaks on Streamlit Cloud because st.secrets is not populated yet at
# that point. Accessing the value goes through get_gemini_keys() at RUNTIME
# instead, so keep this alias EMPTY and never fill it.
GEMINI_KEYS: list = []

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

    def get_client(self, custom_key: str = None, capability: str = "answer_synthesis"):
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
            # Enforce Strict Guest Mode (BYOK): guests MUST supply their own API key
            try:
                from core.auth import is_guest
                if is_guest():
                    raise ValueError(
                        "Guest Mode requires your own free Gemini API key to ask questions. "
                        "Please enter your key in the sidebar (get a free key at https://aistudio.google.com/app/apikey)."
                    )
            except ImportError:
                pass

            keys = get_gemini_keys()
            if keys:
                key = keys[self.key_idx % len(keys)]
            else:
                raise ValueError(
                    "No Gemini API key available. Please enter your free Gemini API key in the sidebar "
                    "(get one at https://aistudio.google.com/app/apikey)."
                )

        # WP-12: the model is chosen per capability (routing table), with the existing
        # fallback chain still driven by model_idx.
        model = resolve_model(capability, self.model_idx)
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=key,
            temperature=0.3
        )

    def invoke(self, *args, capability: str = "answer_synthesis", **kwargs):
        """Invoke the model for *capability* with budget, rotation and 429 handling."""
        retries = 0
        backoff = self.initial_backoff
        budget = session_budget()

        while True:
            budget.check()
            try:
                client = self.get_client(capability=capability)
                raw_response = client.invoke(*args, **kwargs)
                # Read usage before normalisation: the normaliser may replace the
                # provider object with a TextResponse that has no usage metadata.
                tokens = response_tokens(raw_response)
                # Normalize content to prevent "'list' object has no attribute 'strip'" crashes
                if hasattr(raw_response, 'content'):
                    raw_response.content = _normalize_content(raw_response.content)
                else:
                    raw_response = TextResponse(_normalize_content(raw_response))
                budget.record(tokens or 0)
                record_call(capability, resolve_model(capability, self.model_idx), "ok", tokens)
                return raw_response
            except BudgetExceeded:
                raise
            except Exception as e:
                if isinstance(e, ValueError) and "Guest Mode requires" in str(e):
                    raise e
                retries, backoff = self._rotate_for_retry(retries, backoff, e, capability)
                if retries > self.max_retries:
                    raise e
                msg = f"Retrying with {resolve_model(capability, self.model_idx)} in {backoff:.1f}s..."
                print(msg)
                try:
                    st.toast(msg, icon="🔄")
                except Exception:
                    pass
                time.sleep(backoff)

    def _rotate_for_retry(self, retries: int, backoff: float, error: Exception, capability: str):
        """Rotate the key/model, honour Retry-After, and return (retries, next_backoff)."""
        retries += 1
        keys = get_gemini_keys()
        if keys:
            self.key_idx = (self.key_idx + 1) % len(keys)
            if retries % len(keys) == 0:
                self.model_idx = (self.model_idx + 1) % len(MODELS)
        else:
            self.model_idx = (self.model_idx + 1) % len(MODELS)

        # WP-R1: read the provider's hint from whatever the exception really exposes
        # (response headers, `.details` JSON body, rendered message, or the cause chain).
        # `getattr(error, "response", None)` alone returned None for every real SDK error.
        retry_after = retry_hint_from_exception(error) if is_rate_limit_error(error) else None
        next_backoff = compute_backoff(
            retries,
            base=self.initial_backoff,
            factor=1.2,
            retry_after=retry_after,
        )
        record_call(
            capability,
            resolve_model(capability, self.model_idx),
            "rate_limited" if retry_after is not None else "error",
            retry_after=retry_after,
        )
        return retries, next_backoff

    def stream(self, *args, capability: str = "answer_synthesis", **kwargs):
        retries = 0
        backoff = self.initial_backoff

        while True:
            session_budget().check()
            try:
                client = self.get_client(capability=capability)
                record_call(capability, resolve_model(capability, self.model_idx), "stream_open")
                return client.stream(*args, **kwargs)
            except BudgetExceeded:
                raise
            except Exception as e:
                if isinstance(e, ValueError) and "Guest Mode requires" in str(e):
                    raise e
                retries, backoff = self._rotate_for_retry(retries, backoff, e, capability)
                if retries > self.max_retries:
                    raise e
                time.sleep(backoff)

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
            session_budget().check()
            try:
                client = self.get_client(capability="vision")
                raw_response = client.invoke(messages)
                if hasattr(raw_response, "content"):
                    raw_response.content = _normalize_content(raw_response.content)
                else:
                    raw_response = TextResponse(_normalize_content(raw_response))
                tokens = response_tokens(raw_response)
                session_budget().record(tokens or 0)
                record_call("attachment_vision", resolve_model("vision", self.model_idx), "ok", tokens)
                return raw_response
            except BudgetExceeded:
                raise
            except Exception as e:
                if isinstance(e, ValueError) and "Guest Mode requires" in str(e):
                    raise e
                retries, backoff = self._rotate_for_retry(retries, backoff, e, "attachment_vision")
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