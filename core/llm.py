import os
import time
import base64
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Gather all 6 rotated Gemini keys from .env
GEMINI_KEYS = [os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 7) if os.getenv(f"GEMINI_API_KEY_{i}")]
if not GEMINI_KEYS:
    single_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if single_key:
        GEMINI_KEYS = [single_key]

if not GEMINI_KEYS:
    # In public or BYOK deployments, GEMINI_KEYS may be empty if the host does not provide server keys.
    GEMINI_KEYS = []

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
        elif GEMINI_KEYS:
            key = GEMINI_KEYS[self.key_idx % len(GEMINI_KEYS)]
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