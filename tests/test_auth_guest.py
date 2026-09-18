import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unittest.mock import patch, MagicMock
from core.llm import RotatingGeminiLLM
from core.auth import is_owner, owner_emails


class TestStrictGuestMode(unittest.TestCase):
    def test_owner_emails_supports_both_env_names(self):
        with patch.dict("os.environ", {"APP_OWNER_EMAILS": "admin1@wipro.com, admin2@gmail.com"}):
            emails = owner_emails()
            self.assertIn("admin1@wipro.com", emails)
            self.assertIn("admin2@gmail.com", emails)

    def test_guest_without_key_raises_error(self):
        """Guests who do not supply guest_api_key must be blocked from backend rotation keys."""
        llm = RotatingGeminiLLM()
        with patch("core.auth.is_guest", return_value=True):
            with patch("streamlit.session_state", {}):
                with self.assertRaises(ValueError) as ctx:
                    llm.get_client()
                self.assertIn("Guest Mode requires", str(ctx.exception))

    def test_guest_with_key_succeeds(self):
        """Guests who provide guest_api_key are allowed with their own key."""
        llm = RotatingGeminiLLM()
        with patch("core.auth.is_guest", return_value=True):
            with patch("streamlit.session_state", {"guest_api_key": "AIzaSyFakeKeyForGuest12345"}):
                client = llm.get_client()
                self.assertEqual(client.google_api_key.get_secret_value(), "AIzaSyFakeKeyForGuest12345")

    def test_owner_uses_backend_keys(self):
        """Owner sessions are allowed to use rotated backend keys."""
        llm = RotatingGeminiLLM()
        with patch("core.auth.is_guest", return_value=False):
            with patch("core.llm.get_gemini_keys", return_value=["backend_key_1", "backend_key_2"]):
                client = llm.get_client()
                self.assertEqual(client.google_api_key.get_secret_value(), "backend_key_1")


if __name__ == "__main__":
    unittest.main()
