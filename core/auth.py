"""Simple authentication gate for the Tessent Mentor AI app.

Config (env vars via .env, or Streamlit secrets):
  APP_PASSWORD        single shared password (or comma-separated list)
  APP_ALLOWED_EMAILS  comma-separated allowlist; empty/omitted = password alone
                      is enough (the email field acts as a simple identity tag)

Call require_auth() at the very top of ui.py, before any other st.* widgets.
"""
import os
import hmac
import streamlit as st


def _get(name: str, default: str = "") -> str:
    """Fetch a config value from env vars, then Streamlit secrets."""
    val = os.getenv(name, "")
    if val:
        return val
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def _passwords() -> list:
    raw = _get("APP_PASSWORD") or _get("APP_PASSWORDS", "")
    return [p.strip() for p in raw.split(",") if p.strip()]


def _allowed_emails() -> list:
    raw = _get("APP_ALLOWED_EMAILS", "")
    return [e.strip().lower() for e in raw.split(",") if e.strip()]


def is_authenticated() -> bool:
    return bool(st.session_state.get("auth_ok", False))


def current_user() -> str:
    return st.session_state.get("auth_user", "")


def logout():
    st.session_state.auth_ok = False
    st.session_state.auth_user = ""


def require_auth() -> bool:
    """Block rendering until the user passes the gate. Returns True when authed.

    Behavior:
      - No APP_PASSWORD configured -> app open (auth disabled), returns True.
      - Password required. If APP_ALLOWED_EMAILS is set, the entered email must
        be on the list AND the password must match.
      - Session persists per browser tab until logout/refresh clears state.
    """
    passwords = _passwords()
    if not passwords:
        return True  # auth not configured -> open access (local dev default)

    if is_authenticated():
        return True

    allowed = _allowed_emails()
    st.markdown(
        """
        <div style="text-align:center; padding-top:8vh;">
          <h1>🧠 Tessent Mentor AI</h1>
          <p style="color:#888;">Private, manual-grounded DFT study assistant.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("auth_form", clear_on_submit=False):
        email = st.text_input(
            "Email" + (" (must be on the allowlist)" if allowed else " (optional ID)"),
            key="auth_email_input",
        )
        pwd = st.text_input("Password", type="password", key="auth_pwd_input")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if submitted:
        pwd_ok = any(hmac.compare_digest(pwd.strip(), p) for p in passwords)
        email_clean = (email or "").strip().lower()
        email_ok = (not allowed) or (email_clean in allowed)
        if pwd_ok and email_ok:
            st.session_state.auth_ok = True
            st.session_state.auth_user = email_clean or "user"
            st.rerun()
        else:
            if not pwd_ok:
                st.error("Incorrect password.")
            if not email_ok:
                st.error("This email is not on the access list.")
    return False


def auth_badge_sidebar():
    """Optional tiny user badge + logout in the sidebar (call inside st.sidebar)."""
    if not is_authenticated():
        return
    user = current_user()
    with st.container(horizontal=True, border=False):
        st.caption(f"👤 {user}")
        if st.button("Sign out", key="auth_signout"):
            logout()
            st.rerun()