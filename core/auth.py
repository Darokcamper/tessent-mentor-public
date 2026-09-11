"""Authentication gate for the Tessent Mentor AI app.

Layers (auto-detected, in this order):
  1. GOOGLE SIGN-IN (OIDC)    - active when an [auth] section exists in Streamlit
                                secrets (Streamlit >= 1.42 native auth; repo pins 1.58).
                                Restrict who may sign in with APP_ALLOWED_EMAILS.
  2. EMAIL + PASSWORD (+ 2FA) - active when APP_PASSWORD (or APP_PASSWORDS) is set.
                                Comma-separated list = one password per person
                                (revoke one by removing it). Optional TOTP 2FA:
                                APP_TOTP_SECRET (everyone) or APP_TOTP_SECRETS
                                ("email=BASE32" per user; users without an entry
                                skip 2FA unless a global secret is also set).
  3. OPEN ACCESS              - nothing configured -> app runs open (local dev
                                default). The sidebar badge warns about this.

Config (env vars via .env, or Streamlit secrets) - see AUTH_SETUP.md:
  APP_PASSWORD        shared password, or comma-separated per-user passwords
  APP_ALLOWED_EMAILS  comma-separated allowlist; empty = password alone is enough
                      (the email field acts as a self-reported identity tag)
  APP_TOTP_SECRET     optional global base32 TOTP secret -> 2FA for everyone
  APP_TOTP_SECRETS    optional per-user map "email=BASE32,email2=BASE32"
  [auth]              Streamlit native OIDC config for Google - see AUTH_SETUP.md

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


def _totp_secrets() -> dict:
    """Parse APP_TOTP_SECRETS = "email=BASE32,email2=BASE32" into a dict."""
    raw = _get("APP_TOTP_SECRETS", "")
    out = {}
    for pair in raw.split(","):
        if "=" not in pair:
            continue
        email, secret = pair.split("=", 1)
        email = email.strip().lower()
        secret = secret.strip()
        if email and secret:
            out[email] = secret
    return out


def _oidc_enabled() -> bool:
    """True when Streamlit native auth is configured ([auth] in secrets)."""
    try:
        return bool(st.secrets.get("auth", None))
    except Exception:
        return False


def _oidc_logged_in() -> bool:
    try:
        return bool(st.user.is_logged_in)
    except Exception:
        return False


def _oidc_email() -> str:
    try:
        return (st.user.email or "").strip().lower()
    except Exception:
        return ""


def _totp_required() -> bool:
    return bool(_get("APP_TOTP_SECRET", "") or _totp_secrets())


def _verify_totp(email: str, code: str) -> bool:
    """Verify the 6-digit TOTP code for this user.

    Uses the per-user secret from APP_TOTP_SECRETS if present, else the global
    APP_TOTP_SECRET. If neither applies to this user, 2FA is skipped (True).
    """
    secret = _totp_secrets().get(email) or _get("APP_TOTP_SECRET", "")
    if not secret:
        return True  # no TOTP configured for this user
    if not code or not code.strip():
        return False
    try:
        import pyotp  # lazy: only needed when 2FA is actually enabled
        return pyotp.TOTP(secret).verify(code.strip().replace(" ", ""), valid_window=1)
    except Exception:
        return False


def is_authenticated() -> bool:
    if st.session_state.get("auth_ok", False):
        return True
    # An OIDC session survives page reloads via cookie; re-derive from st.user.
    if _oidc_enabled() and _oidc_logged_in():
        st.session_state.auth_ok = True
        st.session_state.auth_user = _oidc_email() or "google-user"
        return True
    return False


def current_user() -> str:
    user = st.session_state.get("auth_user", "")
    if user:
        return user
    if _oidc_logged_in():
        return _oidc_email() or "google-user"
    return "user"


def logout():
    """End the session (and the Google/OIDC session when present)."""
    st.session_state.auth_ok = False
    st.session_state.auth_user = ""
    if _oidc_enabled():
        try:
            st.logout()
        except Exception:
            pass


def _render_hero():
    st.markdown(
        """
        <div style="text-align:center; padding-top:8vh;">
          <h1>🧠 Tessent Mentor AI</h1>
          <p style="color:#888;">Private, manual-grounded DFT study assistant.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _password_form():
    """Email + password (+ optional 2FA code) sign-in. Returns True when authed."""
    allowed = _allowed_emails()
    with st.form("auth_form", clear_on_submit=False):
        email = st.text_input(
            "Email" + (" (must be on the allowlist)" if allowed else " (optional ID)"),
            key="auth_email_input",
        )
        pwd = st.text_input("Password", type="password", key="auth_pwd_input")
        code = ""
        if _totp_required():
            code = st.text_input(
                "2FA code (6 digits from your authenticator app)",
                key="auth_totp_input",
            )
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if not submitted:
        return False

    email_clean = (email or "").strip().lower()
    pwd_ok = any(hmac.compare_digest((pwd or "").strip(), p) for p in _passwords())
    email_ok = (not allowed) or (email_clean in allowed)
    totp_ok = _verify_totp(email_clean, code or "")

    if not pwd_ok:
        st.error("Incorrect password.")
    if not email_ok:
        st.error("This email is not on the access list.")
    if not totp_ok:
        st.error("Invalid or missing 2FA code.")
    if pwd_ok and email_ok and totp_ok:
        st.session_state.auth_ok = True
        st.session_state.auth_user = email_clean or "user"
        st.rerun()
    return False


def require_auth() -> bool:
    """Block rendering until the user passes the gate. Returns True when authed.

    Behavior:
      - [auth] in secrets  -> Google sign-in (OIDC); password form as fallback
                              if the OIDC redirect cannot be initiated.
      - APP_PASSWORD set   -> email + password (+ 2FA) form.
      - APP_ALLOWED_EMAILS -> entered/Google email must be on the list.
      - Neither configured -> app open (auth disabled, local dev default).
    """
    if is_authenticated():
        return True

    passwords = _passwords()
    oidc = _oidc_enabled()

    if not passwords and not oidc:
        return True  # auth not configured -> open access (local dev default)

    _render_hero()

    # ---- Layer 1: Google sign-in via Streamlit native OIDC ----
    if oidc:
        if _oidc_logged_in():
            email = _oidc_email()
            allowed = _allowed_emails()
            if allowed and email not in allowed:
                st.error(
                    f"{email or 'Your Google account'} is not on the access list. "
                    "Ask the administrator to add it to APP_ALLOWED_EMAILS."
                )
                if st.button("Sign out of Google", key="auth_google_deny"):
                    logout()
                    st.rerun()
                return False
            st.session_state.auth_ok = True
            st.session_state.auth_user = email or "google-user"
            st.rerun()
        else:
            st.info("Sign in with your Google account to continue.")
            try:
                st.login()
            except Exception as e:
                # e.g. missing redirect_uri / cookie_secret in [auth]
                st.error(f"Google sign-in could not start: {e}")
                if not passwords:
                    return False
                st.warning("Falling back to password sign-in.")
                st.markdown("---")
                return _password_form()
            return False  # redirect initiated; Streamlit takes over

    # ---- Layer 2: email + password (+ optional 2FA) ----
    if not passwords:
        return False
    return _password_form()


def auth_badge_sidebar():
    """Tiny user badge + logout in the sidebar (call inside st.sidebar)."""
    if not is_authenticated():
        return
    user = current_user()
    mode = "google" if (_oidc_enabled() and _oidc_logged_in()) else "password"
    with st.container(horizontal=True, border=False):
        st.caption(f"👤 {user} ({mode})")
        if st.button("Sign out", key="auth_signout"):
            logout()
            st.rerun()

