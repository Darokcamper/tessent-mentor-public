# Auth Setup — Tessent Mentor AI

The sign-in gate is **layered and auto-detected**. You can enable one layer or
combine them. Configure everything in **Streamlit Cloud → App → Settings →
Secrets** (same format works in a local `.streamlit/secrets.toml` or `.env`).

| Layer | Enable with | What users see |
|---|---|---|
| 1. Google sign-in (OIDC) | `[auth]` section in secrets | "Sign in with Google" button; real Google identity |
| 2. Email + password | `APP_PASSWORD` in secrets | email + password form |
| 2b. TOTP 2FA | `APP_TOTP_SECRET` / `APP_TOTP_SECRETS` | extra 6-digit code field on the password form |
| 3. Open access | *nothing set* ⚠️ | app is **open to anyone with the URL** (local dev default) |

When **both** `[auth]` (Google) and `APP_PASSWORD` are set, the gate shows a
method chooser — users pick **Google** *or* **Email + Password (+2FA)**, so the
2FA path is never hidden behind the Google button. If only `[auth]` is set,
Google is the sole method. If the OIDC redirect cannot start (e.g. wrong
redirect URI) and a password is configured, the gate falls back to the
password form automatically.

---

## Layer 1 — Google sign-in (recommended)

Requires Streamlit ≥ 1.42 — `requirements.txt` pins `streamlit==1.58.0` ✅.

### 1. Create a Google OAuth client

1. Go to <https://console.cloud.google.com/apis/credentials> (any Google account).
2. **Create Credentials → OAuth client ID → Web application**.
3. Add **Authorized redirect URIs**:
   - Production: `https://YOUR-APP-SUBDOMAIN.streamlit.app/oauth2callback`
     (the exact subdomain of your Streamlit Cloud app URL, then `/oauth2callback`)
   - Local testing: `http://localhost:8501/oauth2callback`
4. Copy the **Client ID** (`...apps.googleusercontent.com`) and **Client secret** (`GOCSPX-...`).

### 2. Generate a cookie secret

```powershell
venv\Scripts\python.exe tools\gen_auth_secrets.py
```

### 3. Paste into Streamlit secrets

```toml
[auth]
redirect_uri = "https://YOUR-APP-SUBDOMAIN.streamlit.app/oauth2callback"
cookie_secret = "paste-generated-value"
client_id = "....apps.googleusercontent.com"
client_secret = "GOCSPX-...."
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

APP_ALLOWED_EMAILS = "you@gmail.com, friend@gmail.com"
```

`APP_ALLOWED_EMAILS` also restricts **Google** sign-in to listed Gmail
addresses — without it, any Google account that finds the URL could sign in.

Restart the app. The sidebar badge shows `👤 you@gmail.com (google)`.

---

## Layer 2 — Email + password

### Shared password (current behavior)

```toml
APP_PASSWORD = "assess2026"
APP_ALLOWED_EMAILS = ""   # empty = any email works as an identity tag
```

### Per-user passwords (recommended over shared)

Give one password per person; revoke someone by removing their entry:

```toml
APP_PASSWORD = "hazar-Mk42, friend-Pw17, teammate-Xy9"
APP_ALLOWED_EMAILS = "hazar@x.com, friend@x.com, teammate@x.com"
```

---

## Layer 2b — TOTP 2FA (authenticator app codes)

Add `pyotp` (already in `requirements.txt`). Generate the base32 secret with
`tools\gen_auth_secrets.py`, then:

### Everyone must supply a 2FA code

```toml
APP_PASSWORD = "assess2026"
APP_TOTP_SECRET = "BASE32SECRET"
```

Each user adds the secret to Google Authenticator / Authy via **manual entry**
(type the base32 secret). Codes rotate every 30 s.

### 2FA only for specific users

```toml
APP_PASSWORD = "hazar-Mk42, friend-Pw17"
APP_TOTP_SECRETS = "hazar@x.com=BASE32SECRET1, friend@x.com=BASE32SECRET2"
```

Users **without** an entry (and with no global `APP_TOTP_SECRET`) skip 2FA.

---

## Combos

- Google + password + 2FA together: `[auth]` + `APP_PASSWORD` +
  `APP_TOTP_SECRETS` → users choose Google **or** email+password+2FA at the gate.
- Google + allowlist: `[auth]` + `APP_ALLOWED_EMAILS` → real Gmail identity,
  only listed people. (2FA is unnecessary here — Google enforces its own 2FA.)
- Password + 2FA + per-user passwords: `APP_PASSWORD` (list) + `APP_TOTP_SECRETS`.
- Local dev: leave everything unset → open access.

## Security notes & limits

- Passwords are compared with `hmac.compare_digest` but stored **in plaintext
  in secrets** — use unique, long passwords; rotate after the assessment week.
- There is **no brute-force lockout** — keep the URL private.
- Auth is UI-level gating (protects model usage); there is no per-user data
  isolation (chat history is per browser session).
- On Streamlit Cloud, secrets are encrypted at rest and never leave the container.
