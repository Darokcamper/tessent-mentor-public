"""Prints ready-to-paste auth secrets for Tessent Mentor AI.

Usage:
    venv\\Scripts\\python.exe tools\\gen_auth_secrets.py

Prints:
  - An OIDC cookie_secret for the [auth] section (Streamlit Google sign-in).
  - A base32 TOTP secret for APP_TOTP_SECRET / APP_TOTP_SECRETS (2FA).
Each run generates fresh values; only the ones you actually paste matter.
"""
import base64
import secrets

print('[auth] cookie_secret  = "%s"' % secrets.token_urlsafe(32))
print('APP_TOTP_SECRET (b32) = "%s"'
      % base64.b32encode(secrets.token_bytes(20)).decode())
