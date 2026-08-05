"""
auth.py — server-side authentication for Juiced 2.0.

The current product authenticates only in the browser (Supabase JS); the API trusts anyone.
This module verifies the Supabase-issued JWT on the backend so per-user data (watchlists,
portfolio) can be gated to its owner, and mirrors the identity into our own users table.

Security posture — FAIL CLOSED:
  • With SUPABASE_JWT_SECRET set, tokens are verified (HS256, aud=authenticated, exp).
  • Without it, protected endpoints return 401 — never "allow everyone".
  • A local-only dev escape hatch (JUICED_AUTH_DEV=1 AND no secret) trusts an X-Dev-User
    header so the API is testable offline. It is explicitly insecure and must never be set
    in production; it is ignored the moment a real secret is present.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import Depends, Header, HTTPException, status

import store

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")   # legacy HS256 symmetric secret
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")     # new asymmetric (ES256/RS256) via JWKS
JWT_AUDIENCE = os.getenv("SUPABASE_JWT_AUD", "authenticated")
AUTH_DEV = os.getenv("JUICED_AUTH_DEV", "").lower() in ("1", "true", "yes")

# Auth is "configured" if EITHER a JWKS project URL or a legacy HS256 secret is present.
CONFIGURED = bool(SUPABASE_URL or SUPABASE_JWT_SECRET)

_jwks_client = None


def _jwks():
    """Cached PyJWKClient for the Supabase project's public signing keys."""
    global _jwks_client
    if _jwks_client is None and SUPABASE_URL:
        import jwt
        _jwks_client = jwt.PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json")
    return _jwks_client


class AuthError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail,
                         headers={"WWW-Authenticate": "Bearer"})


def verify_token(token: str) -> dict[str, Any]:
    """
    Verify a Supabase JWT and return its claims. Raises AuthError on any failure.

    Two verification paths, in preference order:
      • JWKS (asymmetric ES256/RS256) when SUPABASE_URL is set — Supabase's current default;
        the public signing key is fetched from the project's JWKS endpoint (no secret needed).
      • HS256 with SUPABASE_JWT_SECRET — the legacy symmetric secret.
    """
    if not CONFIGURED:
        raise AuthError("Authentication is not configured (set SUPABASE_URL or SUPABASE_JWT_SECRET).")
    try:
        import jwt  # PyJWT
    except Exception as exc:  # pragma: no cover
        raise AuthError("Auth library unavailable (pip install pyjwt).") from exc

    try:
        if SUPABASE_URL:
            signing_key = _jwks().get_signing_key_from_jwt(token).key
            claims = jwt.decode(
                token, signing_key, algorithms=["ES256", "RS256"],
                audience=JWT_AUDIENCE, options={"require": ["exp", "sub"]})
        else:
            claims = jwt.decode(
                token, SUPABASE_JWT_SECRET, algorithms=["HS256"],
                audience=JWT_AUDIENCE, options={"require": ["exp", "sub"]})
    except Exception as exc:
        raise AuthError(f"Invalid or expired token: {exc}")
    if not claims.get("sub"):
        raise AuthError("Token missing subject (sub).")
    return claims


def _identity_from_claims(claims: dict[str, Any]) -> dict[str, Any]:
    """Upsert the verified identity into our users table and return the local user row.

    Also syncs `tier` from the JWT's real user_metadata.plan — the SAME field the frontend's
    isPremium() already reads (dashboard.html: `(plan||'lifetime')!=='free'`, the founding-
    member grandfather clause: every signup is stamped 'lifetime' while PAYWALL_LIVE=false).
    Mirrored exactly here so our own tier column (used for the AI Juice daily-limit gate)
    never disagrees with what the rest of the app already calls "Pro" — a missing plan still
    means premium, only an explicit 'free' means FREE, same as the frontend."""
    db = store.get_database()
    users = store.UserRepository(db)
    meta = claims.get("user_metadata") or {}
    email = claims.get("email") or meta.get("email")
    name = meta.get("full_name")
    user = users.upsert(claims["sub"], email, name)
    tier = "FREE" if str(meta.get("plan") or "lifetime").lower() == "free" else "ELITE"
    if user.get("tier") != tier:
        users.set_tier(claims["sub"], tier)
        user["tier"] = tier
    return user


def _bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def get_current_user(
    authorization: Optional[str] = Header(default=None),
    x_dev_user: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    """FastAPI dependency: require a valid identity, else 401. Upserts the user row."""
    # Real verification path.
    token = _bearer(authorization)
    if token:
        return _identity_from_claims(verify_token(token))
    # Local dev escape hatch — only when auth is NOT configured at all.
    if AUTH_DEV and not CONFIGURED and x_dev_user:
        db = store.get_database()
        return store.UserRepository(db).upsert(
            f"dev:{x_dev_user}", f"{x_dev_user}@dev.local", x_dev_user)
    raise AuthError("Missing bearer token.")


def optional_user(
    authorization: Optional[str] = Header(default=None),
    x_dev_user: Optional[str] = Header(default=None),
) -> Optional[dict[str, Any]]:
    """Like get_current_user but returns None instead of raising (for public-with-extras)."""
    try:
        return get_current_user(authorization, x_dev_user)
    except HTTPException:
        return None


def require_role(*roles: str):
    """Dependency factory: require the user to hold one of `roles` (e.g. ADMIN)."""
    def _dep(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        if user.get("role") not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail="Insufficient permissions.")
        return user
    return _dep
