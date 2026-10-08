"""API key check. Runs before the body is read, so a bad key never reaches validation."""

from __future__ import annotations

import hmac

from fastapi import Request
from fastapi.responses import JSONResponse

from app.errors import error_response

_CHALLENGE = {"WWW-Authenticate": "Bearer"}


def _presented_keys(request: Request) -> list[str]:
    keys: list[str] = []
    header = request.headers.get("x-api-key")
    if header:
        keys.append(header)
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        keys.append(token.strip())
    return keys


def authorize(request: Request) -> JSONResponse | None:
    """Return a 401 response, or None when the caller may continue.

    An unset or empty API_KEY rejects everyone instead of running open.
    """
    presented = _presented_keys(request)
    if not presented:
        return error_response(
            request,
            401,
            "unauthorized",
            "Missing API key. Send X-API-Key or Authorization Bearer.",
            headers=_CHALLENGE,
        )
    expected = request.app.state.settings.api_key
    if expected:
        wanted = expected.encode("utf-8")
        if any(hmac.compare_digest(key.encode("utf-8"), wanted) for key in presented):
            return None
    return error_response(request, 401, "unauthorized", "Invalid API key.", headers=_CHALLENGE)
