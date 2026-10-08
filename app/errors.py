"""ErrorResponse bodies in the exact shape of spec/error_response.schema.json."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

MAX_HEADER_ID = 128


def with_request_id(request: Request, headers: dict[str, str] | None = None) -> dict[str, str]:
    merged = dict(headers or {})
    value = request.headers.get("x-request-id")
    if value and len(value) <= MAX_HEADER_ID:
        merged["X-Request-ID"] = value
    return merged


def error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: list[dict] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    error: dict = {"code": code, "message": message}
    if details:
        error["details"] = details
    return JSONResponse(
        status_code=status_code,
        content={"error": error},
        headers=with_request_id(request, headers),
    )


def validation_error(request: Request, details: list[dict]) -> JSONResponse:
    return error_response(request, 422, "validation_error", "Request failed validation.", details)
