"""Request checks behind the spec's 415 / 413 / 400 / 422 responses.

Order for every prediction and job submission:
auth (401), content type (415), size (413), JSON parse (400), field rules (422).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.requests import ClientDisconnect

from app.errors import error_response, validation_error

MIB = 1024 * 1024
PREDICT_MAX_BYTES = 1 * MIB
BATCH_MAX_BYTES = 5 * MIB
JOB_MAX_BYTES = 25 * MIB
CHANNELS = ("email", "chat", "call_transcript")


def check_content_type(request: Request) -> JSONResponse | None:
    media = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media == "application/json":
        return None
    return error_response(
        request, 415, "unsupported_media_type", "Content-Type must be application/json."
    )


def _too_large(request: Request) -> JSONResponse:
    return error_response(
        request, 413, "payload_too_large", "Request body exceeds the maximum allowed size."
    )


# Answering 413 while the client is still uploading makes the server close a
# socket with unread data, which resets the connection before the client can
# read the response. Oversized bodies are therefore read and discarded, up to
# this multiple of the limit, before the 413 is sent.
DRAIN_FACTOR = 4


async def _discard(chunks: AsyncIterator[bytes], budget: int) -> None:
    try:
        async for chunk in chunks:
            budget -= len(chunk)
            if budget <= 0:
                return
    except ClientDisconnect:
        return


async def read_limited(request: Request, limit: int) -> tuple[bytes | None, JSONResponse | None]:
    drain_cap = limit * DRAIN_FACTOR
    stream = request.stream()
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        if int(declared) <= drain_cap:
            await _discard(stream, int(declared))
        return None, _too_large(request)
    chunks: list[bytes] = []
    size = 0
    async for chunk in stream:
        size += len(chunk)
        if size > limit:
            await _discard(stream, drain_cap - size)
            return None, _too_large(request)
        chunks.append(chunk)
    return b"".join(chunks), None


def parse_json(request: Request, body: bytes) -> tuple[object, JSONResponse | None]:
    try:
        if not body.strip():
            raise ValueError("empty body")
        return json.loads(body), None
    except (ValueError, UnicodeDecodeError, RecursionError):
        return None, error_response(request, 400, "malformed_json", "Request body is not valid JSON.")


def _detail(field: str, issue: str, index: int | None = None) -> dict:
    item: dict = {} if index is None else {"index": index}
    item["field"] = field
    item["issue"] = issue
    return item


def _ticket_issues(item: object, index: int | None, require_id: bool) -> list[dict]:
    if not isinstance(item, dict):
        return [_detail("ticket" if index is not None else "body", "must be an object", index)]

    issues: list[dict] = []
    ticket_id = item.get("ticket_id")
    if ticket_id is None:
        if require_id:
            issues.append(_detail("ticket_id", "is required", index))
    elif not isinstance(ticket_id, str):
        issues.append(_detail("ticket_id", "must be a string", index))
    elif len(ticket_id) > 64:
        issues.append(_detail("ticket_id", "must be at most 64 characters", index))

    channel = item.get("channel")
    if channel is None:
        issues.append(_detail("channel", "is required", index))
    elif not isinstance(channel, str) or channel not in CHANNELS:
        issues.append(_detail("channel", "must be one of email, chat, call_transcript", index))

    subject = item.get("subject")
    if subject is not None:
        if not isinstance(subject, str):
            issues.append(_detail("subject", "must be a string", index))
        elif len(subject) > 500:
            issues.append(_detail("subject", "must be at most 500 characters", index))

    text = item.get("text")
    if text is None:
        issues.append(_detail("text", "is required", index))
    elif not isinstance(text, str):
        issues.append(_detail("text", "must be a string", index))
    elif len(text) > 10000:
        issues.append(_detail("text", "must be at most 10000 characters", index))
    elif text.isspace() or not text:
        issues.append(_detail("text", "must contain at least one non-whitespace character", index))
    return issues


def validate_single(request: Request, payload: object) -> JSONResponse | None:
    issues = _ticket_issues(payload, index=None, require_id=False)
    return validation_error(request, issues) if issues else None


def validate_tickets(
    request: Request, payload: object, max_items: int
) -> tuple[list[dict] | None, JSONResponse | None]:
    if not isinstance(payload, dict):
        return None, validation_error(request, [_detail("body", "must be a JSON object")])
    tickets = payload.get("tickets")
    if tickets is None:
        return None, validation_error(request, [_detail("tickets", "is required")])
    if not isinstance(tickets, list):
        return None, validation_error(request, [_detail("tickets", "must be an array")])
    if not 1 <= len(tickets) <= max_items:
        return None, validation_error(
            request, [_detail("tickets", f"must contain between 1 and {max_items} items")]
        )

    issues: list[dict] = []
    seen: set[str] = set()
    for index, item in enumerate(tickets):
        issues.extend(_ticket_issues(item, index, require_id=True))
        ticket_id = item.get("ticket_id") if isinstance(item, dict) else None
        if isinstance(ticket_id, str):
            if ticket_id in seen:
                issues.append(_detail("ticket_id", "duplicate of an earlier item", index))
            seen.add(ticket_id)
    if issues:
        return None, validation_error(request, issues)
    return tickets, None


def normalize_ticket(item: dict) -> dict:
    """Keep only the fields the model may see, plus the id to echo back."""
    ticket_id = item.get("ticket_id")
    subject = item.get("subject")
    return {
        "ticket_id": ticket_id if isinstance(ticket_id, str) else None,
        "channel": item["channel"],
        "subject": subject if isinstance(subject, str) else "",
        "text": item["text"],
    }
