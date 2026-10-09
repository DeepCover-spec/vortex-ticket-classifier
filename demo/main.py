"""VORTEX demo server.

The browser only talks to this process. Classification is proxied to the hosted
API. API_KEY is read from the environment and is never written to a response,
a log line, or the page.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

DEFAULT_API_BASE = "https://vortex-ticket-classifier.up.railway.app"
CHANNELS = {"email", "chat", "call_transcript"}
PREDICTION_FIELDS = (
    "ticket_id",
    "category",
    "secondary_category",
    "team",
    "is_urgent",
    "confidence",
    "model_version",
    "needs_human_review",
)
UNAVAILABLE = "VORTEX couldn't connect to the classification service. Please try again."
NOT_CONFIGURED = "VORTEX couldn't reach the classification service. Please try again later."
UNREADABLE = "The classifier returned a response the demo could not read."
TIMEOUT = httpx.Timeout(60.0)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


class DemoError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message


def asset_dirs() -> tuple[Path, Path]:
    """Repo-root assets are the source of truth.

    A Vercel project whose root is this folder cannot see the repo root at
    runtime, so the same files are also kept beside this module.
    """
    local_front = HERE / "frontend"
    repo_front = REPO / "frontend"
    if os.environ.get("VERCEL") and (local_front / "index.html").is_file():
        return local_front, HERE / "public"
    if (repo_front / "index.html").is_file():
        return repo_front, REPO / "public"
    return local_front, HERE / "public"


FRONTEND_DIR, PUBLIC_DIR = asset_dirs()

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def api_base() -> str:
    if "API_BASE_URL" in os.environ:
        return os.environ["API_BASE_URL"].strip().rstrip("/")
    return DEFAULT_API_BASE


def api_key() -> str:
    return os.environ.get("API_KEY", "").strip()


def new_ticket_id() -> str:
    return f"vx-{uuid.uuid4().hex[:20]}"


def supplied_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    ticket_id = value.strip()
    if not ticket_id or len(ticket_id) > 64:
        return None
    return ticket_id


def public_message(status: int, payload: object) -> str:
    message = ""
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            message = error["message"].strip()
        elif isinstance(error, str):
            message = error.strip()
    lowered = message.lower()
    secret = api_key()
    leaked = bool(secret) and secret in message
    unsafe = (
        not message
        or len(message) > 300
        or leaked
        or any(token in lowered for token in ("traceback", "api_key", "x-api-key", "exception"))
        or "<" in message
    )
    if unsafe:
        if status in (400, 422):
            return "The request was not valid."
        if status in (401, 403):
            return "The classifier refused the request."
        return UNAVAILABLE
    return message


def require_config() -> None:
    if not api_base() or not api_key():
        raise DemoError(503, NOT_CONFIGURED)


def clean_channel(value: object) -> str:
    if value is None or value == "":
        return "chat"
    if not isinstance(value, str) or value not in CHANNELS:
        raise DemoError(400, "Choose a channel: chat, email, or call transcript.")
    return value


def clean_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DemoError(400, "Enter a support ticket before classifying.")
    text = value.strip()
    if len(text) > 10000:
        raise DemoError(400, "A ticket can be at most 10000 characters.")
    return text


def clean_subject(value: object) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise DemoError(400, "The subject is not valid.")
    subject = value.strip()
    if not subject:
        return None
    if len(subject) > 500:
        raise DemoError(400, "A subject can be at most 500 characters.")
    return subject


def build_ticket(text: str, channel: str, subject: str | None, ticket_id: str | None = None) -> dict:
    ticket = {
        "ticket_id": ticket_id or new_ticket_id(),
        "channel": channel,
        "text": text,
    }
    if subject:
        ticket["subject"] = subject
    return ticket


def public_prediction(item: object, ticket: dict) -> dict:
    if not isinstance(item, dict) or not isinstance(item.get("category"), str) or not item.get("category"):
        raise DemoError(502, UNREADABLE)
    prediction = {field: item[field] for field in PREDICTION_FIELDS if field in item}
    prediction["ticket_id"] = prediction.get("ticket_id") or ticket["ticket_id"]
    prediction["text"] = ticket["text"]
    prediction["ticket"] = ticket["text"]
    return prediction


async def call_api(method: str, path: str, payload: dict | None = None) -> tuple[int, object]:
    require_config()
    headers = {"X-API-Key": api_key()}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=False) as client:
            response = await client.request(method, f"{api_base()}{path}", json=payload, headers=headers)
    except httpx.HTTPError:
        raise DemoError(503, UNAVAILABLE) from None
    try:
        body = response.json()
    except ValueError:
        body = None
    return response.status_code, body


def error_response(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


@app.exception_handler(DemoError)
async def handle_demo_error(_request, exc: DemoError) -> JSONResponse:
    return error_response(exc.status, exc.message)


@app.exception_handler(Exception)
async def handle_unexpected(_request, _exc: Exception) -> JSONResponse:
    return error_response(500, UNAVAILABLE)


async def hosted_health() -> dict:
    base = api_base()
    if not base:
        return {"status": "unavailable", "model_version": None}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0), follow_redirects=False) as client:
            response = await client.get(f"{base}/health")
        body = response.json()
    except (httpx.HTTPError, ValueError):
        return {"status": "unavailable", "model_version": None}
    if not isinstance(body, dict) or body.get("status") not in {"ok", "loading"}:
        return {"status": "unavailable", "model_version": None}
    version = body.get("model_version")
    if not isinstance(version, str):
        version = None
    return {"status": body["status"], "model_version": version}


@app.get("/health")
async def health() -> dict:
    return await hosted_health()


@app.get("/ui/status")
async def ui_status() -> dict:
    data = await hosted_health()
    data["classifier"] = "ready" if data.get("status") == "ok" else "unavailable"
    return data


async def read_body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        raise DemoError(400, "The request was not valid.") from None
    if not isinstance(body, dict):
        raise DemoError(400, "The request was not valid.")
    return body


@app.post("/ui/predict")
async def ui_predict(request: Request) -> JSONResponse:
    body = await read_body(request)
    channel = clean_channel(body.get("channel"))
    text = clean_text(body.get("text"))
    subject = clean_subject(body.get("subject")) if channel == "email" else None
    ticket = build_ticket(text, channel, subject, supplied_id(body.get("ticket_id")))
    status, payload = await call_api("POST", "/predict", ticket)
    if status != 200:
        raise DemoError(status if status in {400, 401, 403, 422} else 503, public_message(status, payload))
    return JSONResponse(public_prediction(payload, ticket))


@app.post("/ui/batch")
async def ui_batch(request: Request) -> JSONResponse:
    body = await read_body(request)
    if not isinstance(body.get("tickets"), list):
        raise DemoError(400, "Paste at least one ticket. Put each ticket on its own line.")
    raw_tickets = body["tickets"]
    if not raw_tickets:
        raise DemoError(400, "Paste at least one ticket. Put each ticket on its own line.")
    if len(raw_tickets) > 100:
        raise DemoError(400, f"Synchronous batch classification accepts 1 to 100 tickets. You have {len(raw_tickets)}.")
    channel = clean_channel(body.get("channel"))
    subject = clean_subject(body.get("subject")) if channel == "email" else None
    tickets = []
    seen = set()
    for item in raw_tickets:
        if isinstance(item, str):
            text = clean_text(item)
            ticket_channel = channel
            ticket_subject = subject
            ticket_id = None
        elif isinstance(item, dict):
            text = clean_text(item.get("text"))
            ticket_channel = clean_channel(item.get("channel", channel))
            ticket_subject = clean_subject(item.get("subject")) if ticket_channel == "email" else None
            ticket_id = supplied_id(item.get("ticket_id"))
        else:
            raise DemoError(400, "The request was not valid.")
        ticket = build_ticket(text, ticket_channel, ticket_subject, ticket_id)
        if ticket["ticket_id"] in seen:
            raise DemoError(400, "Each ticket needs its own id.")
        seen.add(ticket["ticket_id"])
        tickets.append(ticket)
    status, payload = await call_api("POST", "/predict/batch", {"tickets": tickets})
    if status != 200:
        raise DemoError(status if status in {400, 401, 403, 422} else 503, public_message(status, payload))
    predictions = payload.get("predictions") if isinstance(payload, dict) else None
    if not isinstance(predictions, list) or len(predictions) != len(tickets):
        raise DemoError(502, UNREADABLE)
    rows = [public_prediction(item, ticket) for item, ticket in zip(predictions, tickets)]
    return JSONResponse({"predictions": rows})


@app.get("/")
def home() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/ambient.mp4")
def ambient_video() -> FileResponse:
    return FileResponse(PUBLIC_DIR / "ambient.mp4", media_type="video/mp4")
