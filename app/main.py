"""Serve the VORTEX demo.

The browser talks only to these same-origin routes. API_KEY is never sent
to the client. Judge-facing endpoints stay on the API service.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.classifier import ClassifierError, classifier_status, classify_texts

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
BATCH_LIMIT = 100
MAX_TICKET_LENGTH = 10000

router = APIRouter()


class PredictRequest(BaseModel):
    text: str = ""


class BatchRequest(BaseModel):
    tickets: list[str] = Field(default_factory=list)


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def _clean_ticket(value: str) -> str:
    return value.strip()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/ui/status")
def ui_status() -> dict:
    return classifier_status()


@router.post("/ui/predict")
def ui_predict(body: PredictRequest):
    text = _clean_ticket(body.text)
    if not text:
        return _error("Enter a support ticket before classifying.", 400)
    if len(text) > MAX_TICKET_LENGTH:
        return _error("That ticket is too long to classify in the demo.", 400)
    try:
        result = classify_texts([text])[0]
    except ClassifierError as exc:
        return _error(exc.message, exc.status_code)
    return result


@router.post("/ui/batch")
def ui_batch(body: BatchRequest):
    tickets = [_clean_ticket(item) for item in body.tickets]
    tickets = [item for item in tickets if item]
    if not tickets:
        return _error(
            "Paste at least one ticket. Put each ticket on its own line.",
            400,
        )
    if len(tickets) > BATCH_LIMIT:
        return _error(
            f"Synchronous batch classification accepts 1 to {BATCH_LIMIT} tickets. "
            f"You submitted {len(tickets)}.",
            400,
        )
    if any(len(item) > MAX_TICKET_LENGTH for item in tickets):
        return _error("One of those tickets is too long to classify in the demo.", 400)
    try:
        results = classify_texts(tickets)
    except ClassifierError as exc:
        return _error(exc.message, exc.status_code)
    return {"results": results}


def create_app() -> FastAPI:
    application = FastAPI(title="VORTEX Ticket Classifier", docs_url=None, redoc_url=None)
    application.include_router(router)

    @application.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND / "index.html")

    # On Vercel the file in public/ is served by the CDN. Locally, FastAPI serves it.
    if not os.environ.get("VERCEL"):
        @application.get("/ambient.mp4")
        def ambient_video() -> FileResponse:
            return FileResponse(ROOT / "public" / "ambient.mp4", media_type="video/mp4")

    @application.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return _error("The request was not valid.", 422)

    @application.exception_handler(Exception)
    async def unhandled_error(_request: Request, exc: Exception) -> JSONResponse:
        logging.getLogger("vortex").exception("Unhandled demo error", exc_info=exc)
        return _error("The classifier is unavailable right now.", 500)

    application.mount(
        "/static",
        StaticFiles(directory=FRONTEND),
        name="static",
    )
    return application


app = create_app()
