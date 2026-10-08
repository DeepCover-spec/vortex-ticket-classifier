"""FastAPI service for spec/tensorforge-phase2-openapi-v2.yaml (v2.1.0)."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import predictor
from app.auth import authorize
from app.config import Settings
from app.errors import MAX_HEADER_ID, error_response, validation_error, with_request_id
from app.jobs import JobStore, worker_loop
from app.routing import finalize_prediction
from app.validation import (
    BATCH_MAX_BYTES,
    JOB_MAX_BYTES,
    PREDICT_MAX_BYTES,
    check_content_type,
    normalize_ticket,
    parse_json,
    read_limited,
    validate_single,
    validate_tickets,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

POLL_AFTER_S = "3"
QUEUE_FULL_RETRY_S = "30"


def create_app(settings: Settings | None = None, model=None) -> FastAPI:
    """Settings, model and job store are resolved at startup, not import time."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved = settings or Settings.from_env()
        if not resolved.api_key:
            logger.warning("API_KEY is not set: every protected endpoint will answer 401")
        app.state.settings = resolved
        app.state.model = model or predictor.load(resolved.model_dir)
        app.state.store = JobStore(resolved)
        app.state.store.mark_interrupted()
        stop = threading.Event()
        worker = threading.Thread(target=worker_loop, args=(app, stop), name="batch-jobs", daemon=True)
        worker.start()
        try:
            yield
        finally:
            stop.set()
            worker.join(timeout=5)
            app.state.store.close()

    app = FastAPI(
        title="TensorForge 2.0 Phase 2 - Support Ticket Classification API",
        version="2.1.0",
        lifespan=lifespan,
        redirect_slashes=False,
    )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404:
            return error_response(request, 404, "not_found", "No resource at this path.")
        if exc.status_code == 405:
            return error_response(
                request, 405, "method_not_allowed", "Method not allowed for this path.",
                headers=dict(exc.headers or {}),
            )
        return error_response(request, exc.status_code, "http_error", "Request failed.")

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return error_response(request, 500, "internal_error", "The service could not complete the request.")

    async def classify(request: Request, tickets: list[dict]) -> list[dict]:
        model = request.app.state.model
        raw = await run_in_threadpool(
            model.predict, [t["text"] for t in tickets], [t["subject"] for t in tickets]
        )
        if len(raw) != len(tickets):
            raise RuntimeError("model returned a different number of predictions")
        return [finalize_prediction(t, r, model.model_version) for t, r in zip(tickets, raw)]

    async def read_body(request: Request, limit: int):
        """auth -> 415 -> 413 -> 400. Returns (payload, error_response)."""
        for check in (authorize, _bad_request_id, check_content_type):
            failed = check(request)
            if failed:
                return None, failed
        body, too_big = await read_limited(request, limit)
        if too_big:
            return None, too_big
        return parse_json(request, body)

    @app.get("/health")
    async def health(request: Request):
        body = {"status": "ok", "model_version": request.app.state.model.model_version, "model_loaded": True}
        return JSONResponse(body, headers=with_request_id(request))

    @app.post("/predict")
    async def predict(request: Request):
        payload, failed = await read_body(request, PREDICT_MAX_BYTES)
        if failed:
            return failed
        failed = validate_single(request, payload)
        if failed:
            return failed
        [prediction] = await classify(request, [normalize_ticket(payload)])
        return JSONResponse(prediction, headers=with_request_id(request))

    @app.post("/predict/batch")
    async def predict_batch(request: Request):
        payload, failed = await read_body(request, BATCH_MAX_BYTES)
        if failed:
            return failed
        tickets, failed = validate_tickets(request, payload, max_items=100)
        if failed:
            return failed
        started = time.perf_counter()
        predictions = await classify(request, [normalize_ticket(t) for t in tickets])
        meta = {
            "count": len(predictions),
            "model_version": request.app.state.model.model_version,
            "processing_time_ms": int((time.perf_counter() - started) * 1000),
        }
        return JSONResponse({"predictions": predictions, "meta": meta}, headers=with_request_id(request))

    @app.post("/batch/jobs")
    async def submit_job(request: Request):
        payload, failed = await read_body(request, JOB_MAX_BYTES)
        if failed:
            return failed
        tickets, failed = validate_tickets(request, payload, max_items=5000)
        if failed:
            return failed
        key = (request.headers.get("idempotency-key") or "").strip() or None
        if key and len(key) > MAX_HEADER_ID:
            return validation_error(request, [{"field": "Idempotency-Key", "issue": "must be at most 128 characters"}])

        kind, status = request.app.state.store.create(
            [normalize_ticket(t) for t in tickets], key, request.app.state.model.model_version
        )
        if kind == "full":
            return error_response(
                request, 429, "too_many_jobs", "Job queue is full. Retry later.",
                headers={"Retry-After": QUEUE_FULL_RETRY_S},
            )
        headers = {"Location": f"/batch/jobs/{status['job_id']}", "Retry-After": POLL_AFTER_S}
        return JSONResponse(status, status_code=202, headers=with_request_id(request, headers))

    @app.get("/batch/jobs/{job_id}")
    async def job_status(job_id: str, request: Request):
        failed = _job_guard(request, job_id)
        if failed:
            return failed
        kind, status = request.app.state.store.get(job_id)
        if kind != "ok":
            return _job_gone(request, kind)
        headers = {"Retry-After": POLL_AFTER_S} if status["status"] in ("queued", "running") else {}
        return JSONResponse(status, headers=with_request_id(request, headers))

    @app.get("/batch/jobs/{job_id}/results")
    async def job_results(job_id: str, request: Request):
        failed = _job_guard(request, job_id)
        if failed:
            return failed
        kind, status, predictions = request.app.state.store.results(job_id)
        if kind == "not_ready":
            return error_response(
                request, 409, "job_not_ready",
                f"Job status is '{status['status']}'. Results are available once it is 'succeeded'.",
            )
        if kind != "ok":
            return _job_gone(request, kind)
        window, failed = _page(request, len(predictions))
        if failed:
            return failed
        offset, limit, next_offset = window
        body = {
            "job_id": status["job_id"],
            "status": "succeeded",
            "total": status["total"],
            "offset": offset,
            "limit": limit,
            "next_offset": next_offset,
            "model_version": status["model_version"],
            "predictions": predictions[offset : offset + limit],
        }
        return JSONResponse(body, headers=with_request_id(request))

    @app.delete("/batch/jobs/{job_id}")
    async def delete_job(job_id: str, request: Request):
        failed = _job_guard(request, job_id)
        if failed:
            return failed
        if not request.app.state.store.delete(job_id):
            return _job_gone(request, "missing")
        return Response(status_code=204, headers=with_request_id(request))

    return app


def _bad_request_id(request: Request):
    value = request.headers.get("x-request-id")
    if value is not None and len(value) > MAX_HEADER_ID:
        return validation_error(request, [{"field": "X-Request-ID", "issue": "must be at most 128 characters"}])
    return None


def _job_guard(request: Request, job_id: str):
    failed = authorize(request) or _bad_request_id(request)
    if failed:
        return failed
    if len(job_id) > MAX_HEADER_ID:
        return validation_error(request, [{"field": "job_id", "issue": "must be at most 128 characters"}])
    return None


def _job_gone(request: Request, kind: str):
    if kind == "expired":
        return error_response(request, 410, "job_expired", "Job results have expired.")
    return error_response(request, 404, "job_not_found", "No job with this id.")


def _page(request: Request, total: int):
    """Validate offset/limit. Default is the whole result from offset 0."""
    raw_offset = request.query_params.get("offset", "0")
    raw_limit = request.query_params.get("limit")
    issues = []
    if not raw_offset.isdigit():
        issues.append({"field": "offset", "issue": "must be an integer of 0 or more"})
    if raw_limit is not None and not (raw_limit.isdigit() and 1 <= int(raw_limit) <= 5000):
        issues.append({"field": "limit", "issue": "must be an integer from 1 to 5000"})
    if issues:
        return None, validation_error(request, issues)
    offset = int(raw_offset)
    limit = int(raw_limit) if raw_limit is not None else max(total - offset, 1)
    next_offset = offset + limit if offset + limit < total else None
    return (offset, limit, next_offset), None


app = create_app()
