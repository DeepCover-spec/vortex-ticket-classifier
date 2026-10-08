"""Async batch jobs, stored in SQLite.

Run one Uvicorn process per container: job state lives in a single SQLite file,
and one background thread does the work so /health and /predict stay free.
If the process restarts, queued and running jobs end as failed with code
`interrupted` instead of disappearing.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from app.config import Settings
from app.predictor import ticket_text
from app.routing import finalize_prediction

logger = logging.getLogger(__name__)

_PURGE_EVERY_S = 600


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_now() -> str:
    return _iso(datetime.now(timezone.utc))


def _expiry(hours: int) -> str:
    return _iso(datetime.now(timezone.utc) + timedelta(hours=hours))


def _status(row: sqlite3.Row) -> dict:
    error = None
    if row["status"] == "failed":
        error = {"code": row["error_code"] or "failed", "message": row["error_message"] or "The job failed."}
    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "total": row["total"],
        "processed": row["processed"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "expires_at": row["expires_at"],
        "model_version": row["model_version"],
        "error": error,
    }


def _expired(row: sqlite3.Row) -> bool:
    return bool(row["expires_at"]) and row["expires_at"] <= utc_now()


class JobStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.RLock()
        settings.jobs_db.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            settings.jobs_db, check_same_thread=False, isolation_level=None, timeout=30
        )
        self.conn.row_factory = sqlite3.Row
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE,
                    status TEXT NOT NULL,
                    total INTEGER NOT NULL,
                    processed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    expires_at TEXT,
                    model_version TEXT NOT NULL,
                    error_code TEXT,
                    error_message TEXT,
                    tickets_json TEXT NOT NULL,
                    predictions_json TEXT
                )
                """
            )

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    @contextmanager
    def _transaction(self):
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            self.conn.execute("COMMIT")

    def _row(self, job_id: str) -> sqlite3.Row | None:
        with self.lock:
            return self.conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()

    def mark_interrupted(self) -> None:
        with self.lock:
            self.conn.execute(
                """
                UPDATE jobs
                SET status='failed', error_code='interrupted',
                    error_message='Service restarted while the job was running.',
                    finished_at=?, expires_at=?
                WHERE status IN ('queued', 'running')
                """,
                (utc_now(), _expiry(self.settings.retention_hours)),
            )

    def create(self, tickets: list[dict], idempotency_key: str | None, model_version: str) -> tuple[str, dict]:
        """Return ("created" | "existing" | "full", status body)."""
        with self._transaction() as conn:
            if idempotency_key:
                row = conn.execute("SELECT * FROM jobs WHERE idempotency_key=?", (idempotency_key,)).fetchone()
                if row is not None and not _expired(row):
                    return "existing", _status(row)
                if row is not None:
                    conn.execute("DELETE FROM jobs WHERE job_id=?", (row["job_id"],))

            active = conn.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')").fetchone()[0]
            if active >= self.settings.max_active_jobs:
                return "full", {}

            job_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO jobs (job_id, idempotency_key, status, total, created_at, model_version, tickets_json)
                VALUES (?, ?, 'queued', ?, ?, ?, ?)
                """,
                (job_id, idempotency_key, len(tickets), utc_now(), model_version, json.dumps(tickets, ensure_ascii=False)),
            )
            return "created", _status(conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())

    def get(self, job_id: str) -> tuple[str, dict | None]:
        """Return ("missing" | "expired" | "ok", status body)."""
        row = self._row(job_id)
        if row is None:
            return "missing", None
        if _expired(row):
            return "expired", None
        return "ok", _status(row)

    def results(self, job_id: str) -> tuple[str, dict | None, list[dict] | None]:
        """Return ("missing" | "expired" | "not_ready" | "ok", status body, predictions)."""
        row = self._row(job_id)
        if row is None:
            return "missing", None, None
        if _expired(row):
            return "expired", None, None
        if row["status"] != "succeeded":
            return "not_ready", _status(row), None
        return "ok", _status(row), json.loads(row["predictions_json"] or "[]")

    def delete(self, job_id: str) -> bool:
        with self.lock:
            return self.conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,)).rowcount > 0

    def claim_next(self) -> tuple[str, list[dict]] | None:
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT job_id, tickets_json FROM jobs WHERE status='queued' ORDER BY created_at, rowid LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE jobs SET status='running', started_at=? WHERE job_id=?", (utc_now(), row["job_id"]))
            return row["job_id"], json.loads(row["tickets_json"])

    def still_running(self, job_id: str) -> bool:
        with self.lock:
            row = self.conn.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return row is not None and row["status"] == "running"

    def save_progress(self, job_id: str, processed: int) -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE jobs SET processed=? WHERE job_id=? AND status='running'", (processed, job_id)
            )

    def finish(self, job_id: str, predictions: list[dict]) -> None:
        with self.lock:
            self.conn.execute(
                """
                UPDATE jobs
                SET status='succeeded', processed=?, predictions_json=?, finished_at=?, expires_at=?
                WHERE job_id=? AND status='running'
                """,
                (
                    len(predictions),
                    json.dumps(predictions, ensure_ascii=False),
                    utc_now(),
                    _expiry(self.settings.retention_hours),
                    job_id,
                ),
            )

    def fail(self, job_id: str, code: str, message: str) -> None:
        with self.lock:
            self.conn.execute(
                """
                UPDATE jobs SET status='failed', error_code=?, error_message=?, finished_at=?, expires_at=?
                WHERE job_id=? AND status IN ('queued', 'running')
                """,
                (code, message, utc_now(), _expiry(self.settings.retention_hours), job_id),
            )

    def purge_expired(self) -> None:
        """Drop the bulky payloads of expired jobs. The row stays so the id answers 410, not 404."""
        with self.lock:
            self.conn.execute(
                """
                UPDATE jobs SET tickets_json='[]', predictions_json=NULL
                WHERE expires_at IS NOT NULL AND expires_at <= ? AND tickets_json != '[]'
                """,
                (utc_now(),),
            )


def run_job(app, job_id: str, tickets: list[dict]) -> None:
    store: JobStore = app.state.store
    model = app.state.model
    predictions: list[dict] = []
    try:
        for start in range(0, len(tickets), store.settings.chunk_size):
            if not store.still_running(job_id):
                return
            chunk = tickets[start : start + store.settings.chunk_size]
            raw = model.predict([ticket_text(t["subject"], t["text"]) for t in chunk])
            if len(raw) != len(chunk):
                raise RuntimeError("model returned a different number of predictions")
            predictions.extend(finalize_prediction(t, r, model.model_version) for t, r in zip(chunk, raw))
            store.save_progress(job_id, len(predictions))
        store.finish(job_id, predictions)
    except Exception:
        logger.exception("job %s failed", job_id)
        store.fail(job_id, "internal_error", "The job failed while classifying tickets.")


def worker_loop(app, stop: threading.Event) -> None:
    store: JobStore = app.state.store
    last_purge = 0.0
    while not stop.is_set():
        try:
            if time.monotonic() - last_purge > _PURGE_EVERY_S:
                store.purge_expired()
                last_purge = time.monotonic()
            claimed = store.claim_next()
        except Exception:
            logger.exception("job worker could not read the job store")
            stop.wait(1)
            continue
        if claimed is None:
            stop.wait(0.05)
            continue
        run_job(app, *claimed)
