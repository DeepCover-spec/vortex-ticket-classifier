"""Backend cases the black-box contract suite cannot force: a full queue, results
requested mid-job, restart recovery, retention expiry, a 5,000-ticket job and a
real joblib model. In-process only."""
import os
import threading
import time

import pytest

if os.environ.get("TF_BASE_URL"):
    pytest.skip("in-process backend tests; skipped when testing a live server", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.jobs import JobStore  # noqa: E402
from app.main import create_app  # noqa: E402
from app.predictor import StubModel, load  # noqa: E402
from tests.conftest import assert_valid, make_tickets  # noqa: E402

KEY = "internal-test-key"
AUTH = {"X-API-Key": KEY}


class HeldModel(StubModel):
    """Blocks inside predict() until released, so a job stays 'running'."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def predict(self, texts, subjects=None):
        self.started.set()
        self.release.wait(10)
        return super().predict(texts, subjects)


def build(tmp_path, model=None, key=KEY):
    settings = Settings(api_key=key, jobs_db=tmp_path / "jobs.sqlite", model_dir=tmp_path)
    return create_app(settings, model=model)


@pytest.fixture
def held(tmp_path):
    model = HeldModel()
    with TestClient(build(tmp_path, model)) as client:
        yield client, model
        model.release.set()


def wait_status(client, job_id, wanted, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/batch/jobs/{job_id}", headers=AUTH).json()
        if body["status"] == wanted:
            return body
        time.sleep(0.05)
    pytest.fail(f"job {job_id} never reached {wanted}")


def test_results_while_running_are_409(held, validators):
    client, model = held
    job = client.post("/batch/jobs", json={"tickets": make_tickets(3)}, headers=AUTH).json()
    assert model.started.wait(5)
    status = client.get(f"/batch/jobs/{job['job_id']}", headers=AUTH)
    assert status.json()["status"] == "running"
    assert status.headers["retry-after"] == "3"
    early = client.get(f"/batch/jobs/{job['job_id']}/results", headers=AUTH)
    assert early.status_code == 409
    assert early.json()["error"]["code"] == "job_not_ready"
    assert_valid(validators["error_response"], early.json())


def test_fifth_active_job_is_429(held):
    client, model = held
    assert client.post("/batch/jobs", json={"tickets": make_tickets(1, "A")}, headers=AUTH).status_code == 202
    assert model.started.wait(5)
    for prefix in "BCD":
        assert client.post("/batch/jobs", json={"tickets": make_tickets(1, prefix)}, headers=AUTH).status_code == 202
    full = client.post("/batch/jobs", json={"tickets": make_tickets(1, "E")}, headers=AUTH)
    assert full.status_code == 429
    assert full.headers["retry-after"] == "30"
    assert full.json()["error"]["code"] == "too_many_jobs"


def test_delete_cancels_a_running_job(held):
    client, model = held
    job = client.post("/batch/jobs", json={"tickets": make_tickets(600)}, headers=AUTH).json()
    assert model.started.wait(5)
    assert client.delete(f"/batch/jobs/{job['job_id']}", headers=AUTH).status_code == 204
    model.release.set()
    time.sleep(0.2)
    assert client.get(f"/batch/jobs/{job['job_id']}", headers=AUTH).status_code == 404


def test_restart_marks_unfinished_jobs_interrupted(tmp_path, validators):
    # Rows left behind by a process that died mid-job.
    store = JobStore(Settings(api_key=KEY, jobs_db=tmp_path / "jobs.sqlite", model_dir=tmp_path))
    _, running = store.create(make_tickets(2, "R"), None, "stub-0.1")
    _, queued = store.create(make_tickets(2, "Q"), None, "stub-0.1")
    store.claim_next()
    store.close()

    with TestClient(build(tmp_path)) as client:
        for job in (running, queued):
            body = client.get(f"/batch/jobs/{job['job_id']}", headers=AUTH).json()
            assert body["status"] == "failed"
            assert body["error"]["code"] == "interrupted"
            assert body["finished_at"] and body["expires_at"]
            assert_valid(validators["batch_job_status"], body)


def test_expired_job_is_410(tmp_path):
    with TestClient(build(tmp_path)) as client:
        job = client.post("/batch/jobs", json={"tickets": make_tickets(2)}, headers=AUTH).json()
        wait_status(client, job["job_id"], "succeeded")
        client.app.state.store.conn.execute(
            "UPDATE jobs SET expires_at='2000-01-01T00:00:00Z' WHERE job_id=?", (job["job_id"],)
        )
        client.app.state.store.purge_expired()
        for path in (f"/batch/jobs/{job['job_id']}", f"/batch/jobs/{job['job_id']}/results"):
            response = client.get(path, headers=AUTH)
            assert response.status_code == 410
            assert response.json()["error"]["code"] == "job_expired"


def test_five_thousand_ticket_job(tmp_path, validators):
    tickets = make_tickets(5000, "F")
    with TestClient(build(tmp_path)) as client:
        started = time.perf_counter()
        created = client.post("/batch/jobs", json={"tickets": tickets}, headers=AUTH)
        assert created.status_code == 202
        assert time.perf_counter() - started < 5, "202 must come back within 5 s"
        wait_status(client, created.json()["job_id"], "succeeded")
        body = client.get(f"/batch/jobs/{created.json()['job_id']}/results", headers=AUTH).json()
        assert_valid(validators["batch_job_results"], body)
        assert [p["ticket_id"] for p in body["predictions"]] == [t["ticket_id"] for t in tickets]
        assert body["next_offset"] is None


def test_bad_paging_is_422(tmp_path):
    with TestClient(build(tmp_path)) as client:
        job = client.post("/batch/jobs", json={"tickets": make_tickets(3)}, headers=AUTH).json()
        wait_status(client, job["job_id"], "succeeded")
        for params in ({"limit": 0}, {"limit": 5001}, {"offset": -1}, {"offset": "x"}):
            response = client.get(f"/batch/jobs/{job['job_id']}/results", params=params, headers=AUTH)
            assert response.status_code == 422, params


def test_unset_api_key_refuses_everyone(tmp_path):
    with TestClient(build(tmp_path, key=None)) as client:
        ticket = make_tickets(1)[0]
        assert client.post("/predict", json=ticket, headers={"X-API-Key": ""}).status_code == 401
        assert client.post("/predict", json=ticket, headers={"X-API-Key": "anything"}).status_code == 401
        assert client.get("/health").status_code == 200


def test_trained_bundle_replaces_the_stub(tmp_path, validators):
    """Same file layout P2 ships: category / secondary / urgent heads plus labels.json."""
    import json

    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    from app.routing import CATEGORY_TO_TEAM
    from app.textproc import clean_text

    texts = ["refund my money please", "driver was rude and unsafe", "food arrived cold and stale",
             "my order is very late", "i left my phone in the car", "app crashes when i pay"] * 3
    labels = ["payment_refund", "safety_conduct", "food_quality",
              "delivery_delay", "lost_item", "app_technical"] * 3

    def head(y):
        return make_pipeline(TfidfVectorizer(preprocessor=clean_text), LogisticRegression(max_iter=500)).fit(texts, y)

    joblib.dump(
        {
            "category": head(labels),
            "secondary": head(["none", "none", "payment_refund", "payment_refund", "none", "none"] * 3),
            "urgent": head([label == "safety_conduct" for label in labels]),
        },
        tmp_path / "model.joblib",
    )
    (tmp_path / "labels.json").write_text(
        json.dumps({"model_version": "v-test", "category_to_team": CATEGORY_TO_TEAM}), encoding="utf-8"
    )

    model = load(tmp_path)
    assert model.model_version == "v-test"
    with TestClient(build(tmp_path, model)) as client:
        assert client.get("/health").json()["model_version"] == "v-test"
        body = client.post("/predict", json={"channel": "chat", "text": "refund my money please"}, headers=AUTH).json()
        assert_valid(validators["predict_response"], body)
        assert body["category"] == "payment_refund"
        assert body["team"] == "Payments & Refunds"
        assert body["model_version"] == "v-test"


def test_broken_model_file_fails_loudly(tmp_path):
    (tmp_path / "model.joblib").write_bytes(b"not a model")
    with pytest.raises(Exception):
        load(tmp_path)
