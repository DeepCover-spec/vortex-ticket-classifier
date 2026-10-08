"""Happy-path contract tests: every response must match the organizers' JSON Schemas."""
import uuid

import pytest

from tests.conftest import (TEAM_BY_CATEGORY, VALID_TICKETS, assert_valid, make_tickets,
                      wait_for_job)


def check_prediction(p, request_ticket=None):
    assert "secondary_category" in p, "secondary_category key must always be present (null if none)"
    assert p["team"] == TEAM_BY_CATEGORY[p["category"]], f"team does not match category {p['category']}"
    assert p["secondary_category"] != p["category"]
    assert 0.0 <= p["confidence"] <= 1.0
    if p["category"] == "spam_irrelevant":
        assert p["is_urgent"] is False and p["secondary_category"] is None
    if request_ticket and "ticket_id" in request_ticket:
        assert p.get("ticket_id") == request_ticket["ticket_id"]


# ---------- /health ----------

def test_health_public_and_valid(client, validators):
    r = client.get("/health")  # no API key on purpose
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert_valid(validators["health_response"], body)
    assert body["status"] == "ok" and body["model_version"]


# ---------- /predict ----------

@pytest.mark.parametrize("ticket", VALID_TICKETS, ids=[t["ticket_id"] for t in VALID_TICKETS])
def test_predict_valid_tickets(client, jsonh, validators, ticket):
    assert_valid(validators["predict_request"], ticket)
    r = client.post("/predict", json=ticket, headers=jsonh)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert_valid(validators["predict_response"], body)
    check_prediction(body, ticket)


def test_predict_accepts_bearer_token(client, jsonh):
    from tests.conftest import API_KEY
    h = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    r = client.post("/predict", json=VALID_TICKETS[0], headers=h)
    assert r.status_code == 200, r.text


def test_predict_without_ticket_id(client, jsonh, validators):
    r = client.post("/predict", json={"channel": "chat", "text": "app keeps crashing on login"}, headers=jsonh)
    assert r.status_code == 200
    assert_valid(validators["predict_response"], r.json())


def test_predict_echoes_request_id(client, jsonh):
    rid = str(uuid.uuid4())
    r = client.post("/predict", json=VALID_TICKETS[0], headers={**jsonh, "X-Request-ID": rid})
    assert r.status_code == 200
    assert r.headers.get("x-request-id") == rid  # optional in spec, but cheap to support


def test_model_version_consistent_everywhere(client, jsonh):
    h = client.get("/health").json()["model_version"]
    p = client.post("/predict", json=VALID_TICKETS[0], headers=jsonh).json()["model_version"]
    b = client.post("/predict/batch", json={"tickets": VALID_TICKETS[:1]}, headers=jsonh).json()
    assert h == p == b["predictions"][0]["model_version"]


def test_predict_is_deterministic(client, jsonh):
    a = client.post("/predict", json=VALID_TICKETS[4], headers=jsonh).json()
    b = client.post("/predict", json=VALID_TICKETS[4], headers=jsonh).json()
    assert a == b


# ---------- /predict/batch ----------

def test_batch_small_order_and_ids(client, jsonh, validators):
    req = {"tickets": VALID_TICKETS}
    assert_valid(validators["batch_request"], req)
    r = client.post("/predict/batch", json=req, headers=jsonh)
    assert r.status_code == 200, r.text
    body = r.json()
    assert_valid(validators["batch_response"], body)
    assert [p["ticket_id"] for p in body["predictions"]] == [t["ticket_id"] for t in VALID_TICKETS]
    for p, t in zip(body["predictions"], VALID_TICKETS):
        check_prediction(p, t)


def test_batch_of_100(client, jsonh, validators):
    tickets = make_tickets(100)
    r = client.post("/predict/batch", json={"tickets": tickets}, headers=jsonh)
    assert r.status_code == 200, r.text
    body = r.json()
    assert_valid(validators["batch_response"], body)
    assert [p["ticket_id"] for p in body["predictions"]] == [t["ticket_id"] for t in tickets]


def test_batch_matches_single_predictions(client, jsonh):
    """Items must be classified independently of position/other tickets."""
    tickets = make_tickets(6)
    batch = client.post("/predict/batch", json={"tickets": tickets}, headers=jsonh).json()["predictions"]
    rev = client.post("/predict/batch", json={"tickets": tickets[::-1]}, headers=jsonh).json()["predictions"][::-1]
    for t, b, rv in zip(tickets, batch, rev):
        single = client.post("/predict", json=t, headers=jsonh).json()
        for key in ("category", "secondary_category", "team", "is_urgent"):
            assert b[key] == single[key] == rv[key], f"{t['ticket_id']}: {key} differs by context"


# ---------- /batch/jobs ----------

def test_job_full_lifecycle(client, jsonh, auth, validators):
    tickets = make_tickets(250, prefix="J")
    r = client.post("/batch/jobs", json={"tickets": tickets}, headers=jsonh)
    assert r.status_code == 202, r.text
    assert "location" in r.headers and "retry-after" in r.headers
    status = r.json()
    assert_valid(validators["batch_job_status"], status)
    assert status["total"] == 250
    job_id = status["job_id"]

    final = wait_for_job(client, auth, job_id)
    assert final["status"] == "succeeded", final
    assert_valid(validators["batch_job_status"], final)
    assert final["processed"] == final["total"] == 250
    assert final["expires_at"]

    r = client.get(f"/batch/jobs/{job_id}/results", headers=auth)
    assert r.status_code == 200, r.text
    res = r.json()
    assert_valid(validators["batch_job_results"], res)
    assert res["total"] == 250 and res["next_offset"] is None
    assert [p["ticket_id"] for p in res["predictions"]] == [t["ticket_id"] for t in tickets]
    for p, t in zip(res["predictions"], tickets):
        check_prediction(p, t)

    # same predictions as the synchronous endpoint (same model, same rules)
    sync = client.post("/predict/batch", json={"tickets": tickets[:20]}, headers=jsonh).json()["predictions"]
    for a, b in zip(res["predictions"][:20], sync):
        assert (a["category"], a["secondary_category"], a["is_urgent"]) == \
               (b["category"], b["secondary_category"], b["is_urgent"])

    # paging is stable and contiguous
    p1 = client.get(f"/batch/jobs/{job_id}/results", params={"offset": 0, "limit": 100}, headers=auth).json()
    p2 = client.get(f"/batch/jobs/{job_id}/results", params={"offset": p1["next_offset"], "limit": 100}, headers=auth).json()
    p3 = client.get(f"/batch/jobs/{job_id}/results", params={"offset": p2["next_offset"], "limit": 100}, headers=auth).json()
    assert len(p1["predictions"]) == len(p2["predictions"]) == 100 and len(p3["predictions"]) == 50
    assert p3["next_offset"] is None
    paged = p1["predictions"] + p2["predictions"] + p3["predictions"]
    assert paged == res["predictions"]

    assert client.delete(f"/batch/jobs/{job_id}", headers=auth).status_code == 204


def test_job_idempotency_key(client, jsonh, auth):
    key = str(uuid.uuid4())
    h = {**jsonh, "Idempotency-Key": key}
    body = {"tickets": make_tickets(5, prefix="I")}
    a = client.post("/batch/jobs", json=body, headers=h)
    b = client.post("/batch/jobs", json=body, headers=h)
    assert a.status_code == 202 and b.status_code == 202
    assert a.json()["job_id"] == b.json()["job_id"]
    wait_for_job(client, auth, a.json()["job_id"])
    client.delete(f"/batch/jobs/{a.json()['job_id']}", headers=auth)


def test_job_results_before_done_is_409_or_ready(client, jsonh, auth):
    r = client.post("/batch/jobs", json={"tickets": make_tickets(1500, prefix="E")}, headers=jsonh)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    early = client.get(f"/batch/jobs/{job_id}/results", headers=auth)
    status_now = client.get(f"/batch/jobs/{job_id}", headers=auth).json()["status"]
    if early.status_code == 409:
        assert early.json()["error"]["code"]
    else:  # only acceptable if the job really had finished already
        assert early.status_code == 200 and status_now == "succeeded"
    wait_for_job(client, auth, job_id)
    client.delete(f"/batch/jobs/{job_id}", headers=auth)


def test_job_unknown_id_is_404(client, auth, validators):
    for path in (f"/batch/jobs/{uuid.uuid4()}", f"/batch/jobs/{uuid.uuid4()}/results"):
        r = client.get(path, headers=auth)
        assert r.status_code == 404
        assert_valid(validators["error_response"], r.json())
    r = client.delete(f"/batch/jobs/{uuid.uuid4()}", headers=auth)
    assert r.status_code == 404


def test_health_and_predict_responsive_during_job(client, jsonh, auth):
    import time
    r = client.post("/batch/jobs", json={"tickets": make_tickets(3000, prefix="L")}, headers=jsonh)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    for _ in range(3):
        t0 = time.time()
        assert client.get("/health").status_code == 200
        assert client.post("/predict", json=VALID_TICKETS[0], headers=jsonh).status_code == 200
        assert time.time() - t0 < 3, "live endpoints stalled while a job runs"
    client.delete(f"/batch/jobs/{job_id}", headers=auth)
