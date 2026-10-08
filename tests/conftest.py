"""Shared fixtures for the TensorForge contract tests.

Two ways to run:
  1. In-process (default):   pytest            -> imports app.main:app and uses TestClient
  2. Against a live server:  set TF_BASE_URL=http://localhost:8000  (or the hosted URL)

Environment variables:
  TF_API_KEY   key to send / configure (default: test-key)
  TF_BASE_URL  if set, tests call that server instead of the in-process app
  TF_JOB_TIMEOUT  seconds to wait for batch jobs (default 120)
"""
import json
import os
import time
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

API_KEY = os.environ.get("TF_API_KEY", "test-key")
BASE_URL = os.environ.get("TF_BASE_URL")
JOB_TIMEOUT = float(os.environ.get("TF_JOB_TIMEOUT", "120"))
SPEC_DIR = Path(__file__).resolve().parent.parent / "spec"

TEAM_BY_CATEGORY = {
    "payment_refund": "Payments & Refunds",
    "ride_trip_issue": "Ride Operations",
    "lost_item": "Lost & Found",
    "order_missing_wrong": "Food Operations",
    "delivery_delay": "Delivery Operations",
    "food_quality": "Restaurant Quality",
    "account_promo": "Account Services",
    "safety_conduct": "Trust & Safety",
    "app_technical": "Tech Support",
    "general_inquiry": "Front-line Support",
    "spam_irrelevant": "Auto-close / Spam Filter",
}

# Spec examples: English, Singlish, Tamil, Sinhala, polite email, plus emoji-only.
VALID_TICKETS = [
    {"ticket_id": "T-1", "channel": "chat", "subject": "",
     "text": "bro mage order eka hour ekakata wada late, driver call ganne na. refund ekak denna"},
    {"ticket_id": "T-2", "channel": "call_transcript", "subject": "",
     "text": "customer: hello hello yes uh the driver he is um he took a wrong turn and he's not stopping the car i am scared please"},
    {"ticket_id": "T-3", "channel": "chat", "text": "என் ஆர்டர்ல ஒரு ஐட்டம் வரல, பணம் திருப்பி தாங்க"},
    {"ticket_id": "T-4", "channel": "chat", "text": "කාර් එකේ මගේ බෑග් එක අමතක වුණා, ඩ්‍රයිවර්ට කතා කරන්න පුළුවන්ද?"},
    {"ticket_id": "T-5", "channel": "email", "subject": "Re: Order #48213",
     "text": "Hello, my father is diabetic and the insulin pack in my order has not arrived. The rider's phone is off. Kindly advise."},
    {"ticket_id": "T-6", "channel": "chat", "text": "😡😡😡"},
]


def make_tickets(n, prefix="B"):
    base = VALID_TICKETS
    return [
        {"ticket_id": f"{prefix}-{i:05d}", "channel": base[i % len(base)]["channel"],
         "subject": base[i % len(base)].get("subject", ""),
         "text": base[i % len(base)]["text"] + f" ref{i}"}
        for i in range(n)
    ]


def _validator(name):
    with open(SPEC_DIR / f"{name}.schema.json", encoding="utf-8") as f:
        return Draft202012Validator(json.load(f))


@pytest.fixture(scope="session")
def validators():
    names = ["predict_request", "predict_response", "batch_request", "batch_response",
             "batch_job_request", "batch_job_status", "batch_job_results",
             "health_response", "error_response"]
    return {n: _validator(n) for n in names}


def assert_valid(validator, instance):
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
    assert not errors, "Schema violations:\n" + "\n".join(
        f"  at {list(e.path)}: {e.message}" for e in errors[:8])


@pytest.fixture(scope="session")
def client():
    if BASE_URL:
        import httpx
        with httpx.Client(base_url=BASE_URL, timeout=60) as c:
            yield c
    else:
        os.environ["API_KEY"] = API_KEY  # must be set before the app reads it
        from fastapi.testclient import TestClient
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture(scope="session")
def auth():
    return {"X-API-Key": API_KEY}


@pytest.fixture(scope="session")
def jsonh(auth):
    return {**auth, "Content-Type": "application/json"}


def wait_for_job(client, auth, job_id, timeout=JOB_TIMEOUT):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = client.get(f"/batch/jobs/{job_id}", headers=auth)
        assert r.status_code == 200, r.text
        last = r.json()
        if last["status"] in ("succeeded", "failed", "cancelled"):
            return last
        time.sleep(0.5)
    pytest.fail(f"job {job_id} not finished after {timeout}s, last status: {last}")
