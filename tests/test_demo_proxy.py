"""Demo proxy tests. They never call the hosted API and never use the real key.

The container CI job installs pytest and httpx only, then runs this suite against
the API image. These tests need FastAPI, so they skip there.
"""

import httpx
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import demo.main as demo

SECRET = "super-secret-key"


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.post_handler = FakeClient.post_handler
        self.get_handler = FakeClient.get_handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def request(self, method, url, json=None, headers=None):
        assert headers["X-API-Key"] == SECRET
        assert SECRET not in url
        return self.post_handler(method, url, json, headers)

    async def get(self, url):
        return self.get_handler(url)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", SECRET)
    monkeypatch.setenv("API_BASE_URL", "https://classifier.example")
    monkeypatch.setattr(demo.httpx, "AsyncClient", FakeClient)
    FakeClient.post_handler = lambda *_args: FakeResponse(200, {})
    FakeClient.get_handler = lambda _url: FakeResponse(200, {"status": "ok", "model_version": "p3"})
    return TestClient(demo.app)


def test_page_and_script_do_not_contain_the_key(client):
    page = client.get("/")
    script = client.get("/static/app.js")
    assert page.status_code == 200
    assert script.status_code == 200
    combined = page.text + script.text
    assert SECRET not in combined
    assert "X-API-Key" not in combined


def test_missing_key_is_a_friendly_error(monkeypatch):
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("API_BASE_URL", "https://classifier.example")
    local = TestClient(demo.app)
    response = local.post("/ui/predict", json={"text": "The ride was charged twice.", "channel": "chat"})
    assert response.status_code == 503
    assert "traceback" not in response.text.lower()
    assert "API_KEY" not in response.text


def test_predict_proxies_the_contract_and_hides_the_key(client):
    seen = {}

    def handle(_method, url, payload, _headers):
        seen["url"] = url
        seen["payload"] = payload
        return FakeResponse(
            200,
            {
                "ticket_id": payload["ticket_id"],
                "category": "payment_refund",
                "secondary_category": None,
                "team": "Payments & Refunds",
                "is_urgent": False,
                "confidence": 0.947,
                "model_version": "p3-tfidf",
            },
        )

    FakeClient.post_handler = handle
    response = client.post(
        "/ui/predict",
        json={"text": "I was charged twice for the same ride.", "channel": "chat"},
    )
    body = response.json()
    assert response.status_code == 200
    assert seen["url"] == "https://classifier.example/predict"
    assert seen["payload"]["channel"] == "chat"
    assert seen["payload"]["text"] == "I was charged twice for the same ride."
    assert body["category"] == "payment_refund"
    assert body["confidence"] == 0.947
    assert body["team"] == "Payments & Refunds"
    assert SECRET not in response.text


def test_upstream_error_message_is_shown_without_the_raw_body(client):
    FakeClient.post_handler = lambda *_args: FakeResponse(
        422,
        {"error": {"code": "validation_error", "message": "Field 'text' must contain text.", "details": []}},
    )
    response = client.post("/ui/predict", json={"text": "hello", "channel": "chat"})
    assert response.status_code == 422
    assert response.json() == {"error": "Field 'text' must contain text."}


def test_upstream_message_containing_the_key_is_replaced(client):
    FakeClient.post_handler = lambda *_args: FakeResponse(
        500,
        {"error": {"code": "boom", "message": f"failed with {SECRET}"}},
    )
    response = client.post("/ui/predict", json={"text": "hello", "channel": "chat"})
    assert response.status_code == 503
    assert SECRET not in response.text
    assert "traceback" not in response.text.lower()


def test_batch_preserves_order(client):
    def handle(_method, url, payload, _headers):
        assert url.endswith("/predict/batch")
        predictions = []
        for ticket in payload["tickets"]:
            predictions.append(
                {
                    "ticket_id": ticket["ticket_id"],
                    "category": "delivery_delay",
                    "secondary_category": None,
                    "team": "Delivery Operations",
                    "is_urgent": True,
                    "confidence": 0.8,
                    "model_version": "p3-tfidf",
                }
            )
        return FakeResponse(200, {"predictions": predictions})

    FakeClient.post_handler = handle
    response = client.post(
        "/ui/batch",
        json={"channel": "chat", "tickets": ["First order is late.", "Second order is late.", "Third order is late."]},
    )
    body = response.json()
    assert response.status_code == 200
    assert [row["text"] for row in body["predictions"]] == [
        "First order is late.",
        "Second order is late.",
        "Third order is late.",
    ]
    assert SECRET not in response.text


def test_unreachable_api_is_friendly(client, monkeypatch):
    class BrokenClient(FakeClient):
        async def request(self, *args, **kwargs):
            raise httpx.ConnectError("connection failed")

    monkeypatch.setattr(demo.httpx, "AsyncClient", BrokenClient)
    response = client.post("/ui/predict", json={"text": "hello", "channel": "chat"})
    assert response.status_code == 503
    assert "connection failed" not in response.text
    assert SECRET not in response.text


def test_wrong_base_url_health_is_unavailable(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "https://classifier.invalid")
    monkeypatch.delenv("API_KEY", raising=False)

    class BrokenClient(FakeClient):
        async def get(self, _url):
            raise httpx.ConnectError("no route")

    monkeypatch.setattr(demo.httpx, "AsyncClient", BrokenClient)
    response = TestClient(demo.app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"
    assert "no route" not in response.text
    assert "Traceback" not in response.text
