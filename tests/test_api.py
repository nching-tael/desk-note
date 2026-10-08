from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.analytics import Analytics
from app.portfolio import load_holdings, load_theses
from app.server import MISSING_KEY, create_app


@pytest.fixture()
def mock_analytics(provider):
    return Analytics(provider, load_holdings(), load_theses())


def fake_client_factory():
    script = [
        SimpleNamespace(
            stop_reason="tool_use",
            content=[
                SimpleNamespace(type="tool_use", id="t1", name="get_attribution", input={"period": "1w"}),
                SimpleNamespace(type="tool_use", id="t2", name="get_chart", input={"kind": "attribution"}),
            ],
        ),
        SimpleNamespace(
            stop_reason="end_turn", content=[SimpleNamespace(type="text", text="Mostly the market.")]
        ),
    ]
    create = lambda **kw: script.pop(0)
    return SimpleNamespace(
        messages=SimpleNamespace(create=create), beta=SimpleNamespace(messages=SimpleNamespace(create=create))
    )


def test_index_and_overview(mock_analytics, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    client = TestClient(create_app(mock=True, analytics=mock_analytics))
    r = client.get("/")
    assert r.status_code == 200 and "Desk Note" in r.text
    r = client.get("/api/overview")
    assert r.status_code == 200
    data = r.json()
    assert data["mock"] is True and data["chat_enabled"] is False
    assert data["total_value"] > 0 and data["positions"]
    assert data["chart"]["kind"] == "portfolio_vs_market" and len(data["chart"]["labels"]) == 22


def test_chat_without_key_returns_clear_error(mock_analytics, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    client = TestClient(create_app(mock=True, analytics=mock_analytics))
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503
    assert r.json() == {"error": MISSING_KEY, "code": "missing_api_key"}


def test_chat_end_to_end_with_fake_client(mock_analytics):
    client = TestClient(create_app(mock=True, analytics=mock_analytics, client_factory=fake_client_factory))
    r = client.post(
        "/api/chat", json={"messages": [{"role": "user", "content": "Why did I lose money this week?"}]}
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["answer"] == "Mostly the market."
    assert [t["tool"] for t in data["trace"]] == ["get_attribution", "get_chart"]
    assert data["charts"][0]["kind"] == "attribution"


def test_chat_validation(mock_analytics):
    client = TestClient(create_app(mock=True, analytics=mock_analytics, client_factory=fake_client_factory))
    assert client.post("/api/chat", json={"messages": []}).status_code == 422
    r = client.post("/api/chat", json={"messages": [{"role": "assistant", "content": "hi"}]})
    assert r.status_code == 400


def test_api_errors_are_mapped(mock_analytics):
    import anthropic
    import httpx2

    def failing():
        def create(**kw):
            req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.AuthenticationError(
                "bad key", response=httpx2.Response(401, request=req), body=None
            )

        return SimpleNamespace(
            messages=SimpleNamespace(create=create),
            beta=SimpleNamespace(messages=SimpleNamespace(create=create)),
        )

    client = TestClient(create_app(mock=True, analytics=mock_analytics, client_factory=failing))
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401 and r.json()["code"] == "auth"
