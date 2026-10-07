import json

import anyio
import pytest
from mcp import Client
from starlette.testclient import TestClient

from app.analytics import AnalyticsHolder
from app.mcp_server import create_http_app, create_server
from app.store import Store

TOKEN = "t" * 32


@pytest.fixture()
def holder(provider):
    store = Store(":memory:")
    store.seed_from_files()
    h = AnalyticsHolder(mock=True, store=store)
    h._provider = provider
    return h


def run(coro_fn):
    return anyio.run(coro_fn)


def test_lists_tools_prompts_and_instructions(holder):
    async def go():
        async with Client(create_server(holder)) as client:
            tools = (await client.list_tools()).tools
            prompts = (await client.list_prompts()).prompts
            return client.instructions, tools, prompts

    instructions, tools, prompts = run(go)
    assert "must come from a Desk Note tool" in instructions
    by_name = {t.name: t for t in tools}
    assert len(by_name) == 14
    assert by_name["get_attribution"].annotations.read_only_hint is True
    assert by_name["record_trade"].annotations.read_only_hint is False
    assert by_name["delete_trade"].annotations.destructive_hint is True
    assert {p.name for p in prompts} == {"morning_note", "check_thesis", "review_decisions"}


def test_call_tools_end_to_end(holder):
    async def go():
        async with Client(create_server(holder)) as client:
            att = await client.call_tool("get_attribution", {"period": "1w"})
            chart = await client.call_tool("get_chart", {"kind": "attribution"})
            bad = await client.call_tool("get_thesis", {"symbol": "ZZZZ"})
            trade = await client.call_tool("record_trade", {"symbol": "NVDA", "side": "sell", "shares": 20,
                                                            "price": 180, "reason": "trim"})
            overview = await client.call_tool("get_overview", {})
            prompt = await client.get_prompt("check_thesis", {"symbol": "nvda"})
            return att, chart, bad, trade, overview, prompt

    att, chart, bad, trade, overview, prompt = run(go)
    assert not att.is_error and json.loads(att.content[0].text)["headline"].startswith("Of your")
    assert att.structured_content["total"] < 0
    assert json.loads(chart.content[0].text)["series"]  # MCP returns the chart data itself
    assert bad.is_error and "error" in json.loads(bad.content[0].text)
    assert not trade.is_error
    nvda = next(p for p in json.loads(overview.content[0].text)["positions"] if p["symbol"] == "NVDA")
    assert nvda["shares"] == 100  # the write is visible on the next read
    assert "NVDA" in prompt.messages[0].content.text


INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "test", "version": "1"}}}
HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def test_http_token_gate(holder):
    with TestClient(create_http_app(holder, TOKEN)) as client:
        assert client.post("/mcp", json=INIT, headers=HEADERS).status_code == 401
        assert client.post("/wrong-token/mcp", json=INIT, headers=HEADERS).status_code == 401
        assert client.post("/mcp", json=INIT, headers={**HEADERS, "Authorization": "Bearer nope"}).status_code == 401

        ok = client.post(f"/{TOKEN}/mcp", json=INIT, headers=HEADERS)
        assert ok.status_code == 200, ok.text
        assert ok.json()["result"]["serverInfo"]["name"] == "desk-note"

        bearer = client.post("/mcp", json=INIT, headers={**HEADERS, "Authorization": f"Bearer {TOKEN}"})
        assert bearer.status_code == 200
        # Behind a tunnel the Host header is the public name; that must still work.
        tunneled = client.post(f"/{TOKEN}/mcp", json=INIT,
                               headers={**HEADERS, "Host": "pi.tailnet-1234.ts.net"})
        assert tunneled.status_code == 200
