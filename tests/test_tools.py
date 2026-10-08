import json

import pytest

from app.tools import TOOLS, run_tool

CALLS = [
    ("get_overview", {}),
    ("get_performance", {"period": "1m"}),
    ("get_attribution", {"period": "1w"}),
    ("get_risk_report", {}),
    ("get_news", {"symbols": ["NVDA", "MSFT"], "days": 7}),
    ("get_thesis", {"symbol": "NVDA"}),
    ("get_upcoming_events", {"days": 30}),
    ("get_chart", {"kind": "attribution", "period": "1w"}),
    ("get_chart", {"kind": "stock_vs_market", "symbol": "LLY", "period": "1m"}),
    ("get_chart", {"kind": "portfolio_vs_market"}),
]


def test_every_tool_has_a_valid_schema():
    names = [t["name"] for t in TOOLS]
    assert len(names) == len(set(names)) == 8
    for t in TOOLS:
        assert t["description"] and t["input_schema"]["type"] == "object"
        json.dumps(t)


@pytest.mark.parametrize("name,inp", CALLS)
def test_dispatch_returns_valid_json(analytics, name, inp):
    out = run_tool(analytics, name, inp)
    assert not out.is_error, out.content
    payload = json.loads(out.content)
    assert isinstance(payload, dict)
    assert out.summary


def test_chart_tool_sends_short_note_and_keeps_spec(analytics):
    out = run_tool(analytics, "get_chart", {"kind": "attribution", "period": "1w"})
    assert json.loads(out.content)["status"] == "Chart shown to user"
    assert "series" not in out.content
    assert out.chart["series"]


@pytest.mark.parametrize(
    "name,inp",
    [
        ("get_performance", {"period": "5y"}),
        ("get_thesis", {"symbol": "NOPE"}),
        ("get_thesis", {}),
        ("get_news", {"symbols": []}),
        ("get_chart", {"kind": "stock_vs_market"}),
        ("get_upcoming_events", {"days": "soon"}),
        ("no_such_tool", {}),
    ],
)
def test_errors_come_back_as_tool_results(analytics, name, inp):
    out = run_tool(analytics, name, inp)
    assert out.is_error
    assert "error" in json.loads(out.content)
    assert out.trace_entry()["ok"] is False


def test_provider_crash_becomes_tool_result(analytics, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("network down")

    monkeypatch.setattr(analytics.provider, "news", boom)
    out = run_tool(analytics, "get_news", {"symbols": ["NVDA"]})
    # news() swallows per-symbol provider failures and returns empty lists
    assert not out.is_error and json.loads(out.content)["news"]["NVDA"] == []

    monkeypatch.setattr(analytics, "risk", boom)
    out = run_tool(analytics, "get_risk_report", {})
    assert out.is_error and "network down" in json.loads(out.content)["error"]


def test_trace_summary_is_human_readable(analytics):
    out = run_tool(analytics, "get_attribution", {"period": "1w"})
    assert out.summary == "Split the last 5 trading days into market, sector and stock-picking"
    assert out.detail.startswith("Of your")
