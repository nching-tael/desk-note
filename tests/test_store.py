import json
from datetime import date

import pytest

from app.analytics import AnalyticsHolder
from app.store import Store, StoreError
from app.tools import run_tool


@pytest.fixture()
def store():
    s = Store(":memory:")
    s.seed_from_files()
    return s


@pytest.fixture()
def holder(provider, store):
    h = AnalyticsHolder(mock=True, store=store)
    h._provider = provider  # share the fixed-date mock provider
    return h


def call(holder, name, inp):
    out = run_tool(holder.get(), name, inp, store=holder.store)
    if out.changed:
        holder.invalidate()
    return out


def test_seed_is_idempotent(store):
    assert store.seeded
    assert not store.seed_from_files()
    assert {h.symbol for h in store.opening()} >= {"NVDA", "LLY"}
    assert store.theses()["NVDA"]["watch"][0] == "MSFT"


def test_file_store_persists(tmp_path):
    path = tmp_path / "d.db"
    s = Store(path)
    s.seed_from_files()
    s.add_trade(
        __import__("app.portfolio", fromlist=["Trade"]).Trade(date(2026, 1, 5), "nvda", "sell", 10, 150.0),
        reason="trim",
    )
    again = Store(path)
    assert [t.symbol for t in again.trades()] == ["NVDA"]
    assert again.journal()[0]["trade"]["side"] == "sell"


def test_record_trade_updates_holdings_and_journal(holder):
    before = holder.get().overview()
    nvda_before = next(p for p in before["positions"] if p["symbol"] == "NVDA")["shares"]
    out = call(
        holder,
        "record_trade",
        {
            "symbol": "nvda",
            "side": "sell",
            "shares": 20,
            "price": 180,
            "reason": "Taking profits after the capex scare",
        },
    )
    assert not out.is_error, out.content
    res = json.loads(out.content)
    assert res["position_now"]["shares"] == nvda_before - 20
    assert res["journaled_reason"] is True
    after = holder.get().overview()  # rebuilt after invalidate()
    assert next(p for p in after["positions"] if p["symbol"] == "NVDA")["shares"] == nvda_before - 20
    trades = json.loads(call(holder, "get_trades", {}).content)
    assert trades["count"] == 1 and trades["trades"][0]["side"] == "sell"


def test_record_trade_defaults_price_and_rejects_bad_input(holder):
    out = call(holder, "record_trade", {"symbol": "AMD", "side": "buy", "shares": 5})
    assert "latest close" in json.loads(out.content)["note"]
    for bad in [
        {"symbol": "AMD", "side": "sell", "shares": 10_000},  # more than held
        {"symbol": "AMD", "side": "short", "shares": 1, "price": 1},  # bad side
        {"symbol": "AMD", "side": "buy", "shares": -1, "price": 1},  # negative
        {"symbol": "AMD", "side": "buy", "shares": 1, "price": 1, "date": "2999-01-01"},
        {"symbol": "AMD", "side": "buy", "shares": 1, "price": 1, "date": "not-a-date"},
    ]:
        out = call(holder, "record_trade", bad)
        assert out.is_error, bad
        assert "error" in json.loads(out.content)


def test_new_symbol_buy_and_delete(holder):
    out = call(holder, "record_trade", {"symbol": "META", "side": "buy", "shares": 3, "price": 700})
    tid = json.loads(out.content)["recorded"]["id"]
    assert "META" in holder.get().symbols
    out = call(holder, "delete_trade", {"trade_id": tid})
    assert not out.is_error
    assert "META" not in holder.get().all_symbols
    assert call(holder, "delete_trade", {"trade_id": 999}).is_error


def test_delete_that_breaks_later_sell_is_refused(holder):
    buy = json.loads(
        call(
            holder,
            "record_trade",
            {"symbol": "ORCL", "side": "buy", "shares": 5, "price": 250, "date": "2026-09-01"},
        ).content
    )["recorded"]["id"]
    call(
        holder,
        "record_trade",
        {"symbol": "ORCL", "side": "sell", "shares": 5, "price": 260, "date": "2026-09-10"},
    )
    out = call(holder, "delete_trade", {"trade_id": buy})
    assert out.is_error and "break later trades" in json.loads(out.content)["error"]


def test_update_thesis(holder):
    out = call(
        holder, "update_thesis", {"symbol": "NVDA", "watch_add": ["ORCL", "nvda"], "watch_remove": ["TSM"]}
    )
    watch = json.loads(out.content)["thesis"]["watch"]
    assert "ORCL" in watch and "TSM" not in watch and "NVDA" not in watch
    assert holder.get().thesis("NVDA")["watch_list"] == watch
    assert call(holder, "update_thesis", {"symbol": "ZZZ"}).is_error  # nothing to save
    out = call(holder, "update_thesis", {"symbol": "META", "thesis": "Ads + AI capex discipline"})
    assert not out.is_error and "META" in holder.store.theses()


def test_journal_review(holder):
    a = holder.get()
    month_ago = a.closes.index[-22].date().isoformat()
    call(
        holder,
        "record_trade",
        {
            "symbol": "NVDA",
            "side": "buy",
            "shares": 10,
            "price": 200,
            "date": month_ago,
            "reason": "AI capex keeps growing",
        },
    )
    call(
        holder, "add_journal_entry", {"text": "Watching LLY oral GLP-1 data", "symbol": "LLY", "kind": "note"}
    )
    review = json.loads(call(holder, "review_journal", {}).content)
    by_kind = {e["kind"]: e for e in review["entries"]}
    buy = by_kind["buy_reason"]
    assert buy["outcome"]["sessions"] == 21
    assert buy["outcome"]["gain_since_buy_dollars"] < 0  # NVDA fell this month in the mock
    assert "market_adjusted_pct" in buy["outcome"]
    assert by_kind["note"]["outcome"] is None  # entered today: too recent
    assert review["summary"]["buys_beating_market_adjusted"].endswith("of 1")


def test_store_tools_without_store_fail_cleanly(analytics):
    out = run_tool(analytics, "record_trade", {"symbol": "NVDA", "side": "buy", "shares": 1})
    assert out.is_error and "no database" in json.loads(out.content)["error"]


def test_store_rejects_bad_symbols(store):
    with pytest.raises(StoreError):
        store.add_journal("x", symbol="DROP TABLE")
