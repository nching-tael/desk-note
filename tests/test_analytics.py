import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.analytics import PERIODS, Analytics, AnalyticsError, normalise_period
from app.data import MockProvider, normalise_news_item, sector_etf
from app.portfolio import Holding, load_holdings, load_theses


def test_mock_is_deterministic(provider):
    other = MockProvider(as_of=provider.as_of())
    a = provider.prices(["NVDA", "SPY"], date(2025, 1, 1))
    b = other.prices(["NVDA", "SPY"], date(2025, 1, 1))
    pd.testing.assert_frame_equal(a, b)


def test_mock_week_story(analytics):
    perf = analytics.performance("1w")
    assert -2.5 < perf["market_spy_pct"] < -1.5  # market falls about 2%
    by_sym = {p["symbol"]: p for p in perf["positions"]}
    assert by_sym["NVDA"]["pct"] < -8
    assert by_sym["LLY"]["pct"] > 5
    assert by_sym["XOM"]["pct"] > 0


@pytest.mark.parametrize("period", PERIODS)
def test_attribution_components_sum_to_total_change(analytics, period):
    df, w = analytics.attribution_frame(period)
    parts = df["market"] + df["sector_move"] + df["stock_specific"]
    np.testing.assert_allclose(parts, df["total"], atol=1e-6)
    # ... and the total equals the actual change in portfolio value.
    actual = analytics.port.iloc[w.end_pos] - analytics.port.iloc[w.start_pos]
    assert df["total"].sum() == pytest.approx(actual, abs=1e-6)


def test_rounded_attribution_is_consistent(analytics):
    att = analytics.attribution("1w")
    assert abs(att["market"] + att["sector"] + att["stock_specific"] - att["total"]) <= 2
    assert att["headline"].startswith("Of your -$")


def test_nvda_drop_is_mostly_stock_specific(analytics):
    df, _ = analytics.attribution_frame("1w")
    nvda = df.loc["NVDA"]
    assert nvda["total"] < 0
    assert nvda["stock_specific"] < 0
    assert abs(nvda["stock_specific"]) > 0.5 * abs(nvda["total"])
    assert abs(nvda["stock_specific"]) > abs(nvda["market"])
    assert abs(nvda["stock_specific"]) > abs(nvda["sector_move"])


def test_risk_shares_sum_to_100(analytics):
    risk = analytics.risk()
    total = sum(p["risk_share_pct"] for p in risk["risk_contributions"])
    assert total == pytest.approx(100, abs=0.5)
    assert sum(risk["sector_weights_pct"].values()) == pytest.approx(100, abs=0.5)
    assert 1 <= risk["effective_number_of_positions"] <= risk["number_of_positions"]
    assert risk["stress_test_spy_down_10pct"]["dollars"] < 0


def test_risk_finds_semiconductor_cluster(analytics):
    risk = analytics.risk()
    group = risk["correlated_groups"][0]
    assert {"NVDA", "TSM", "AVGO", "AMD"} <= set(group["symbols"])
    assert 35 < group["weight_pct"] < 50
    semis = {"NVDA", "TSM", "AVGO", "AMD"}
    assert sum(set(p["pair"]) <= semis for p in risk["correlated_pairs_above_0_7"]) >= 3


@pytest.mark.parametrize("period,sessions", [("1d", 1), ("1w", 5), ("1m", 21), ("3m", 63), ("1y", 252)])
def test_periods_select_right_number_of_sessions(analytics, period, sessions):
    w = analytics.window(period)
    assert w.sessions == sessions
    assert w.end_pos - w.start_pos == sessions
    assert w.end == analytics.closes.index[-1]
    perf = analytics.performance(period)
    assert perf["sessions"] == sessions


def test_ytd_starts_at_last_close_of_previous_year(analytics):
    w = analytics.window("ytd")
    assert w.start.year == w.end.year - 1
    assert analytics.closes.index[w.start_pos + 1].year == w.end.year


def test_period_aliases_and_errors():
    assert normalise_period("week") == "1w"
    assert normalise_period("YTD") == "ytd"
    with pytest.raises(AnalyticsError):
        normalise_period("5y")


def test_overview(analytics):
    o = analytics.overview()
    assert o["total_value"] == pytest.approx(sum(p["value"] for p in o["positions"]), abs=11)
    assert sum(p["weight_pct"] for p in o["positions"]) == pytest.approx(100, abs=0.5)
    assert o["week_change"]["dollars"] < 0
    json.dumps(o)


def test_thesis_includes_watch_list_read_across(analytics):
    t = analytics.thesis("NVDA")
    assert "MSFT" in t["watch_list"]
    msft_titles = [n["title"] for n in t["news_14d"]["watch_list"]["MSFT"]]
    assert any("data center spending" in title for title in msft_titles)
    assert t["performance"]["1w"]["stock_specific_dollars"] < 0
    with pytest.raises(AnalyticsError):
        analytics.thesis("ZZZZ")


def test_events_include_tsmc(analytics):
    ev = analytics.events(30)
    tsm = next(e for e in ev["events"] if e["symbol"] == "TSM")
    assert tsm["days_until"] == 4
    assert tsm["implied_move_pct"] == pytest.approx(6.1)
    assert tsm["implied_move_dollars"] > 0
    assert all(e["days_until"] <= 30 for e in ev["events"])


def test_charts(analytics):
    c = analytics.chart("portfolio_vs_market", period="1m")
    assert len(c["labels"]) == 22 and all(len(s["data"]) == 22 for s in c["series"])
    assert c["series"][0]["data"][0] == c["series"][1]["data"][0]
    s = analytics.chart("stock_vs_market", symbol="NVDA", period="1w")
    assert s["series"][0]["data"][0] == 0
    a = analytics.chart("attribution", period="1w")
    assert len(a["series"]) == 3 and len(a["labels"]) == len(analytics.symbols)
    with pytest.raises(AnalyticsError):
        analytics.chart("pie")


def test_unknown_symbol_and_missing_sector_degrade(tmp_path, provider):
    csv = tmp_path / "p.csv"
    csv.write_text("symbol,shares,cost_basis\nNVDA,10,100\nZZTOP,5,\n")
    a = Analytics(provider, load_holdings(csv), {}).load()
    att = a.attribution("1w")
    assert {h["symbol"] for h in att["holdings"]} == {"NVDA", "ZZTOP"}
    z = next(h for h in att["holdings"] if h["symbol"] == "ZZTOP")
    assert z["sector_move"] == 0 and z["sector_etf"] is None
    assert a.overview()["unrealised_pl"]["cost_basis_total"] == 1000


def test_holdings_merge_duplicates(tmp_path):
    csv = tmp_path / "p.csv"
    csv.write_text("symbol,shares,cost_basis\nnvda,10,100\nNVDA,10,200\n")
    (h,) = load_holdings(csv)
    assert h == Holding("NVDA", 20, 150)


def test_theses_load():
    t = load_theses()
    assert t["NVDA"]["watch"][0] == "MSFT"
    assert t["NVDA"]["breaks_if"]


def test_sector_mapping():
    assert sector_etf("Technology") == "XLK"
    assert sector_etf("Health Care") == "XLV"
    assert sector_etf(None) is None


def test_news_normalisation_both_schemas():
    old = {"title": "A", "publisher": "P", "link": "http://x", "providerPublishTime": 1700000000}
    new = {"content": {"title": "B", "pubDate": "2026-10-01T12:00:00Z",
                       "provider": {"displayName": "Q"}, "canonicalUrl": {"url": "http://y"}}}
    assert normalise_news_item("X", old)["published"] == "2023-11-14T22:13:20Z"
    assert normalise_news_item("X", new)["publisher"] == "Q"
    assert normalise_news_item("X", {"content": {}}) is None
