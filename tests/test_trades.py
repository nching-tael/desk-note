from datetime import date

import numpy as np
import pytest

from app.analytics import Analytics
from app.portfolio import Holding, PortfolioError, Trade, load_holdings, load_theses, replay


def test_replay_average_cost_and_realised():
    opening = [Holding("NVDA", 10, 100.0)]
    trades = [
        Trade(date(2026, 1, 5), "NVDA", "buy", 10, 200.0),  # avg -> 150
        Trade(date(2026, 2, 5), "NVDA", "sell", 5, 180.0),  # realised 5 * 30
        Trade(date(2026, 3, 5), "AMD", "buy", 4, 50.0),
    ]
    book = replay(opening, trades)
    assert book["NVDA"].shares == 15
    assert book["NVDA"].avg_cost == pytest.approx(150)
    assert book["NVDA"].realised == pytest.approx(150)
    assert book["AMD"].avg_cost == 50


def test_replay_rejects_overselling():
    with pytest.raises(PortfolioError, match="only 10 held"):
        replay([Holding("NVDA", 10, 100.0)], [Trade(date(2026, 1, 5), "NVDA", "sell", 11, 120.0)])


@pytest.fixture(scope="module")
def traded(provider):
    """Mock portfolio, plus: bought 100 AMD three sessions ago, sold all LLY two
    sessions ago, sold a few NVDA today."""
    a = Analytics(provider, load_holdings(), load_theses()).load()
    idx = a.closes.index
    trades = [
        Trade(idx[-4].date(), "AMD", "buy", 100, float(a.closes["AMD"].iloc[-4]), id=1),
        Trade(idx[-3].date(), "LLY", "sell", 18, float(a.closes["LLY"].iloc[-3]), id=2),
        Trade(idx[-1].date(), "NVDA", "sell", 20, float(a.closes["NVDA"].iloc[-1]), id=3),
    ]
    return Analytics(provider, load_holdings(), load_theses(), trades=trades).load()


def test_shares_change_at_trade_dates(traded):
    s = traded.shares_df
    assert s["AMD"].iloc[-5] == 60 and s["AMD"].iloc[-4] == 160
    assert s["LLY"].iloc[-4] == 18 and s["LLY"].iloc[-3] == 0
    assert s["NVDA"].iloc[-1] == 100
    assert "LLY" not in traded.symbols and "LLY" in traded.all_symbols


def test_attribution_still_sums_with_trades(traded):
    for period in ("1d", "1w", "1m"):
        df, w = traded.attribution_frame(period)
        np.testing.assert_allclose(
            df["market"] + df["sector_move"] + df["stock_specific"], df["total"], atol=1e-6
        )
        assert df["total"].sum() == pytest.approx(traded.pnl.iloc[w.start_pos + 1 : w.end_pos + 1].sum())
    df, _ = traded.attribution_frame("1w")
    assert "LLY" in df.index  # sold mid-week, but its gain before selling counts


def test_performance_separates_purchases_from_gains(traded):
    perf = traded.performance("1w")
    assert "net_purchases_dollars" in perf
    # Buying AMD and selling LLY moved the value, but only gains count as change.
    assert perf["change_dollars"] == traded.attribution("1w")["total"]
    over = traded.overview()
    assert over["week_change"]["dollars"] == perf["change_dollars"]
    assert over["realised_pl"]["by_symbol"]["LLY"] > 0
    assert "LLY" not in {p["symbol"] for p in over["positions"]}


def test_risk_and_chart_use_current_holdings(traded):
    risk = traded.risk()
    assert "LLY" not in {p["symbol"] for p in risk["risk_contributions"]}
    assert sum(p["risk_share_pct"] for p in risk["risk_contributions"]) == pytest.approx(100, abs=0.5)
    chart = traded.chart_portfolio_vs_market("1w")
    assert chart["series"][0]["data"][0] == chart["series"][1]["data"][0]
    assert len(chart["series"][0]["data"]) == len(chart["labels"])


def test_thesis_for_holding_bought_today(provider):
    a = Analytics(provider, load_holdings(), load_theses()).load()
    today = a.closes.index[-1].date()
    b = Analytics(
        provider, load_holdings(), load_theses(), trades=[Trade(today, "META", "buy", 5, 700.0, id=1)]
    ).load()
    assert "META" in b.symbols
    t = b.thesis("META")
    assert t["performance"] == {}  # no move yet: bought at today's close
    assert t["thesis"] is None
