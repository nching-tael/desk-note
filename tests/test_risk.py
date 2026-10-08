"""Tail risk, downside beta, drawdowns, stress tests and data checks, against
answers worked out by hand."""

import numpy as np
import pandas as pd
import pytest
from helpers import DummyProvider, frame, prices_from_returns

from app.analytics import Analytics
from app.analytics.stress import crash_replays, market_drops, move
from app.analytics.tail import (
    data_quality,
    downside_beta,
    drawdown,
    holding_returns,
    losses,
    recent_volatility,
    worst_periods,
)
from app.portfolio import Holding

DAYS = pd.bdate_range("2026-01-01", periods=100)


def series(values):
    return pd.Series(values, index=DAYS[: len(values)], dtype=float)


def test_bad_day_losses_by_hand():
    # 100 days: four -5% days, one -4% day, ninety-five +1% days.
    # 1 in 20 = the 5th-worst of 100 = -4%; average of those 5 = (4 x -5 + -4) / 5 = -4.8%.
    # 1 in 100 = the single worst day = -5%.
    r = series([-0.05] * 4 + [-0.04] + [0.01] * 95)
    result = losses(r, value=10_000)
    assert result["1_in_20"] == {
        "loss_dollars": -400,
        "loss_pct": -4.0,
        "average_of_worst_dollars": -480,
        "worst_periods_used": 5,
    }
    assert result["1_in_100"]["loss_dollars"] == -500 and result["1_in_100"]["worst_periods_used"] == 1


def test_bad_week_compounds_overlapping_weeks():
    # five -2% days in a row: the worst week is 0.98^5 - 1 = -9.6%
    r = series([0.0] * 10 + [-0.02] * 5 + [0.0] * 10)
    assert losses(r, 1000, horizon=5)["1_in_100"]["loss_pct"] == -9.6
    assert losses(series([0.01] * 3), 1000, horizon=5)["periods"] == 0  # too short: no crash


def test_downside_beta_only_uses_falling_days():
    rng = np.random.default_rng(1)
    market = pd.Series(rng.normal(0, 0.01, 300))
    stock = market.where(market >= 0, 2 * market).where(market < 0, 0.5 * market)  # 2x down, 0.5x up
    assert downside_beta(stock, market) == pytest.approx(2.0)
    assert stock.cov(market) / market.var() < 1.9  # the everyday beta understates it


def test_recent_volatility_of_steady_one_percent_days():
    r = series([0.01, -0.01] * 50)
    assert recent_volatility(r) == pytest.approx(0.01 * np.sqrt(252))


def test_drawdown_and_recovery_by_hand():
    # 1 -> 1.2 -> 0.9 -> 1.2: a 25% fall over one session, recovered one session later
    r = series([0.2, -0.25, 1 / 3])
    dd = drawdown(r, value=12_000, start=DAYS[0] - pd.Timedelta(days=1))
    assert dd["pct"] == -25.0 and dd["dollars_at_todays_value"] == -3000
    assert (dd["sessions_down"], dd["sessions_to_recover"], dd["recovered"]) == (1, 1, True)


def test_worst_periods_by_hand():
    r = series([0.01] * 30 + [-0.03, -0.04, 0.0, -0.01, -0.02] + [0.01] * 30)
    worst = worst_periods(r, value=1000)
    assert worst["day"]["pct"] == -4.0 and worst["day"]["dollars"] == -40
    expected_week = (0.97 * 0.96 * 1.0 * 0.99 * 0.98) - 1
    assert worst["week"]["pct"] == round(expected_week * 100, 1)


# --- a young holding: listed on day 150, moves 1.2x the market from then on


def young_portfolio():
    rng = np.random.default_rng(2)
    m = rng.normal(0.0004, 0.01, 320)
    young = np.array(prices_from_returns(1.2 * m))
    young[:150] = np.nan
    closes = frame({"OLD": prices_from_returns(m), "NEW": list(young), "SPY": prices_from_returns(m, 500)})
    return Analytics(DummyProvider(closes), [Holding("OLD", 10, None), Holding("NEW", 10, None)], {}).load()


def test_days_before_listing_are_filled_from_beta():
    a = young_portfolio()
    returns, estimated = holding_returns(a)
    market = a.returns["SPY"].iloc[1:]
    assert estimated == {"NEW": 150}
    np.testing.assert_allclose(returns["NEW"].iloc[:149], 1.2 * market.iloc[:149])  # filled, not flat zero
    notes = data_quality(a, estimated)
    assert any("NEW has 169 sessions of real price history" in n for n in notes)


def test_stale_price_is_flagged():
    a = young_portfolio()
    a.returns.loc[a.returns.index[-4:], "OLD"] = 0.0  # unchanged for 4 days while the market moved
    assert any("OLD's price hasn't changed for 4 sessions" in n for n in data_quality(a, {}))


# --- stress tests


def test_market_drops_use_downside_beta_and_cap_at_total_loss():
    a = young_portfolio()
    value_old = float(a.values["OLD"].iloc[-1])
    drops = market_drops(a, betas={"OLD": 2.0, "NEW": 4.0})
    ten = drops[0]
    assert ten["market_move_pct"] == -10.0
    expected = value_old * -0.2 + float(a.values["NEW"].iloc[-1]) * -0.4
    assert ten["dollars"] == round(expected)
    # at -35%, a beta of 4 would be -140%: capped at losing everything
    thirty_five = drops[2]
    new = next(h for h in thirty_five["biggest_losers"] if h["symbol"] == "NEW")
    assert new["dollars"] == -round(float(a.values["NEW"].iloc[-1]))


def test_crash_replay_uses_actual_moves_where_it_can():
    a = young_portfolio()
    idx = a.closes.index
    crash = [
        {
            "name": "Test crash",
            "start": idx[10].date().isoformat(),
            "end": idx[40].date().isoformat(),
            "sp500_move": -0.5,
        }
    ]
    (replay,) = crash_replays(a, crash, betas={"OLD": 1.0, "NEW": 1.5})
    market = float(a.closes["SPY"].iloc[40] / a.closes["SPY"].iloc[10] - 1)
    old = float(a.closes["OLD"].iloc[40] / a.closes["OLD"].iloc[10] - 1)
    assert replay["market_source"] == "actual SPY" and replay["market_move_pct"] == round(market * 100, 1)
    by_symbol = {h["symbol"]: h for h in replay["holdings"]}
    assert by_symbol["OLD"]["estimated"] is False and by_symbol["OLD"]["move_pct"] == round(old * 100, 1)
    # NEW didn't exist yet: downside beta 1.5 x the market's actual move
    assert by_symbol["NEW"]["estimated"] is True
    assert by_symbol["NEW"]["move_pct"] == round(1.5 * market * 100, 1)
    assert replay["estimated_holdings"] == ["NEW"]


def test_crash_replay_uses_what_a_young_fund_tracks():
    # IBIT has no prices for the crash, but BTC-USD (what it tracks) does: replay that.
    idx = pd.bdate_range(end="2026-10-07", periods=60)
    ibit = [np.nan] * 40 + list(np.linspace(50, 55, 20))
    closes = pd.DataFrame(
        {"IBIT": ibit, "BTC-USD": np.linspace(100, 40, 60), "SPY": np.linspace(500, 450, 60)}, index=idx
    )
    a = Analytics(DummyProvider(closes), [Holding("IBIT", 10, None)], {}).load()
    crash = [
        {
            "name": "Test",
            "start": idx[5].date().isoformat(),
            "end": idx[30].date().isoformat(),
            "sp500_move": -0.1,
        }
    ]
    (replay,) = crash_replays(a, crash, betas={"IBIT": 0.5})
    (holding,) = replay["holdings"]
    btc = closes["BTC-USD"].iloc[30] / closes["BTC-USD"].iloc[5] - 1
    assert holding["basis"] == "via BTC-USD" and holding["move_pct"] == round(btc * 100, 1)
    assert replay["estimated_holdings"] == []


def test_crash_replay_falls_back_to_reference_market_move(provider, analytics):
    replays = analytics.stress()["crash_replays"]
    covid = next(r for r in replays if r["name"] == "2020 Covid crash")
    assert covid["market_source"].startswith("S&P 500 index") and covid["market_move_pct"] == -33.9
    assert set(covid["estimated_holdings"]) == set(analytics.symbols)  # mock data doesn't go back to 2020


def test_move_needs_trading_through_the_whole_period():
    prices = frame({"A": [10.0, 9.0, 8.0, 12.0]})
    d = prices.index
    assert move(prices, "A", d[0], d[2]) == pytest.approx(-0.2)
    assert move(prices, "A", d[0] - pd.Timedelta(days=30), d[2]) is None  # starts before the data
    assert move(prices, "B", d[0], d[2]) is None


def test_risk_report_has_the_new_measures(analytics):
    r = analytics.risk()
    assert r["bad_day_losses"]["1_in_20"]["loss_dollars"] < 0
    assert r["bad_week_losses"]["1_in_20"]["loss_dollars"] < r["bad_day_losses"]["1_in_20"]["loss_dollars"]
    assert r["downside_beta_to_spy"] > 0 and r["volatility_recent_pct"] > 0
    assert r["max_drawdown_full_history"]["pct"] <= r["max_drawdown_1y"]["pct"]
    assert [d["market_move_pct"] for d in r["market_drops"]] == [-10.0, -20.0, -35.0]
    assert r["data_quality"] == []
