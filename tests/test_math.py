"""Validates Desk Note's maths against answers worked out by hand or planted in
the data, not against the code's own output.

Each test builds a tiny dummy portfolio from prices chosen so the expected
result can be checked with pencil and paper (shown in the comments).
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.analytics import TRADING_DAYS, Analytics
from app.data import DataProvider
from app.data.mock import STOCKS as MOCK_STOCKS
from app.data.yahoo import implied_move
from app.portfolio import Holding, Trade, replay


class DummyProvider(DataProvider):
    """Serves fixed closes. ``sectors`` maps symbol -> Yahoo sector name."""

    name = "dummy"
    is_mock = True

    def __init__(self, closes: pd.DataFrame, sectors: dict[str, str] | None = None):
        self.closes = closes
        self.sectors = sectors or {}

    def as_of(self) -> date:
        return self.closes.index[-1].date()

    def prices(self, symbols, start):
        return self.closes.reindex(columns=[s.upper() for s in symbols])

    def profile(self, symbol):
        return {"symbol": symbol, "name": symbol, "sector": self.sectors.get(symbol), "quote_type": "EQUITY"}

    def news(self, symbol, days=14):
        return []

    def earnings(self, symbol):
        return None


def frame(columns: dict[str, list[float]], end: str = "2026-10-07") -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=pd.bdate_range(end=end, periods=n))


def prices_from_returns(returns: np.ndarray, end_price: float = 100.0) -> list[float]:
    """Price path whose day-on-day returns are exactly ``returns`` (first is ignored)."""
    growth = np.cumprod(1 + np.asarray(returns, dtype=float))
    growth = growth / growth[0]
    return list(end_price * growth / growth[-1])


# ---------------------------------------------------------------------------
# 1-2. Daily P/L, cost basis, time-weighted return: a four-day hand example
# ---------------------------------------------------------------------------
#
#   day   A price   SPY    shares held at close
#   d0      100     400         10   (opening position, cost $90/share)
#   d1      102     404         15   (bought 5 at $102 at the close)
#   d2       99     400         15
#   d3      105     408         15
#
#   P/L:  d1 = 10 x (102-100) = +20
#         d2 = 15 x ( 99-102) = -45
#         d3 = 15 x (105- 99) = +90          total = +65
#   Value: 1000 -> 1530 -> 1485 -> 1575       (+575 = 65 gain + 510 bought)
#   Daily returns: 20/1000 = 2%, -45/1530, 90/1485
#   Time-weighted: 1.02 x (1485/1530) x (1575/1485) - 1 = 1.02 x 0.97059 x 1.06061 - 1 = 5.0%
#   SPY: 408/400 - 1 = 2.0%   -> beat the market by 3.0 points
#   Average cost after the buy: (10 x 90 + 5 x 102) / 15 = 94
#   Unrealised: 15 x 105 - 15 x 94 = 165

HAND = frame({"A": [100, 102, 99, 105], "SPY": [400, 404, 400, 408]})
D1 = HAND.index[1].date()


def hand_portfolio(scale: float = 1.0, extra_trades=()) -> Analytics:
    trades = [Trade(D1, "A", "buy", 5 * scale, 102.0, id=1), *extra_trades]
    return Analytics(DummyProvider(HAND), [Holding("A", 10 * scale, 90.0)], {}, trades).load()


def test_daily_pnl_by_hand():
    a = hand_portfolio()
    assert list(a.pnl.round(9)) == [0, 20, -45, 90]
    assert list(a.port.round(9)) == [1000, 1530, 1485, 1575]


def test_performance_by_hand():
    perf = hand_portfolio().performance("1w")  # clipped to the 3 available sessions
    assert perf["change_dollars"] == 65
    assert perf["net_purchases_dollars"] == 510
    assert perf["change_pct"] == 5.0
    assert perf["market_spy_pct"] == 2.0
    assert perf["vs_market_pct_points"] == 3.0


def test_overview_by_hand():
    o = hand_portfolio().overview()
    assert o["total_value"] == 1575
    assert o["day_change"] == {"dollars": 90, "pct": 6.1}  # 90 / 1485
    a = o["positions"][0]
    assert a["shares"] == 15 and a["unrealised_pl"] == 165  # 15 x (105 - 94)
    assert o["unrealised_pl"]["cost_basis_total"] == 1410  # 15 x 94


def test_scaling_positions_scales_dollars_not_percent():
    base, double = hand_portfolio().performance("1w"), hand_portfolio(scale=2).performance("1w")
    assert double["change_dollars"] == 2 * base["change_dollars"]
    assert double["change_pct"] == base["change_pct"]


def test_round_trip_trade_changes_nothing():
    d2 = HAND.index[2].date()
    churn = [Trade(d2, "A", "buy", 7, 99.0, id=2), Trade(d2, "A", "sell", 7, 99.0, id=3)]
    assert hand_portfolio(extra_trades=churn).performance("1w") == hand_portfolio().performance("1w")


def test_without_trades_return_is_simple_value_change():
    a = Analytics(DummyProvider(HAND), [Holding("A", 10, 90.0)], {}).load()
    assert a.performance("1w")["change_pct"] == 5.0  # 105/100 - 1
    assert a.performance("1w")["change_dollars"] == 50


def test_average_cost_and_realised_by_hand():
    book = replay(
        [Holding("A", 10, 90.0)],
        [
            Trade(date(2026, 1, 2), "A", "buy", 5, 102.0),  # avg (900 + 510) / 15 = 94
            Trade(date(2026, 1, 5), "A", "sell", 6, 105.0),  # realised (105 - 94) x 6 = 66
        ],
    )
    assert book["A"].avg_cost == pytest.approx(94)
    assert book["A"].realised == pytest.approx(66)
    assert book["A"].shares == 9


def test_attribution_total_matches_hand_pnl():
    df, _ = hand_portfolio().attribution_frame("1w")
    assert df["total"].sum() == pytest.approx(65)


# ---------------------------------------------------------------------------
# 3. Attribution betas: planted answers
# ---------------------------------------------------------------------------


def planted(noise: float = 0.0, n: int = 320, seed: int = 1):
    """STOCK return = 0.0002 + 1.5 x SPY + 0.8 x (XLK - SPY) + noise, exactly."""
    rng = np.random.default_rng(seed)
    m = rng.normal(0.0004, 0.01, n)
    e = rng.normal(0, 0.006, n)
    r = 0.0002 + 1.5 * m + 0.8 * e + rng.normal(0, noise, n) * (noise > 0)
    closes = frame(
        {
            "STOCK": prices_from_returns(r),
            "SPY": prices_from_returns(m, 500),
            "XLK": prices_from_returns(m + e, 200),
        }
    )
    return Analytics(DummyProvider(closes, {"STOCK": "Technology"}), [Holding("STOCK", 100, None)], {}).load()


def test_regression_recovers_exact_planted_betas():
    a = planted(noise=0.0)
    df, w = a.attribution_frame("1w")
    row = df.loc["STOCK"]
    assert not row["fallback"]
    assert row["beta_mkt"] == pytest.approx(1.5, abs=1e-9)
    assert row["beta_sector"] == pytest.approx(0.8, abs=1e-9)
    # With no noise, "your pick" is only the daily intercept x yesterday's value.
    prev = a.values["STOCK"].iloc[w.start_pos : w.end_pos]
    assert row["stock_specific"] == pytest.approx(0.0002 * prev.sum(), rel=1e-6)


def test_regression_recovers_noisy_planted_betas():
    row = planted(noise=0.01).attribution_frame("1w")[0].loc["STOCK"]
    # Standard error of b_mkt here is about 0.01 / (0.01 x sqrt(252)) = 0.06.
    assert row["beta_mkt"] == pytest.approx(1.5, abs=0.2)
    assert row["beta_sector"] == pytest.approx(0.8, abs=0.35)


def test_regression_recovers_mock_generator_betas(analytics):
    """The mock market is generated from known betas; each fitted beta should be
    within 3 standard errors of the value it was generated with."""
    df, w = analytics.attribution_frame("1w")
    r = analytics.returns
    rows = slice(w.start_pos - TRADING_DAYS + 1, w.start_pos + 1)
    for sym, row in df.iterrows():
        m = r["SPY"].iloc[rows].to_numpy()
        X = np.column_stack([np.ones_like(m), m, r[analytics.etf_of[sym]].iloc[rows].to_numpy() - m])
        y = r[sym].iloc[rows].to_numpy()
        coef = np.array([row["beta_mkt"], row["beta_sector"]])
        resid = y - X @ np.concatenate([[y.mean() - (X[:, 1:] @ coef).mean()], coef])
        se = np.sqrt(np.diag(resid.var(ddof=3) * np.linalg.inv(X.T @ X)))[1:]
        truth = MOCK_STOCKS[sym]
        assert abs(row["beta_mkt"] - truth.beta_mkt) < 3 * se[0], sym
        assert abs(row["beta_sector"] - truth.beta_sec) < 3 * se[1], sym


def test_fit_matches_independent_ols_including_flat_days():
    """A thinly traded stock that is unchanged on many days: the fit must use
    those zero-return days (true beta ~0.5 here, not ~1.0)."""
    rng = np.random.default_rng(3)
    n = 320
    m = rng.normal(0.0004, 0.01, n)
    trades_today = rng.random(n) < 0.5
    r = np.where(trades_today, m, 0.0)  # moves with the market only half the time
    closes = frame({"THIN": prices_from_returns(r), "SPY": prices_from_returns(m, 500)})
    a = Analytics(DummyProvider(closes), [Holding("THIN", 100, None)], {}).load()
    w = a.window("1w")
    fit = a.fit_betas("THIN", w)
    rr = a.returns.iloc[max(1, w.start_pos - TRADING_DAYS + 1) : w.start_pos + 1]
    X = np.column_stack([np.ones(len(rr)), rr["SPY"]])
    expected = np.linalg.solve(X.T @ X, X.T @ rr["THIN"].to_numpy())  # normal equations
    assert fit["beta_mkt"] == pytest.approx(expected[1], abs=1e-9)
    assert 0.35 < fit["beta_mkt"] < 0.65


def test_fit_ignores_days_before_listing():
    rng = np.random.default_rng(4)
    n = 320
    m = rng.normal(0.0004, 0.01, n)
    new = np.array(prices_from_returns(1.2 * m))
    new[:150] = np.nan  # listed on day 150
    closes = frame({"NEW": list(new), "SPY": prices_from_returns(m, 500)})
    a = Analytics(DummyProvider(closes), [Holding("NEW", 10, None)], {}).load()
    fit = a.fit_betas("NEW", a.window("1w"))
    assert fit["beta_mkt"] == pytest.approx(1.2, abs=1e-9)  # exact: flat pre-listing days excluded


# ---------------------------------------------------------------------------
# 4. Risk
# ---------------------------------------------------------------------------


def symmetric_pair():
    """Two stocks with equal volatility and zero correlation, equal weights.
    x = s(+,-,+,-...), y = s(+,+,-,-...): orthogonal, same variance.
    SPY = (x + y) / 2, so each stock's beta is exactly 1."""
    s = 0.01
    k = np.arange(TRADING_DAYS + 1)
    x = s * np.where(k % 2 == 0, 1, -1)
    y = s * np.where(k % 4 < 2, 1, -1)
    closes = frame(
        {
            "X": prices_from_returns(x),
            "Y": prices_from_returns(y),
            "SPY": prices_from_returns((x + y) / 2, 500),
        }
    )
    return Analytics(DummyProvider(closes), [Holding("X", 10, None), Holding("Y", 10, None)], {}).load(), x, y


def test_risk_shares_split_evenly_for_symmetric_pair():
    a, _, _ = symmetric_pair()
    risk = a.risk()
    assert [p["risk_share_pct"] for p in risk["risk_contributions"]] == [50.0, 50.0]
    assert risk["effective_number_of_positions"] == 2.0
    assert risk["beta_to_spy"] == 1.0


def test_volatility_by_hand():
    a, x, y = symmetric_pair()
    # Equal-weight portfolio = (x + y) / 2; its sample std x sqrt(252).
    port = ((x + y) / 2)[1:]
    expected = np.std(port, ddof=1) * np.sqrt(TRADING_DAYS) * 100
    assert a.risk()["volatility_annual_pct"] == pytest.approx(expected, abs=0.05)


def test_portfolio_beta_is_weighted_sum_of_betas(analytics):
    rets = analytics.returns[analytics.symbols].iloc[-TRADING_DAYS:]
    spy = analytics.returns["SPY"].iloc[-TRADING_DAYS:]
    betas = rets.apply(lambda r: np.cov(r, spy)[0, 1] / np.var(spy, ddof=1))
    weights = analytics.values.iloc[-1][analytics.symbols]
    weights = weights / weights.sum()
    assert analytics.risk()["beta_to_spy"] == pytest.approx(float((betas * weights).sum()), abs=0.006)


def test_effective_positions_equal_weights():
    closes = frame({s: [100.0] * 60 for s in ["A", "B", "C", "D"]} | {"SPY": list(np.linspace(400, 420, 60))})
    closes.iloc[::2, :4] *= 1.001  # some movement so covariance isn't singular
    a = Analytics(DummyProvider(closes), [Holding(s, 10, None) for s in "ABCD"], {}).load()
    assert a.risk()["effective_number_of_positions"] == 4.0


def test_max_drawdown_by_hand():
    # 100 -> 120 -> 90 -> 110: worst fall is 120 -> 90 = -25%, -$30 on one share.
    closes = frame({"A": [100, 120, 90, 110], "SPY": [400, 404, 400, 408]})
    dd = Analytics(DummyProvider(closes), [Holding("A", 1, None)], {}).load().risk()["max_drawdown_1y"]
    assert dd["pct"] == -25.0 and dd["dollars"] == -30
    assert dd["peak_date"] == closes.index[1].date().isoformat()
    assert dd["trough_date"] == closes.index[2].date().isoformat()
    assert dd["recovered"] is False


def test_stress_test_by_hand():
    # Stock moves exactly 1.5x SPY -> beta 1.5. 100 shares ending at $100 = $10,000.
    # SPY -10% -> 1.5 x -10% x $10,000 = -$1,500.
    rng = np.random.default_rng(5)
    m = rng.normal(0, 0.01, TRADING_DAYS + 1)
    closes = frame({"A": prices_from_returns(1.5 * m, 100), "SPY": prices_from_returns(m, 500)})
    stress = (
        Analytics(DummyProvider(closes), [Holding("A", 100, None)], {})
        .load()
        .risk()["stress_test_spy_down_10pct"]
    )
    assert stress["by_holding"][0]["beta"] == 1.5
    assert stress["dollars"] == -1500 and stress["pct"] == -15.0


# ---------------------------------------------------------------------------
# 5. Options-implied earnings move
# ---------------------------------------------------------------------------


def fake_ticker(expiries):
    calls = pd.DataFrame(
        {
            "strike": [95, 100, 105],
            "bid": [6.0, 3.0, 1.0],
            "ask": [6.4, 3.2, 1.2],
            "lastPrice": [6.2, 3.1, 1.1],
        }
    )
    puts = pd.DataFrame(
        {
            "strike": [95, 100, 105],
            "bid": [1.0, 2.8, 6.0],
            "ask": [1.2, 3.0, 6.4],
            "lastPrice": [1.1, 2.9, 6.2],
        }
    )
    return SimpleNamespace(
        options=expiries,
        option_chain=lambda e: SimpleNamespace(calls=calls, puts=puts),
        history=lambda **k: pd.DataFrame({"Close": [99.0, 100.4]}),
    )


def test_implied_move_by_hand():
    # Spot 100.4 -> nearest strike 100. Call mid 3.1 + put mid 2.9 = 6.0; 6.0 / 100.4 = 5.98% -> 6.0
    move = implied_move(fake_ticker(("2026-10-09", "2026-10-16")), date(2026, 10, 8))
    assert move == 6.0


def test_implied_move_ignores_missing_latest_price():
    ticker = fake_ticker(("2026-10-09",))
    ticker.history = lambda **k: pd.DataFrame({"Close": [100.4, float("nan")]})
    assert implied_move(ticker, date(2026, 10, 8)) == 6.0


def test_implied_move_needs_expiry_soon_after_earnings():
    assert implied_move(fake_ticker(("2026-10-30",)), date(2026, 10, 8)) is None


# ---------------------------------------------------------------------------
# 7. Journal review
# ---------------------------------------------------------------------------


def test_journal_market_adjusted_move_by_hand():
    """MKT moves exactly with SPY (beta 1); HOT moves 2x SPY (beta 2).
    Since the entry, SPY rose g%. Market-adjusted: MKT 0%, HOT (1+r)^2-ish minus 2x."""
    rng = np.random.default_rng(6)
    n = 320
    m = rng.normal(0.0005, 0.01, n)
    closes = frame(
        {"MKT": prices_from_returns(m), "HOT": prices_from_returns(2 * m), "SPY": prices_from_returns(m, 500)}
    )
    a = Analytics(DummyProvider(closes), [Holding("MKT", 10, 50.0), Holding("HOT", 10, 50.0)], {}).load()
    when = closes.index[-21].date().isoformat()
    review = a.review_journal(
        [
            {
                "id": 1,
                "date": when,
                "symbol": "MKT",
                "kind": "buy_reason",
                "text": "index-like",
                "trade": {"side": "buy", "shares": 10, "price": float(closes["MKT"].iloc[-21])},
            },
            {"id": 2, "date": when, "symbol": "HOT", "kind": "buy_reason", "text": "levered"},
        ]
    )
    mkt, hot = (e["outcome"] for e in review["entries"])
    assert mkt["beta"] == 1.0 and mkt["market_adjusted_pct"] == 0.0
    assert mkt["stock_pct"] == mkt["spy_pct"]
    # Gain since buy = 10 shares x (price now - price paid)
    assert mkt["gain_since_buy_dollars"] == round(10 * (closes["MKT"].iloc[-1] - closes["MKT"].iloc[-21]))
    # HOT: beta 2; adjusted = stock - 2 x SPY, which is small but not zero (compounding).
    assert hot["beta"] == 2.0
    stock = closes["HOT"].iloc[-1] / closes["HOT"].iloc[-21] - 1
    spy = closes["SPY"].iloc[-1] / closes["SPY"].iloc[-21] - 1
    assert hot["market_adjusted_pct"] == pytest.approx(round((stock - 2 * spy) * 100, 1), abs=0.05)
