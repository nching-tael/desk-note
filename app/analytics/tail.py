"""Risk measured from history rather than assumed: how bad a bad day or week
has been, how holdings behave when the market falls, and how much to trust
the numbers.

Everything here works on today's holdings at today's weights, replayed over
the price history that's loaded (about two years).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..data import MARKET
from .periods import TRADING_DAYS, pct, usd

MIN_DAYS = 40  # fewer real days than this and a beta isn't worth fitting
EWMA_DECAY = 0.94  # RiskMetrics' daily decay: recent days count most
STALE_DAYS = 3  # this many unchanged closes in a row while the market moved


def holding_returns(a):
    """Daily returns of today's holdings over the loaded history.

    Before a holding started trading its price is held flat, which would make
    a young fund look calmer than it is. Those days are filled with its beta
    times the market's return instead, and reported as estimated."""
    market = a.returns[MARKET]
    returns, estimated = {}, {}
    for symbol in a.symbols:
        r = a.returns[symbol].copy()
        first = a.listed_pos.get(symbol, 0)  # index of the first real price
        if first > 0:
            real, m = r.iloc[first + 1 :], market.iloc[first + 1 :]
            beta = real.cov(m) / m.var() if len(real) >= MIN_DAYS else 1.0
            r.iloc[1 : first + 1] = beta * market.iloc[1 : first + 1]
            estimated[symbol] = first
        returns[symbol] = r
    return pd.DataFrame(returns).iloc[1:], estimated  # day 0 has no previous close


def portfolio_returns(a):
    returns, estimated = holding_returns(a)
    return returns @ a.weights(), estimated


def losses(daily_returns, value, horizon=1):
    """Historical value at risk and expected shortfall.

    "1 in 20": the loss on the k-th worst period out of n, where k = n/20
    rounded up, and the average of those k worst periods. No bell curve is
    assumed; it's what actually happened to this mix of holdings."""
    r = daily_returns
    if horizon > 1:
        r = (1 + r).rolling(horizon).apply(np.prod, raw=True).dropna() - 1  # overlapping periods
    ordered = np.sort(r.to_numpy())
    result = {"periods": len(ordered)}
    if len(ordered) == 0:
        result["note"] = "Not enough history for this horizon."
        return result
    for label, tail in (("1_in_20", 0.05), ("1_in_100", 0.01)):
        k = max(1, math.ceil(tail * len(ordered)))
        worst = ordered[:k]
        result[label] = {
            "loss_dollars": usd(value * worst[-1]),
            "loss_pct": pct(worst[-1]),
            "average_of_worst_dollars": usd(value * worst.mean()),
            "worst_periods_used": k,
        }
    return result


def downside_beta(r, market):
    """Slope of a holding's returns on the market's, using only days the
    market fell. Many holdings drop harder in sell-offs than their everyday
    beta suggests."""
    down = market < 0
    if down.sum() < MIN_DAYS:
        return float(r.cov(market) / market.var())
    x, y = market[down], r[down]
    return float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())


def recent_volatility(daily_returns, decay=EWMA_DECAY):
    """Annualised volatility with recent days weighted most (exponential decay)."""
    r = daily_returns.to_numpy()
    variance = r[:30].var()
    for x in r:
        variance = decay * variance + (1 - decay) * x * x
    return math.sqrt(variance * TRADING_DAYS)


def worst_periods(daily_returns, value):
    worst = {}
    for label, days in (("day", 1), ("week", 5), ("month", 21)):
        if len(daily_returns) < days:
            continue
        r = daily_returns if days == 1 else (1 + daily_returns).rolling(days).apply(np.prod, raw=True) - 1
        end = r.idxmin()
        start = daily_returns.index[max(0, daily_returns.index.get_loc(end) - days + 1)]
        worst[label] = {
            "dollars": usd(value * r[end]),
            "pct": pct(r[end]),
            "from": start.date().isoformat(),
            "to": end.date().isoformat(),
        }
    return worst


def drawdown(daily_returns, value, start):
    """Largest peak-to-trough fall of today's holdings, how long it took to get
    back, and what a fall that size would cost at today's value. start is the
    date of the close the returns are measured from."""
    path = pd.concat([pd.Series([1.0], index=[start]), (1 + daily_returns).cumprod()])
    falls = path / path.cummax() - 1
    trough = falls.idxmin()
    peak = path[:trough].idxmax()
    after = path[trough:]
    recovered = after[after >= path[peak] * (1 - 1e-9)]  # tolerate float rounding at the old peak
    result = {
        "pct": pct(falls.min()),
        "dollars_at_todays_value": usd(value * falls.min()),
        "peak_date": peak.date().isoformat(),
        "trough_date": trough.date().isoformat(),
        "sessions_down": int(path.index.get_loc(trough) - path.index.get_loc(peak)),
        "recovered": not recovered.empty,
    }
    if not recovered.empty:
        back = recovered.index[0]
        result["recovered_date"] = back.date().isoformat()
        result["sessions_to_recover"] = int(path.index.get_loc(back) - path.index.get_loc(trough))
    return result


def data_quality(a, estimated):
    """Plain-language warnings about anything that makes the numbers less reliable."""
    notes = []
    sessions = len(a.closes) - 1
    for symbol, first in estimated.items():
        real = sessions - first
        notes.append(
            f"{symbol} has {real} sessions of real price history; the {first} before that are "
            "estimated from its market beta."
        )
    market_moved = a.returns[MARKET].iloc[-10:].abs() > 1e-6
    for symbol in a.symbols:
        unchanged = (a.returns[symbol].iloc[-10:].abs() < 1e-9) & market_moved
        streak = 0
        for flat in reversed(unchanged.to_list()):
            if not flat:
                break
            streak += 1
        if streak >= STALE_DAYS:
            notes.append(
                f"{symbol}'s price hasn't changed for {streak} sessions while the market moved; "
                "it may be stale."
            )
    if sessions < TRADING_DAYS:
        notes.append(f"Only {sessions} sessions of history are loaded, so tail estimates are rough.")
    return notes
