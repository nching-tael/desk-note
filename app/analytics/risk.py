"""Risk report: volatility, concentration, correlation, drawdown and a stress
test, using one year of daily returns at today's weights."""

from __future__ import annotations

import numpy as np

from ..data import MARKET
from .periods import TRADING_DAYS, pct, usd

PAIR_THRESHOLD = 0.7  # correlated pairs worth listing
GROUP_THRESHOLD = 0.6  # holdings this correlated are grouped together


def report(a):
    a.load()
    symbols = a.symbols
    weights = a.weights()
    w = weights.to_numpy()
    returns = a.returns[symbols].iloc[-TRADING_DAYS:]
    spy = a.returns[MARKET].iloc[-TRADING_DAYS:]
    total_value = float(a.port.iloc[-1])

    cov = returns.cov().to_numpy() * TRADING_DAYS
    variance = w @ cov @ w
    volatility = float(np.sqrt(variance))
    risk_share = w * (cov @ w) / variance  # each holding's share of the total, sums to 1
    spy_var = spy.var()

    def beta(r):
        return float(r.cov(spy) / spy_var)

    largest = weights.sort_values(ascending=False)
    sector_weights = weights.groupby(a.sector_of).sum().sort_values(ascending=False)
    corr = returns.corr()

    return {
        "as_of": a.as_of,
        "total_value": usd(total_value),
        "volatility_annual_pct": pct(volatility),
        "typical_daily_move_dollars": usd(total_value * volatility / np.sqrt(TRADING_DAYS)),
        "market_volatility_annual_pct": pct(spy.std() * np.sqrt(TRADING_DAYS)),
        "beta_to_spy": round(beta(returns @ weights), 2),
        "effective_number_of_positions": round(float(1 / (w**2).sum()), 1),
        "number_of_positions": len(symbols),
        "largest_position": {"symbol": largest.index[0], "weight_pct": pct(largest.iloc[0])},
        "top3_weight_pct": pct(largest.head(3).sum()),
        "risk_contributions": sorted(
            [
                {"symbol": s, "weight_pct": pct(weights[s]), "risk_share_pct": pct(r)}
                for s, r in zip(symbols, risk_share, strict=True)
            ],
            key=lambda p: -p["risk_share_pct"],
        ),
        "sector_weights_pct": {name: pct(v) for name, v in sector_weights.items()},
        "correlated_pairs_above_0_7": correlated_pairs(corr)[:10],
        "correlated_groups": correlated_groups(a, corr, weights, total_value),
        "max_drawdown_1y": max_drawdown(a),
        "stress_test_spy_down_10pct": stress_test(a, {s: beta(returns[s]) for s in symbols}, total_value),
        "method": "One year of daily returns at current weights. Risk share = w_i*(Σw)_i / wᵀΣw.",
    }


def correlated_pairs(corr):
    symbols = list(corr.index)
    pairs = []
    for i, a in enumerate(symbols):
        for b in symbols[i + 1 :]:
            if corr.loc[a, b] > PAIR_THRESHOLD:
                pairs.append({"pair": [a, b], "correlation": round(float(corr.loc[a, b]), 2)})
    return sorted(pairs, key=lambda p: -p["correlation"])


def correlated_groups(a, corr, weights, total_value):
    """Holdings linked by a chain of correlations above GROUP_THRESHOLD, e.g.
    four chip stocks that all move together."""
    groups = []
    unvisited = set(corr.index)
    while unvisited:
        group, stack = set(), [unvisited.pop()]
        while stack:
            symbol = stack.pop()
            group.add(symbol)
            linked = {s for s in unvisited if corr.loc[symbol, s] >= GROUP_THRESHOLD}
            unvisited -= linked
            stack.extend(linked)
        if len(group) > 1:
            groups.append(sorted(group, key=lambda s: -weights[s]))

    result = []
    for members in groups:
        pair_corrs = corr.loc[members, members].to_numpy()[np.triu_indices(len(members), 1)]
        weight = weights[members].sum()
        result.append(
            {
                "symbols": members,
                "sectors": sorted({a.sector_of[s] for s in members}),
                "weight_pct": pct(weight),
                "value_dollars": usd(weight * total_value),
                "average_correlation": round(float(pair_corrs.mean()), 2),
            }
        )
    return sorted(result, key=lambda g: -g["weight_pct"])


def max_drawdown(a):
    """Biggest peak-to-trough fall over the past year, at today's share counts."""
    value = (a.closes[a.symbols] * a.shares[a.symbols]).sum(axis=1).iloc[-(TRADING_DAYS + 1) :]
    drawdown = value / value.cummax() - 1
    trough = drawdown.idxmin()
    peak = value[:trough].idxmax()
    return {
        "pct": pct(drawdown.min()),
        "dollars": usd(value[trough] - value[peak]),
        "peak_date": peak.date().isoformat(),
        "trough_date": trough.date().isoformat(),
        "recovered": bool(value.iloc[-1] >= value[peak]),
    }


def stress_test(a, betas, total_value):
    """Estimated $ impact of the market falling 10%, using each holding's beta."""
    rows = []
    for symbol, beta in betas.items():
        value = float(a.values[symbol].iloc[-1])
        rows.append({"symbol": symbol, "beta": round(beta, 2), "dollars": usd(value * beta * -0.10)})
    total = sum(float(a.values[s].iloc[-1]) * b * -0.10 for s, b in betas.items())
    return {
        "dollars": usd(total),
        "pct": pct(total / total_value),
        "by_holding": sorted(rows, key=lambda r: r["dollars"]),
    }
