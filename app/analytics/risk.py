"""Risk report: volatility, concentration, correlation, tail losses,
drawdowns and stress tests, at today's weights."""

from __future__ import annotations

import numpy as np

from ..data import MARKET
from .exposure import look_through
from .periods import TRADING_DAYS, pct, usd
from .stress import market_drops
from .tail import (
    data_quality,
    downside_beta,
    drawdown,
    holding_returns,
    losses,
    recent_volatility,
    worst_periods,
)

PAIR_THRESHOLD = 0.7  # correlated pairs worth listing
GROUP_THRESHOLD = 0.6  # holdings this correlated are grouped together


def report(a):
    a.load()
    symbols = a.symbols
    weights = a.weights()
    w = weights.to_numpy()
    history, estimated = holding_returns(a)  # young holdings filled from their beta
    portfolio = history @ weights
    market_history = a.returns[MARKET].iloc[1:]
    returns = history.iloc[-TRADING_DAYS:]
    spy = market_history.iloc[-TRADING_DAYS:]
    total_value = float(a.port.iloc[-1])
    betas_down = {s: downside_beta(history[s], market_history) for s in symbols}
    year_start = a.closes.index[max(0, len(a.closes) - TRADING_DAYS - 1)]

    cov = returns.cov().to_numpy() * TRADING_DAYS
    variance = w @ cov @ w
    volatility = float(np.sqrt(variance))
    risk_share = w * (cov @ w) / variance  # each holding's share of the total, sums to 1
    spy_var = spy.var()

    def beta(r):
        return float(r.cov(spy) / spy_var)

    largest = weights.sort_values(ascending=False)
    corr = returns.corr()

    return {
        "as_of": a.as_of,
        "total_value": usd(total_value),
        "volatility_annual_pct": pct(volatility),
        "volatility_recent_pct": pct(recent_volatility(portfolio)),
        "typical_daily_move_dollars": usd(total_value * volatility / np.sqrt(TRADING_DAYS)),
        "market_volatility_annual_pct": pct(spy.std() * np.sqrt(TRADING_DAYS)),
        "beta_to_spy": round(beta(returns @ weights), 2),
        "downside_beta_to_spy": round(downside_beta(portfolio, market_history), 2),
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
        # looked through funds, so VOO counts as its sectors rather than "Diversified fund"
        "sector_weights_pct": look_through(a)["sector_exposure_pct"],
        "correlated_pairs_above_0_7": correlated_pairs(corr)[:10],
        "correlated_groups": correlated_groups(a, corr, weights, total_value),
        "bad_day_losses": losses(portfolio, total_value),
        "bad_week_losses": losses(portfolio, total_value, horizon=5),
        "worst_periods": worst_periods(portfolio, total_value),
        "max_drawdown_1y": drawdown(portfolio.iloc[-TRADING_DAYS:], total_value, year_start),
        "max_drawdown_full_history": drawdown(portfolio, total_value, a.closes.index[0]),
        "stress_test_spy_down_10pct": stress_test(a, betas_down, total_value),
        "market_drops": market_drops(a, betas_down),
        "history": {
            "from": portfolio.index[0].date().isoformat(),
            "sessions": len(portfolio),
            "estimated_holdings": sorted(estimated),
        },
        "data_quality": data_quality(a, estimated),
        "method": (
            "Today's holdings at today's weights. Volatility, beta, correlation and risk share use the last "
            "year of daily returns (risk share = w_i*(Σw)_i / wᵀΣw); recent volatility weights recent days "
            "most. Bad-day and bad-week losses, worst periods and drawdowns come from the full loaded "
            "history, not a bell-curve assumption. Stress tests use downside beta, fitted on days the "
            "market fell."
        ),
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


def stress_test(a, betas, total_value):
    """Estimated $ impact of the market falling 10%, using each holding's
    downside beta."""
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
