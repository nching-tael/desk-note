"""Chart data for the UI (or for Claude to draw): labels plus series."""

from __future__ import annotations

import math

from ..data import MARKET
from .periods import AnalyticsError, usd


def chart(a, kind, symbol=None, period=None):
    kind = (kind or "").strip().lower()
    if kind == "portfolio_vs_market":
        return portfolio_vs_market(a, period or "1m")
    if kind == "stock_vs_market":
        if not symbol:
            raise AnalyticsError("stock_vs_market needs a symbol.")
        return stock_vs_market(a, symbol, period or "1m")
    if kind == "attribution":
        return attribution(a, period or "1w")
    raise AnalyticsError("Unknown chart kind. Use portfolio_vs_market, stock_vs_market or attribution.")


def dates(a, w):
    return [d.date().isoformat() for d in a.closes.index[w.start_pos : w.end_pos + 1]]


def portfolio_vs_market(a, period="1m"):
    w = a.window(period)
    start = float(a.port.iloc[w.start_pos])
    # Grow the starting value by the time-weighted return so buying and
    # selling don't show up as jumps.
    growth = (1 + a.port_ret.iloc[w.days]).cumprod()
    portfolio = [start] + list(start * growth)
    spy = a.closes[MARKET].iloc[w.start_pos : w.end_pos + 1]
    market = spy / spy.iloc[0] * start
    return {
        "kind": "portfolio_vs_market",
        "type": "line",
        "title": f"Your portfolio vs the S&P 500, {w.label}",
        "unit": "$",
        "labels": dates(a, w),
        "series": [
            {"name": "Your portfolio", "role": "primary", "data": [usd(x) for x in portfolio]},
            {
                "name": "S&P 500 (SPY), same start value",
                "role": "benchmark",
                "data": [usd(x) for x in market],
            },
        ],
    }


def stock_vs_market(a, symbol, period="1m"):
    w = a.window(period)
    symbol = symbol.strip().upper()
    prices = a.price_series(symbol)
    if prices is None:
        raise AnalyticsError(f"No price data for {symbol}.")

    def rebased(series):
        series = series.iloc[w.start_pos : w.end_pos + 1]
        base = series.dropna().iloc[0]
        return [round((x / base - 1) * 100, 2) if not math.isnan(x) else None for x in series]

    return {
        "kind": "stock_vs_market",
        "type": "line",
        "title": f"{symbol} vs the S&P 500, {w.label}",
        "unit": "%",
        "labels": dates(a, w),
        "series": [
            {"name": symbol, "role": "primary", "data": rebased(prices)},
            {"name": "S&P 500 (SPY)", "role": "benchmark", "data": rebased(a.closes[MARKET])},
        ],
    }


def attribution(a, period="1w"):
    df, w = a.attribution_frame(period)
    df = df.loc[df["total"].abs().sort_values(ascending=False).index]
    return {
        "kind": "attribution",
        "type": "bar",
        "stacked": True,
        "title": f"What drove each holding, {w.label}",
        "unit": "$",
        "labels": df.index.tolist(),
        "series": [
            {"name": "Market", "role": "market", "data": [usd(x) for x in df["market"]]},
            {"name": "Sector", "role": "sector", "data": [usd(x) for x in df["sector_move"]]},
            {"name": "Stock-specific", "role": "specific", "data": [usd(x) for x in df["stock_specific"]]},
        ],
    }
