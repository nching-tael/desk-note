"""Recorded trades and the decision journal review."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import MARKET
from .periods import TRADING_DAYS, pct, usd

MIN_BETA_DAYS = 40


def trades_report(a, symbol=None):
    a.load()
    symbol = symbol.strip().upper() if symbol else None
    trades = [t for t in a.trades if symbol in (None, t.symbol)]
    return {
        "trades": [
            {
                "id": t.id,
                "date": t.date.isoformat(),
                "symbol": t.symbol,
                "side": t.side,
                "shares": t.shares,
                "price": t.price,
                "value_dollars": usd(t.shares * t.price) if t.price else None,
            }
            for t in trades[-50:]
        ],
        "count": len(trades),
        "realised_pl_dollars": {s: usd(v) for s, v in a.realised.items() if symbol in (None, s)},
        "note": "Opening positions from portfolio.csv aren't trades; they're assumed held throughout.",
    }


def review(a, entries):
    """How each journaled decision has worked out since it was made: the
    stock's move, the market's, and the market-adjusted move
    (stock - beta x SPY). For trades, also the $ effect on the shares traded."""
    a.load()
    reviewed, buys, sells = [], [], []

    for entry in entries:
        item = {key: entry.get(key) for key in ("id", "date", "symbol", "kind")}
        item["text"] = (entry.get("text") or "")[:300]
        if entry.get("trade"):
            item["trade"] = entry["trade"]

        outcome, problem = since_decision(a, entry)
        item["outcome"] = outcome
        if problem:
            item["outcome_note"] = problem
        elif entry.get("kind") == "buy_reason":
            buys.append(outcome["market_adjusted_pct"])
        elif entry.get("kind") == "sell_reason":
            sells.append(outcome["market_adjusted_pct"])
        reviewed.append(item)

    summary = {"entries": len(reviewed)}
    if buys:
        summary["buys_avg_market_adjusted_pct"] = round(float(np.mean(buys)), 1)
        summary["buys_beating_market_adjusted"] = f"{sum(b > 0 for b in buys)} of {len(buys)}"
    if sells:
        summary["sells_avg_market_adjusted_pct_since"] = round(float(np.mean(sells)), 1)
        summary["sells_followed_by_underperformance"] = f"{sum(s < 0 for s in sells)} of {len(sells)}"

    return {
        "as_of": a.as_of,
        "summary": summary,
        "entries": reviewed,
        "how_to_read": (
            "market_adjusted_pct = the stock's move minus beta x the S&P 500's move since the "
            "entry. For a buy, positive is good. For a sell, negative means the stock lagged "
            "after you sold (good timing); move_since_sale_dollars > 0 is gain you gave up."
        ),
    }


def since_decision(a, entry):
    """Returns (outcome, None) or (None, reason it can't be evaluated)."""
    symbol = entry.get("symbol")
    if not symbol:
        return None, "No symbol."
    prices = a.price_series(symbol)
    if prices is None:
        return None, f"No price data for {symbol}."

    index = a.closes.index
    when = pd.Timestamp(entry["date"])
    pos = int(index.searchsorted(when))
    if when < index[0] or pos >= len(index) - 1:
        return None, "Too recent or outside the price history to evaluate."

    stock_return = prices.iloc[-1] / prices.iloc[pos] - 1
    spy = a.closes[MARKET]
    market_return = spy.iloc[-1] / spy.iloc[pos] - 1

    # beta from the year before the decision
    history = slice(max(1, pos - TRADING_DAYS + 1), pos + 1)
    stock_daily = prices.pct_change().iloc[history]
    spy_daily = a.returns[MARKET].iloc[history]
    beta = stock_daily.cov(spy_daily) / spy_daily.var() if len(stock_daily) >= MIN_BETA_DAYS else 1.0

    outcome = {
        "from": index[pos].date().isoformat(),
        "sessions": len(index) - 1 - pos,
        "stock_pct": pct(stock_return),
        "spy_pct": pct(market_return),
        "beta": round(float(beta), 2),
        "market_adjusted_pct": pct(stock_return - beta * market_return),
    }

    trade = entry.get("trade") or {}
    if trade.get("price") and trade.get("shares"):
        move = usd(trade["shares"] * (prices.iloc[-1] - trade["price"]))
        if trade.get("side") == "buy":
            outcome["gain_since_buy_dollars"] = move
        else:
            outcome["move_since_sale_dollars"] = move
    return outcome, None
