"""Stress tests: what today's holdings would lose if the market fell, and what
they did (or would have done) in real crashes."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from ..data import MARKET
from .periods import pct, usd
from .tail import downside_beta, holding_returns

SHOCKS = (-0.10, -0.20, -0.35)

# Peak-to-trough closes of the S&P 500 index. The move is only used when
# market prices for the period aren't available (e.g. the mock data).
CRASHES = [
    {"name": "2008 financial crisis", "start": "2007-10-09", "end": "2009-03-09", "sp500_move": -0.568},
    {"name": "Late-2018 sell-off", "start": "2018-09-20", "end": "2018-12-24", "sp500_move": -0.198},
    {"name": "2020 Covid crash", "start": "2020-02-19", "end": "2020-03-23", "sp500_move": -0.339},
    {"name": "2022 bear market", "start": "2022-01-03", "end": "2022-10-12", "sp500_move": -0.254},
]


# Funds that are younger than the crashes but track something older. Replaying
# what they track is far closer than a beta estimate, especially for bitcoin,
# whose own crashes have little to do with the stock market.
# fmt: off
PROXIES = {
    "VOO": "SPY", "IVV": "SPY", "SPLG": "SPY", "SPYM": "SPY",
    "QQQM": "QQQ",
    "IBIT": "BTC-USD", "FBTC": "BTC-USD", "BITB": "BTC-USD", "ARKB": "BTC-USD", "GBTC": "BTC-USD",
    "ETHA": "ETH-USD", "FETH": "ETH-USD",
    "IAUM": "GLD", "GLDM": "GLD", "IAU": "GLD",
}
# fmt: on


def downside_betas(a):
    returns, _ = holding_returns(a)
    market = a.returns[MARKET].iloc[1:]
    return {s: downside_beta(returns[s], market) for s in a.symbols}


def market_drops(a, betas=None):
    """$ impact of the market falling 10%, 20% and 35%, using each holding's
    downside beta (capped at a total loss)."""
    betas = betas or downside_betas(a)
    values = {s: float(a.values[s].iloc[-1]) for s in a.symbols}
    total = sum(values.values())
    results = []
    for shock in SHOCKS:
        moves = {s: max(-1.0, betas[s] * shock) for s in values}
        loss = sum(values[s] * moves[s] for s in values)
        results.append(
            {
                "market_move_pct": pct(shock),
                "dollars": usd(loss),
                "pct": pct(loss / total),
                "biggest_losers": [
                    {"symbol": s, "dollars": usd(values[s] * moves[s])}
                    for s in sorted(values, key=lambda s: values[s] * moves[s])[:3]
                ],
            }
        )
    return results


def crash_replays(a, crashes=CRASHES, betas=None):
    """Each crash replayed on today's holdings. A holding that traded through
    the whole period uses its actual move; one that didn't exist yet is
    estimated as downside beta x the market's move, and says so."""
    betas = betas or downside_betas(a)
    values = {s: float(a.values[s].iloc[-1]) for s in a.symbols}
    total = sum(values.values())
    earliest = min(date.fromisoformat(c["start"]) for c in crashes) - timedelta(days=10)
    proxies = {s: PROXIES[s] for s in a.symbols if s in PROXIES}
    try:
        prices = a.provider.prices(sorted(set(a.symbols) | set(proxies.values()) | {MARKET}), earliest)
    except Exception:
        prices = pd.DataFrame()

    replays = []
    for crash in crashes:
        start, end = pd.Timestamp(crash["start"]), pd.Timestamp(crash["end"])
        market = move(prices, MARKET, start, end)
        market_source = "actual SPY"
        if market is None:
            market, market_source = crash["sp500_move"], "S&P 500 index (reference)"

        holdings, estimated, loss = [], [], 0.0
        for symbol, value in values.items():
            change, basis = move(prices, symbol, start, end), "actual"
            if change is None and symbol in proxies:
                change, basis = move(prices, proxies[symbol], start, end), f"via {proxies[symbol]}"
            if change is None:
                change, basis = max(-1.0, betas[symbol] * market), "estimated from downside beta"
                estimated.append(symbol)
            loss += value * change
            holdings.append(
                {
                    "symbol": symbol,
                    "move_pct": pct(change),
                    "dollars": usd(value * change),
                    "basis": basis,
                    "estimated": basis.startswith("estimated"),
                }
            )

        replays.append(
            {
                "name": crash["name"],
                "from": crash["start"],
                "to": crash["end"],
                "market_move_pct": pct(market),
                "market_source": market_source,
                "dollars": usd(loss),
                "pct": pct(loss / total),
                "holdings": sorted(holdings, key=lambda h: h["dollars"]),
                "estimated_holdings": estimated,
            }
        )
    return replays


def move(prices, symbol, start, end):
    """Price change from the close on or before start to the close on or
    before end, or None if the symbol wasn't trading for the whole period."""
    if symbol not in prices:
        return None
    series = prices[symbol].dropna()
    if series.empty or series.index[0] > start or series.index[-1] < end:
        return None
    before, after = series[:start], series[:end]
    if before.empty or (start - before.index[-1]).days > 7:
        return None
    return float(after.iloc[-1] / before.iloc[-1] - 1)


def report(a):
    a.load()
    betas = downside_betas(a)
    return {
        "as_of": a.as_of,
        "total_value": usd(float(a.port.iloc[-1])),
        "market_drops": market_drops(a, betas),
        "crash_replays": crash_replays(a, betas=betas),
        "downside_betas": {s: round(b, 2) for s, b in sorted(betas.items(), key=lambda kv: -kv[1])},
        "note": "Crash replays use each holding's actual move where it traded through the whole period, "
        "then what it tracks (e.g. SPY for VOO, bitcoin for IBIT), and only then downside beta x the "
        "market's move, marked estimated. A beta estimate only captures how a holding moves with "
        "stocks, so for anything with crashes of its own (crypto, single themes) treat it as a floor "
        "on the loss, not a forecast.",
    }
