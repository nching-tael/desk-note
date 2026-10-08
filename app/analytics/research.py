"""News, theses and upcoming earnings."""

from __future__ import annotations

from datetime import date

from .periods import AnalyticsError, pct, usd


def news(a, symbols, days=7, limit=8):
    days = max(1, min(int(days), 30))
    headlines = {}
    for symbol in symbols:
        symbol = str(symbol).strip().upper()
        if not symbol:
            continue
        try:
            items = a.provider.news(symbol, days)
        except Exception:
            items = []
        headlines[symbol] = [
            {
                "title": item["title"],
                "publisher": item.get("publisher"),
                "published": item["published"][:16].replace("T", " ") + " UTC",
                "summary": (item.get("summary") or "")[:280],
                "url": item.get("url"),
            }
            for item in items[:limit]
        ]
    return {
        "days": days,
        "news": headlines,
        "note": "Headlines only cover what the provider returns; no news doesn't mean nothing happened.",
    }


def thesis(a, symbol):
    """The user's thesis for a holding plus the evidence to judge it: recent
    performance split into market/sector/stock, news for the stock and its
    watch list, and the next earnings date."""
    a.load()
    symbol = str(symbol).strip().upper()
    entry = a.theses.get(symbol)
    held = symbol in a.symbols
    if not held and entry is None:
        raise AnalyticsError(f"{symbol} is not in the portfolio and has no thesis in theses.yaml.")

    result = {"symbol": symbol, "name": a.name(symbol) if held else symbol}
    if entry:
        result.update(thesis=entry["thesis"], breaks_if=entry["breaks_if"], watch_list=entry["watch"])
    else:
        result.update(
            thesis=None,
            breaks_if=[],
            watch_list=[],
            note="No thesis recorded in theses.yaml for this holding.",
        )

    if held:
        result["performance"] = recent_moves(a, symbol)
        result["position_value"] = usd(a.values[symbol].iloc[-1])
        result["weight_pct"] = pct(a.weights()[symbol])

    watch = result["watch_list"]
    headlines = news(a, [symbol] + watch, days=14, limit=6)["news"]
    result["news_14d"] = {
        "own": headlines.get(symbol, []),
        "watch_list": {s: headlines.get(s, []) for s in watch},
    }
    result["next_earnings"] = next_earnings(a, symbol)
    return result


def recent_moves(a, symbol):
    moves = {}
    for period in ("1w", "1m"):
        df, w = a.attribution_frame(period)
        if symbol not in df.index:  # bought too recently to have moved
            continue
        row = df.loc[symbol]
        moves[period] = {
            "dollars": usd(row["total"]),
            "pct": pct(row["total"] / row["start_value"]),
            "market_dollars": usd(row["market"]),
            "sector_dollars": usd(row["sector_move"]),
            "stock_specific_dollars": usd(row["stock_specific"]),
            "stock_specific_pct": pct(row["stock_specific"] / row["start_value"]),
            "spy_pct": pct(a.market_return(w)),
        }
    return moves


def next_earnings(a, symbol):
    try:
        earnings = a.provider.earnings(symbol)
    except Exception:
        earnings = None
    if not earnings or not earnings.get("date"):
        return None

    move = earnings.get("implied_move_pct")
    days_until = (date.fromisoformat(earnings["date"]) - date.fromisoformat(a.as_of)).days
    result = {"date": earnings["date"], "days_until": days_until, "implied_move_pct": move}
    if symbol in a.symbols and move is not None:
        result["implied_move_dollars"] = usd(a.values[symbol].iloc[-1] * move / 100)
    return result


def events(a, days=30):
    a.load()
    days = max(1, min(int(days), 120))
    upcoming, unknown = [], []
    for symbol in a.symbols:
        earnings = next_earnings(a, symbol)
        if earnings is None:
            unknown.append(symbol)
        elif 0 <= earnings["days_until"] <= days:
            upcoming.append(
                {
                    "symbol": symbol,
                    "name": a.name(symbol),
                    "event": "earnings",
                    "position_value": usd(a.values[symbol].iloc[-1]),
                    **earnings,
                }
            )
    return {
        "as_of": a.as_of,
        "window_days": days,
        "events": sorted(upcoming, key=lambda e: e["date"]),
        "no_date_available": unknown,
        "note": "Implied move = the options market's expected move through earnings, in either direction.",
    }
