"""Tool schemas exposed to Claude, and dispatch to the analytics layer.

``run_tool`` never raises: any failure comes back as an error tool result the
model can read and recover from.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .analytics import PERIOD_LABELS, PERIODS, Analytics, AnalyticsError, fmt_usd, normalise_period, usd

log = logging.getLogger(__name__)

_PERIOD = {
    "type": "string",
    "enum": PERIODS,
    "description": "1d = last trading day, 1w = last 5 trading days, 1m, 3m, ytd = year to date, 1y.",
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_overview",
        "description": (
            "Snapshot of the portfolio today: total value, today's and this week's change in $ and %, "
            "unrealised profit/loss versus cost, and every position's value, weight, day move and P/L. "
            "Start here for general 'how am I doing' questions or to see what the user holds."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_performance",
        "description": (
            "Gains and losses over a period: total $ and % change, each position's $ and %, best and worst "
            "positions, and the S&P 500 (SPY) return over the same period for comparison."
        ),
        "input_schema": {"type": "object", "properties": {"period": _PERIOD}, "required": ["period"]},
    },
    {
        "name": "get_attribution",
        "description": (
            "Explains WHY the portfolio moved over a period, in dollars: splits each holding's change into "
            "the market (SPY), its sector (sector ETF vs SPY) and stock-specific moves, using a two-factor "
            "model fit on the prior year. Returns a ready-made headline sentence plus per-holding detail. "
            "Use for 'why did I lose/make money' questions."
        ),
        "input_schema": {"type": "object", "properties": {"period": _PERIOD}, "required": ["period"]},
    },
    {
        "name": "get_risk_report",
        "description": (
            "Risk report at current weights using one year of daily returns: annualised volatility, beta to "
            "SPY, each holding's share of total risk, sector weights, effective number of positions, highly "
            "correlated pairs and groups of holdings that move together, 1-year max drawdown, and a stress "
            "test of the $ impact if SPY falls 10%."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_news",
        "description": (
            "Recent headlines (title, publisher, time, short summary) for one or more ticker symbols. Works "
            "for any symbol, not just holdings. Use it to check whether news explains a move."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 10,
                            "description": "Ticker symbols, e.g. [\"NVDA\", \"MSFT\"]."},
                "days": {"type": "integer", "minimum": 1, "maximum": 30,
                         "description": "How many days back to look. Default 7."},
            },
            "required": ["symbols"],
        },
    },
    {
        "name": "get_thesis",
        "description": (
            "The user's investment thesis for a holding (why they own it, what would prove it wrong, and a "
            "watch list of related companies) together with the evidence needed to judge it: 1-week and "
            "1-month performance including the stock-specific move, 14 days of news for the stock AND every "
            "watch-list company, and the next earnings date with the options-implied move."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "Ticker symbol, e.g. NVDA."}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_upcoming_events",
        "description": (
            "Upcoming earnings dates for holdings within a window, each with the options-implied move in % "
            "and in $ on the user's position."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"days": {"type": "integer", "minimum": 1, "maximum": 120,
                                    "description": "Look-ahead window in calendar days. Default 30."}},
        },
    },
    {
        "name": "get_chart",
        "description": (
            "Show the user a chart. Kinds: 'portfolio_vs_market' (portfolio value vs SPY scaled to the same "
            "start), 'stock_vs_market' (one symbol vs SPY, both rebased to 0%; needs symbol), 'attribution' "
            "(stacked bar per holding: market, sector, stock-specific $). The chart is rendered in the UI; "
            "you only receive a confirmation, so take numbers from the other tools."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["portfolio_vs_market", "stock_vs_market", "attribution"]},
                "symbol": {"type": "string", "description": "Required for stock_vs_market."},
                "period": _PERIOD,
            },
            "required": ["kind"],
        },
    },
]

# Tools that need the store: reading trade history and the journal, and writing.
# The MCP server exposes these; the web demo is read-only and uses TOOLS only.
MEMORY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_trades",
        "description": "Trades the user has recorded (buys and sells after their opening positions), with ids "
                       "and realised profit/loss. Use the id with delete_trade to fix a mistake.",
        "input_schema": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "Optional: only this ticker."}}},
    },
    {
        "name": "review_journal",
        "description": (
            "The user's decision journal (why they bought, sold or changed a thesis), with how each decision "
            "has worked out since: the stock's move, the S&P 500's move, the market-adjusted move, and for "
            "trades the $ gained since buying or given up since selling. Use for 'how have my decisions "
            "worked out', 'was selling X a mistake', or before revisiting a holding."
        ),
        "input_schema": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "Optional: only this ticker."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Default 20, newest first."}}},
    },
]

WRITE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "record_trade",
        "description": (
            "Record a buy or sell the user has ALREADY made, updating their holdings. Only call when the user "
            "clearly states a completed trade; if shares are ambiguous (e.g. 'half'), work them out from "
            "get_overview and confirm with the user first. If they say why, pass it as reason: it goes in the "
            "decision journal. Price defaults to the latest close if not given."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "side": {"type": "string", "enum": ["buy", "sell"]},
                "shares": {"type": "number", "exclusiveMinimum": 0},
                "price": {"type": "number", "exclusiveMinimum": 0, "description": "Price per share in $."},
                "date": {"type": "string", "description": "YYYY-MM-DD. Defaults to today."},
                "reason": {"type": "string", "description": "Why the user made the trade, in their words."},
            },
            "required": ["symbol", "side", "shares"],
        },
    },
    {
        "name": "delete_trade",
        "description": "Delete a recorded trade by id (from get_trades), e.g. to fix a mistake. Confirm with the user first.",
        "input_schema": {"type": "object", "properties": {"trade_id": {"type": "integer"}}, "required": ["trade_id"]},
    },
    {
        "name": "update_thesis",
        "description": (
            "Create or edit the user's thesis for a holding: the thesis text, the list of things that would "
            "break it (replaces the list), and watch-list companies to add or remove. Only change what the "
            "user asked to change."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "thesis": {"type": "string"},
                "breaks_if": {"type": "array", "items": {"type": "string"}},
                "watch_add": {"type": "array", "items": {"type": "string"}},
                "watch_remove": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "add_journal_entry",
        "description": (
            "Save a note to the user's decision journal, e.g. why they're considering a stock or how their "
            "view changed. (Trades with a reason are journaled automatically by record_trade.)"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "symbol": {"type": "string"},
                "kind": {"type": "string", "enum": ["buy_reason", "sell_reason", "thesis_change", "note"]},
            },
            "required": ["text"],
        },
    },
]

TOOL_NAMES = {t["name"] for t in TOOLS + MEMORY_TOOLS + WRITE_TOOLS}
WRITE_TOOL_NAMES = {t["name"] for t in WRITE_TOOLS}


@dataclass
class ToolOutcome:
    name: str
    input: dict[str, Any]
    content: str  # JSON string sent back to the model
    is_error: bool = False
    summary: str = ""
    detail: str | None = None
    chart: dict[str, Any] | None = None
    changed: bool = False  # True after a successful write: rebuild analytics
    result: dict[str, Any] | None = field(default=None, repr=False)

    def trace_entry(self) -> dict[str, Any]:
        entry = {"tool": self.name, "input": self.input, "summary": self.summary, "ok": not self.is_error}
        if self.detail:
            entry["detail"] = self.detail
        return entry


def _period_label(inp: dict[str, Any], default: str) -> str:
    try:
        return PERIOD_LABELS[normalise_period(inp.get("period"), default)]
    except AnalyticsError:
        return str(inp.get("period"))


def _symbols(inp: dict[str, Any]) -> list[str]:
    raw = inp.get("symbols") or []
    if isinstance(raw, str):
        raw = [s for s in raw.replace(",", " ").split()]
    return [str(s).strip().upper() for s in raw if str(s).strip()][:10]


def summarise(name: str, inp: dict[str, Any]) -> str:
    """Short, human-readable description of a tool call for the trace."""
    if name == "get_overview":
        return "Checked today's portfolio snapshot"
    if name == "get_performance":
        return f"Checked gains and losses for the {_period_label(inp, '1w')}"
    if name == "get_attribution":
        return f"Split the {_period_label(inp, '1w')} into market, sector and stock-picking"
    if name == "get_risk_report":
        return "Ran the risk report: volatility, concentration, correlations, stress test"
    if name == "get_news":
        return f"Read headlines for {', '.join(_symbols(inp)) or 'no symbols'} from the last {inp.get('days', 7)} days"
    if name == "get_thesis":
        return f"Reviewed the {str(inp.get('symbol', '?')).upper()} thesis against recent news and price moves"
    if name == "get_upcoming_events":
        return f"Checked earnings dates in the next {inp.get('days', 30)} days"
    if name == "get_trades":
        return f"Looked up recorded trades{' for ' + str(inp['symbol']).upper() if inp.get('symbol') else ''}"
    if name == "review_journal":
        return f"Reviewed journal decisions{' on ' + str(inp['symbol']).upper() if inp.get('symbol') else ''} against what happened since"
    if name == "record_trade":
        return (f"Recorded a {inp.get('side', '?')} of {inp.get('shares', '?')} "
                f"{str(inp.get('symbol', '?')).upper()}")
    if name == "delete_trade":
        return f"Deleted trade #{inp.get('trade_id', '?')}"
    if name == "update_thesis":
        return f"Updated the {str(inp.get('symbol', '?')).upper()} thesis"
    if name == "add_journal_entry":
        return "Saved a journal entry"
    if name == "get_chart":
        kind = str(inp.get("kind") or "")
        if kind == "stock_vs_market":
            return f"Drew a chart of {str(inp.get('symbol') or '?').upper()} vs the market"
        if kind == "portfolio_vs_market":
            return "Drew a chart of your portfolio vs the market"
        if kind == "attribution":
            return "Drew an attribution chart (market, sector, stock-specific)"
        return f"Drew a {kind.replace('_', ' ') or 'chart'} chart"
    return f"Called {name}"


def _detail(name: str, result: dict[str, Any]) -> str | None:
    """One key figure from the result, so the trace shows evidence at a glance."""
    if name == "get_attribution":
        return result.get("headline")
    if name == "get_overview":
        return f"Total value ${result['total_value']:,} as of {result['as_of']}"
    if name == "get_performance":
        return (f"{fmt_usd(result['change_dollars'])} ({result['change_pct']:+.1f}%) vs SPY "
                f"{result['market_spy_pct']:+.1f}%")
    if name == "get_risk_report":
        return f"Volatility {result['volatility_annual_pct']}%/yr, beta {result['beta_to_spy']}"
    if name == "get_upcoming_events":
        return f"{len(result['events'])} earnings event(s) found"
    if name == "get_news":
        n = sum(len(v) for v in result["news"].values())
        return f"{n} headline(s)"
    return None


def _latest_price(analytics: Analytics, symbol: str) -> float | None:
    series = analytics._price_series(symbol)
    if series is None or series.dropna().empty:
        return None
    return float(series.dropna().iloc[-1])


def _call_store(analytics: Analytics, store, name: str, inp: dict[str, Any]) -> dict[str, Any]:
    from datetime import date as _date

    from .portfolio import Trade, replay

    if store is None:
        raise AnalyticsError("Trades and the journal aren't available here (no database).")
    if name == "review_journal":
        entries = store.journal(inp.get("symbol"), int(inp.get("limit") or 20))
        if not entries:
            return {"entries": [], "summary": {"entries": 0},
                    "note": "The journal is empty. Entries are added by record_trade (with a reason) or add_journal_entry."}
        return analytics.review_journal(entries)
    if name == "record_trade":
        symbol = str(inp.get("symbol") or "").strip().upper()
        price = inp.get("price")
        latest = _latest_price(analytics, symbol) if symbol else None
        if latest is None:
            raise AnalyticsError(f"No price data for '{symbol}'. Check the ticker symbol.")
        assumed = price is None
        when = _date.fromisoformat(inp["date"]) if inp.get("date") else _date.fromisoformat(analytics.as_of)
        trade = store.add_trade(
            Trade(when, symbol, str(inp.get("side") or "").lower(), float(inp.get("shares") or 0),
                  float(price) if price is not None else round(latest, 2)),
            reason=inp.get("reason"))
        pos = replay(store.opening(), store.trades()).get(symbol)
        out = {
            "recorded": {"id": trade.id, "date": trade.date.isoformat(), "symbol": symbol, "side": trade.side,
                         "shares": trade.shares, "price": trade.price, "value_dollars": usd(trade.shares * trade.price)},
            "position_now": {"shares": pos.shares if pos else 0.0,
                             "avg_cost": round(pos.avg_cost, 2) if pos and pos.avg_cost else None},
            "journaled_reason": bool(inp.get("reason")),
        }
        if assumed:
            out["note"] = f"No price given; used the latest close (${trade.price:,.2f}). Tell the user."
        if trade.side == "sell" and pos is not None:
            out["realised_pl_total_dollars"] = usd(pos.realised)
        return out
    if name == "delete_trade":
        t = store.delete_trade(int(inp.get("trade_id")))
        return {"deleted": {"id": t.id, "date": t.date.isoformat(), "symbol": t.symbol, "side": t.side,
                            "shares": t.shares, "price": t.price}}
    if name == "update_thesis":
        return {"thesis": store.update_thesis(inp.get("symbol"), inp.get("thesis"), inp.get("breaks_if"),
                                              inp.get("watch_add"), inp.get("watch_remove"))}
    if name == "add_journal_entry":
        entry = store.add_journal(str(inp.get("text") or ""), inp.get("symbol"), inp.get("kind") or "note",
                                  on=_date.fromisoformat(analytics.as_of))
        return {"saved": entry}
    raise AnalyticsError(f"Unknown tool '{name}'.")


def _call(analytics: Analytics, name: str, inp: dict[str, Any]) -> dict[str, Any]:
    if name == "get_overview":
        return analytics.overview()
    if name == "get_performance":
        return analytics.performance(inp.get("period") or "1w")
    if name == "get_attribution":
        return analytics.attribution(inp.get("period") or "1w")
    if name == "get_risk_report":
        return analytics.risk()
    if name == "get_news":
        symbols = _symbols(inp)
        if not symbols:
            raise AnalyticsError("Provide at least one ticker symbol.")
        return analytics.news(symbols, int(inp.get("days") or 7))
    if name == "get_thesis":
        if not inp.get("symbol"):
            raise AnalyticsError("Provide a symbol.")
        return analytics.thesis(str(inp["symbol"]))
    if name == "get_upcoming_events":
        return analytics.events(int(inp.get("days") or 30))
    if name == "get_chart":
        return analytics.chart(str(inp.get("kind") or ""), inp.get("symbol"), inp.get("period"))
    if name == "get_trades":
        return analytics.trades_report(inp.get("symbol"))
    raise AnalyticsError(f"Unknown tool '{name}'. Available: {', '.join(sorted(TOOL_NAMES))}.")


def run_tool(analytics: Analytics, name: str, inp: dict[str, Any] | None, store=None) -> ToolOutcome:
    """Execute a tool. Always returns a ToolOutcome whose ``content`` is valid JSON.
    ``store`` enables the journal and write tools."""
    inp = dict(inp or {})
    summary = summarise(name, inp)
    try:
        if name in WRITE_TOOL_NAMES or name == "review_journal":
            result = _call_store(analytics, store, name, inp)
        else:
            result = _call(analytics, name, inp)
    except (AnalyticsError, ValueError, TypeError, KeyError) as exc:
        msg = str(exc) or exc.__class__.__name__
        return ToolOutcome(name, inp, json.dumps({"error": msg}), True, f"{summary} (failed: {msg})")
    except Exception as exc:  # data provider failures etc.
        log.exception("tool %s failed", name)
        msg = f"{exc.__class__.__name__}: {exc}"
        return ToolOutcome(name, inp, json.dumps({"error": f"Tool failed: {msg}"}), True,
                           f"{summary} (failed: {exc.__class__.__name__})")

    if name == "get_chart":
        note = {"status": "Chart shown to user", "title": result.get("title")}
        return ToolOutcome(name, inp, json.dumps(note), False, summary, chart=result, result=result)
    try:
        detail = _detail(name, result)
    except (KeyError, TypeError, ValueError):
        detail = None
    return ToolOutcome(name, inp, json.dumps(result, default=str, ensure_ascii=False), False, summary,
                       detail=detail, result=result, changed=name in WRITE_TOOL_NAMES)
