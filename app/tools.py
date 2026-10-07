"""Tool schemas exposed to Claude, and dispatch to the analytics layer.

``run_tool`` never raises: any failure comes back as an error tool result the
model can read and recover from.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .analytics import PERIOD_LABELS, PERIODS, Analytics, AnalyticsError, fmt_usd, normalise_period

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

TOOL_NAMES = {t["name"] for t in TOOLS}


@dataclass
class ToolOutcome:
    name: str
    input: dict[str, Any]
    content: str  # JSON string sent back to the model
    is_error: bool = False
    summary: str = ""
    detail: str | None = None
    chart: dict[str, Any] | None = None
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
    if name == "get_chart":
        target = f" for {str(inp['symbol']).upper()}" if inp.get("symbol") else ""
        return f"Drew a {str(inp.get('kind', 'chart')).replace('_', ' ')} chart{target}"
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
    raise AnalyticsError(f"Unknown tool '{name}'. Available: {', '.join(sorted(TOOL_NAMES))}.")


def run_tool(analytics: Analytics, name: str, inp: dict[str, Any] | None) -> ToolOutcome:
    """Execute a tool. Always returns a ToolOutcome whose ``content`` is valid JSON."""
    inp = dict(inp or {})
    summary = summarise(name, inp)
    try:
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
                       detail=detail, result=result)
