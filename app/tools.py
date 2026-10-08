"""The tools Claude can call, each defined next to the code that runs it.

run_tool never raises: a failure comes back as an error result the model can
read and recover from.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from .analytics import PERIOD_LABELS, PERIODS, AnalyticsError, fmt_usd, normalise_period, usd
from .portfolio import Trade, replay

log = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    handler: Callable
    describe: Callable  # args -> short summary for the trace
    group: str  # "analysis", "memory" or "write"

    def api_schema(self):
        return {"name": self.name, "description": self.description, "input_schema": self.schema}


REGISTRY: dict[str, Tool] = {}


def tool(name, description, describe, properties=None, required=(), group="analysis"):
    schema = {"type": "object", "properties": properties or {}}
    if required:
        schema["required"] = list(required)

    def register(handler):
        REGISTRY[name] = Tool(name, " ".join(description.split()), schema, handler, describe, group)
        return handler

    return register


@dataclass
class Context:
    analytics: Any
    store: Any = None


@dataclass
class ToolOutcome:
    name: str
    input: dict
    content: str  # JSON sent back to the model
    is_error: bool = False
    summary: str = ""
    detail: str | None = None
    chart: dict | None = None
    result: dict | None = None
    changed: bool = False  # a write succeeded, so cached analytics are stale

    def trace_entry(self):
        entry = {"tool": self.name, "input": self.input, "summary": self.summary, "ok": not self.is_error}
        if self.detail:
            entry["detail"] = self.detail
        return entry


PERIOD = {
    "type": "string",
    "enum": PERIODS,
    "description": "1d = last trading day, 1w = last 5 trading days, 1m, 3m, ytd = year to date, 1y.",
}
SYMBOL = {"type": "string", "description": "Ticker symbol, e.g. NVDA."}


def period_label(args, default="1w"):
    try:
        return PERIOD_LABELS[normalise_period(args.get("period"), default)]
    except AnalyticsError:
        return str(args.get("period"))


def symbol_list(args):
    symbols = args.get("symbols") or []
    if isinstance(symbols, str):
        symbols = symbols.replace(",", " ").split()
    return [str(s).strip().upper() for s in symbols if str(s).strip()][:10]


def upper(value, default="?"):
    return str(value or default).upper()


def for_symbol(args, prefix=" for "):
    return prefix + upper(args["symbol"]) if args.get("symbol") else ""


# --- analysis (read-only, used by both the web app and the MCP server)


@tool(
    "get_overview",
    """
    Snapshot of the portfolio today: total value, today's and this week's change in $ and %,
    unrealised profit/loss versus cost, and every position's value, weight, day move and P/L.
    Start here for general 'how am I doing' questions or to see what the user holds.""",
    describe=lambda args: "Checked today's portfolio snapshot",
)
def get_overview(ctx, args):
    return ctx.analytics.overview()


@tool(
    "get_performance",
    """
    Gains and losses over a period: total $ and % change, each position's $ and %, best and
    worst positions, and the S&P 500 (SPY) return over the same period for comparison.""",
    describe=lambda args: f"Checked gains and losses for the {period_label(args)}",
    properties={"period": PERIOD},
    required=["period"],
)
def get_performance(ctx, args):
    return ctx.analytics.performance(args.get("period") or "1w")


@tool(
    "get_attribution",
    """
    Explains WHY the portfolio moved over a period, in dollars: splits each holding's change into
    the market (SPY), its sector (sector ETF vs SPY) and stock-specific moves, using a two-factor
    model fit on the prior year. Returns a ready-made headline sentence plus per-holding detail.
    Use for 'why did I lose/make money' questions.""",
    describe=lambda args: f"Split the {period_label(args)} into market, sector and stock-picking",
    properties={"period": PERIOD},
    required=["period"],
)
def get_attribution(ctx, args):
    return ctx.analytics.attribution(args.get("period") or "1w")


@tool(
    "get_risk_report",
    """
    Risk report at current weights using one year of daily returns: annualised volatility, beta to
    SPY, each holding's share of total risk, sector weights, effective number of positions, highly
    correlated pairs and groups of holdings that move together, 1-year max drawdown, and a stress
    test of the $ impact if SPY falls 10%.""",
    describe=lambda args: "Ran the risk report: volatility, concentration, correlations, stress test",
)
def get_risk_report(ctx, args):
    return ctx.analytics.risk()


@tool(
    "get_exposure",
    """
    Look-through view of what the user really owns: their ETFs and funds opened up into the
    underlying stocks (direct holdings plus each fund's share), stocks held through more than one
    fund, and true sector exposure across everything. Use for ETF portfolios and for questions like
    'how much Nvidia do I really own' or 'am I more concentrated than I think'.""",
    describe=lambda args: "Looked through your funds to the stocks and sectors underneath",
)
def get_exposure(ctx, args):
    return ctx.analytics.exposure()


@tool(
    "get_allocation",
    """
    The user's target allocation versus where they are now: each position's weight, target, drift
    in percentage points, whether it's outside its band, and the dollar trade that would put it back
    on target. Core vs satellite totals too. Pass new_money to get a plan for investing a
    contribution that moves toward target without selling anything.""",
    describe=lambda args: (
        "Checked your allocation against your targets"
        + (f" and planned how to invest ${args['new_money']:,}" if args.get("new_money") else "")
    ),
    properties={
        "new_money": {"type": "number", "minimum": 0, "description": "Optional: $ about to be invested."}
    },
)
def get_allocation(ctx, args):
    return ctx.analytics.allocation(float(args.get("new_money") or 0))


@tool(
    "get_scorecard",
    """
    Satellite scorecard: for each satellite (non-core) holding over a period, its actual $ gain
    versus what the same money would have made in the user's core over the same days, so the user
    can see whether each bet beat simply owning more of the core. Uses the S&P 500 as the core if no
    core is set.""",
    describe=lambda args: f"Scored each satellite against the core for the {period_label(args, '1y')}",
    properties={"period": PERIOD},
)
def get_scorecard(ctx, args):
    return ctx.analytics.scorecard(args.get("period") or "1y")


@tool(
    "get_news",
    """
    Recent headlines (title, publisher, time, short summary) for one or more ticker symbols.
    Works for any symbol, not just holdings. Use it to check whether news explains a move.""",
    describe=lambda args: (
        f"Read headlines for {', '.join(symbol_list(args)) or 'no symbols'} "
        f"from the last {args.get('days', 7)} days"
    ),
    properties={
        "symbols": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10,
            "description": 'Ticker symbols, e.g. ["NVDA", "MSFT"].',
        },
        "days": {
            "type": "integer",
            "minimum": 1,
            "maximum": 30,
            "description": "How many days back to look. Default 7.",
        },
    },
    required=["symbols"],
)
def get_news(ctx, args):
    symbols = symbol_list(args)
    if not symbols:
        raise AnalyticsError("Provide at least one ticker symbol.")
    return ctx.analytics.news(symbols, int(args.get("days") or 7))


@tool(
    "get_thesis",
    """
    The user's investment thesis for a holding (why they own it, what would prove it wrong, and a
    watch list of related companies) together with the evidence needed to judge it: 1-week and
    1-month performance including the stock-specific move, 14 days of news for the stock AND every
    watch-list company, and the next earnings date with the options-implied move.""",
    describe=lambda args: (
        f"Reviewed the {upper(args.get('symbol'))} thesis against recent news and price moves"
    ),
    properties={"symbol": SYMBOL},
    required=["symbol"],
)
def get_thesis(ctx, args):
    if not args.get("symbol"):
        raise AnalyticsError("Provide a symbol.")
    return ctx.analytics.thesis(str(args["symbol"]))


@tool(
    "get_upcoming_events",
    """
    Upcoming earnings dates for holdings within a window, each with the options-implied move in %
    and in $ on the user's position.""",
    describe=lambda args: f"Checked earnings dates in the next {args.get('days', 30)} days",
    properties={
        "days": {
            "type": "integer",
            "minimum": 1,
            "maximum": 120,
            "description": "Look-ahead window in calendar days. Default 30.",
        }
    },
)
def get_upcoming_events(ctx, args):
    return ctx.analytics.events(int(args.get("days") or 30))


def describe_chart(args):
    kind = args.get("kind")
    if kind == "stock_vs_market":
        return f"Drew a chart of {upper(args.get('symbol'))} vs the market"
    if kind == "portfolio_vs_market":
        return "Drew a chart of your portfolio vs the market"
    if kind == "attribution":
        return "Drew an attribution chart (market, sector, stock-specific)"
    return "Drew a chart"


@tool(
    "get_chart",
    """
    Show the user a chart. Kinds: 'portfolio_vs_market' (portfolio value vs SPY scaled to the same
    start), 'stock_vs_market' (one symbol vs SPY, both rebased to 0%; needs symbol), 'attribution'
    (stacked bar per holding: market, sector, stock-specific $). The chart is rendered in the UI;
    you only receive a confirmation, so take numbers from the other tools.""",
    describe=describe_chart,
    properties={
        "kind": {"type": "string", "enum": ["portfolio_vs_market", "stock_vs_market", "attribution"]},
        "symbol": {"type": "string", "description": "Required for stock_vs_market."},
        "period": PERIOD,
    },
    required=["kind"],
)
def get_chart(ctx, args):
    return ctx.analytics.chart(str(args.get("kind") or ""), args.get("symbol"), args.get("period"))


# --- memory: trade history and the journal (MCP server only)


@tool(
    "get_trades",
    """
    Trades the user has recorded (buys and sells after their opening positions), with ids and
    realised profit/loss. Use the id with delete_trade to fix a mistake.""",
    describe=lambda args: "Looked up recorded trades" + for_symbol(args),
    properties={"symbol": {"type": "string", "description": "Optional: only this ticker."}},
    group="memory",
)
def get_trades(ctx, args):
    return ctx.analytics.trades_report(args.get("symbol"))


@tool(
    "review_journal",
    """
    The user's decision journal (why they bought, sold or changed a thesis), with how each decision
    has worked out since: the stock's move, the S&P 500's move, the market-adjusted move, and for
    trades the $ gained since buying or given up since selling. Use for 'how have my decisions
    worked out', 'was selling X a mistake', or before revisiting a holding.""",
    describe=lambda args: (
        "Reviewed journal decisions" + for_symbol(args, " on ") + " against what happened since"
    ),
    properties={
        "symbol": {"type": "string", "description": "Optional: only this ticker."},
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "description": "Default 20, newest first.",
        },
    },
    group="memory",
)
def review_journal(ctx, args):
    entries = require_store(ctx).journal(args.get("symbol"), int(args.get("limit") or 20))
    if not entries:
        return {
            "entries": [],
            "summary": {"entries": 0},
            "note": "The journal is empty. Entries come from record_trade (with a reason) "
            "or add_journal_entry.",
        }
    return ctx.analytics.review_journal(entries)


# --- writes (MCP server only)


def require_store(ctx):
    if ctx.store is None:
        raise AnalyticsError("Trades and the journal aren't available here (no database).")
    return ctx.store


def latest_price(analytics, symbol):
    prices = analytics.price_series(symbol)
    if prices is None or prices.isna().all():
        return None
    return float(prices.dropna().iloc[-1])


@tool(
    "record_trade",
    """
    Record a buy or sell the user has ALREADY made, updating their holdings. Only call when the
    user clearly states a completed trade; if shares are ambiguous (e.g. 'half'), work them out
    from get_overview and confirm with the user first. If they say why, pass it as reason: it goes
    in the decision journal. Price defaults to the latest close if not given.""",
    describe=lambda args: (
        f"Recorded a {args.get('side', '?')} of {args.get('shares', '?')} {upper(args.get('symbol'))}"
    ),
    properties={
        "symbol": {"type": "string"},
        "side": {"type": "string", "enum": ["buy", "sell"]},
        "shares": {"type": "number", "exclusiveMinimum": 0},
        "price": {"type": "number", "exclusiveMinimum": 0, "description": "Price per share in $."},
        "date": {"type": "string", "description": "YYYY-MM-DD. Defaults to today."},
        "reason": {"type": "string", "description": "Why the user made the trade, in their words."},
    },
    required=["symbol", "side", "shares"],
    group="write",
)
def record_trade(ctx, args):
    store = require_store(ctx)
    symbol = upper(args.get("symbol"), "")
    latest = latest_price(ctx.analytics, symbol) if symbol else None
    if latest is None:
        raise AnalyticsError(f"No price data for '{symbol}'. Check the ticker symbol.")

    price = args.get("price")
    when = date.fromisoformat(args.get("date") or ctx.analytics.as_of)
    trade = Trade(
        when,
        symbol,
        str(args.get("side") or "").lower(),
        float(args.get("shares") or 0),
        float(price) if price is not None else round(latest, 2),
    )
    trade = store.add_trade(trade, reason=args.get("reason"))
    position = replay(store.opening(), store.trades())[symbol]

    result = {
        "recorded": {
            "id": trade.id,
            "date": trade.date.isoformat(),
            "symbol": symbol,
            "side": trade.side,
            "shares": trade.shares,
            "price": trade.price,
            "value_dollars": usd(trade.shares * trade.price),
        },
        "position_now": {
            "shares": position.shares,
            "avg_cost": round(position.avg_cost, 2) if position.avg_cost else None,
        },
        "journaled_reason": bool(args.get("reason")),
    }
    if price is None:
        result["note"] = f"No price given; used the latest close (${trade.price:,.2f}). Tell the user."
    if trade.side == "sell":
        result["realised_pl_total_dollars"] = usd(position.realised)
    return result


@tool(
    "delete_trade",
    """
    Delete a recorded trade by id (from get_trades), e.g. to fix a mistake. Confirm with the user
    first.""",
    describe=lambda args: f"Deleted trade #{args.get('trade_id', '?')}",
    properties={"trade_id": {"type": "integer"}},
    required=["trade_id"],
    group="write",
)
def delete_trade(ctx, args):
    trade = require_store(ctx).delete_trade(int(args.get("trade_id")))
    return {
        "deleted": {
            "id": trade.id,
            "date": trade.date.isoformat(),
            "symbol": trade.symbol,
            "side": trade.side,
            "shares": trade.shares,
            "price": trade.price,
        }
    }


@tool(
    "set_targets",
    """
    Save the user's target allocation, e.g. VOO 70% core and four satellites at 7.5%. Each target
    has a role (core or satellite) and an optional drift band in percentage points (default 5 for
    positions of 20% or more, 2 otherwise). With replace=true (default) the targets must add up to
    100%; with replace=false only the listed symbols change. Only save what the user asked for.""",
    describe=lambda args: f"Saved target allocation for {len(args.get('targets') or [])} position(s)",
    properties={
        "targets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "target_pct": {"type": "number", "minimum": 0, "maximum": 100},
                    "role": {"type": "string", "enum": ["core", "satellite"]},
                    "band_pct": {"type": "number", "minimum": 0},
                },
                "required": ["symbol", "target_pct"],
            },
        },
        "replace": {"type": "boolean", "description": "Replace all targets (default true)."},
    },
    required=["targets"],
    group="write",
)
def set_targets(ctx, args):
    replace = args.get("replace", True)
    return {"targets": require_store(ctx).set_targets(args.get("targets") or [], replace=replace)}


@tool(
    "update_thesis",
    """
    Create or edit the user's thesis for a holding: the thesis text, the list of things that would
    break it (replaces the list), and watch-list companies to add or remove. Only change what the
    user asked to change.""",
    describe=lambda args: f"Updated the {upper(args.get('symbol'))} thesis",
    properties={
        "symbol": {"type": "string"},
        "thesis": {"type": "string"},
        "breaks_if": {"type": "array", "items": {"type": "string"}},
        "watch_add": {"type": "array", "items": {"type": "string"}},
        "watch_remove": {"type": "array", "items": {"type": "string"}},
    },
    required=["symbol"],
    group="write",
)
def update_thesis(ctx, args):
    thesis = require_store(ctx).update_thesis(
        args.get("symbol"),
        args.get("thesis"),
        args.get("breaks_if"),
        args.get("watch_add"),
        args.get("watch_remove"),
    )
    return {"thesis": thesis}


@tool(
    "add_journal_entry",
    """
    Save a note to the user's decision journal, e.g. why they're considering a stock or how their
    view changed. (Trades with a reason are journaled automatically by record_trade.)""",
    describe=lambda args: "Saved a journal entry",
    properties={
        "text": {"type": "string"},
        "symbol": {"type": "string"},
        "kind": {"type": "string", "enum": ["buy_reason", "sell_reason", "thesis_change", "note"]},
    },
    required=["text"],
    group="write",
)
def add_journal_entry(ctx, args):
    entry = require_store(ctx).add_journal(
        str(args.get("text") or ""),
        args.get("symbol"),
        args.get("kind") or "note",
        on=date.fromisoformat(ctx.analytics.as_of),
    )
    return {"saved": entry}


def schemas(*groups):
    return [t.api_schema() for t in REGISTRY.values() if t.group in groups]


TOOLS = schemas("analysis")  # the web app's agent uses these
MEMORY_TOOLS = schemas("memory")
WRITE_TOOLS = schemas("write")
WRITE_TOOL_NAMES = {t["name"] for t in WRITE_TOOLS}


def headline(name, result):
    """A key figure from the result, so the trace shows evidence at a glance."""
    if name == "get_attribution":
        return result["headline"]
    if name == "get_overview":
        return f"Total value ${result['total_value']:,} as of {result['as_of']}"
    if name == "get_performance":
        return (
            f"{fmt_usd(result['change_dollars'])} ({result['change_pct']:+.1f}%) "
            f"vs SPY {result['market_spy_pct']:+.1f}%"
        )
    if name == "get_risk_report":
        return f"Volatility {result['volatility_annual_pct']}%/yr, beta {result['beta_to_spy']}"
    if name == "get_upcoming_events":
        return f"{len(result['events'])} earnings event(s) found"
    if name == "get_allocation" and result.get("targets_set"):
        flagged = result["outside_band"]
        return f"{len(flagged)} position(s) outside their band" + (
            f": {', '.join(flagged)}" if flagged else ""
        )
    if name == "get_scorecard":
        net = result["total_added_vs_core_dollars"]
        return f"{result['satellites_beating_core']} satellites beat the core, net ${net:,}"
    if name == "get_exposure" and result["top_exposures"]:
        top = result["top_exposures"][0]
        return f"Largest underlying holding: {top['symbol']} at {top['pct']}% (${top['total_dollars']:,})"
    if name == "get_news":
        return f"{sum(len(items) for items in result['news'].values())} headline(s)"
    return None


def run_tool(analytics, name, args, store=None):
    args = dict(args or {})
    spec = REGISTRY.get(name)
    if spec is None:
        error = f"Unknown tool '{name}'. Available: {', '.join(sorted(REGISTRY))}."
        return ToolOutcome(name, args, json.dumps({"error": error}), True, f"Called {name} (failed)")

    summary = spec.describe(args)
    try:
        result = spec.handler(Context(analytics, store), args)
    except (ValueError, TypeError, KeyError) as e:  # bad input; the message is safe to show
        message = str(e) or type(e).__name__
        return ToolOutcome(name, args, json.dumps({"error": message}), True, f"{summary} (failed: {message})")
    except Exception as e:  # provider failures and bugs
        log.exception("tool %s failed", name)
        error = f"Tool failed: {type(e).__name__}: {e}"
        return ToolOutcome(
            name, args, json.dumps({"error": error}), True, f"{summary} (failed: {type(e).__name__})"
        )

    if name == "get_chart":
        note = {"status": "Chart shown to user", "title": result.get("title")}
        return ToolOutcome(name, args, json.dumps(note), summary=summary, chart=result, result=result)

    return ToolOutcome(
        name,
        args,
        json.dumps(result, default=str, ensure_ascii=False),
        summary=summary,
        detail=headline(name, result),
        result=result,
        changed=spec.group == "write",
    )
