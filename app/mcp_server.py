"""Desk Note as an MCP server, so Claude (desktop, mobile, Claude Code) can use
the same portfolio tools as the web app.

    python -m app.mcp_server --mock                      # stdio (Claude Desktop / Claude Code)
    python -m app.mcp_server --http --port 8001          # remote connector, e.g. on a Raspberry Pi
    python -m app.mcp_server --new-token                 # print a random access token

Over HTTP the endpoint is protected by DESK_NOTE_MCP_TOKEN. Claude reaches a
remote connector from Anthropic's servers, so the endpoint has to be public;
the token is what keeps your holdings private. Supply it either as a path
prefix (``https://host/<token>/mcp``, for the Claude app's connector form) or
as an ``Authorization: Bearer <token>`` header.
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
from typing import Any

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server

from .analytics import AnalyticsHolder
from .tools import MEMORY_TOOLS, TOOLS, WRITE_TOOL_NAMES, WRITE_TOOLS, run_tool

INSTRUCTIONS = """\
Desk Note gives you the user's real portfolio: holdings, performance, attribution, risk, \
investment theses and upcoming earnings, all computed in Python.

- Any number about the user's portfolio (values, $ and % changes, betas, weights, implied moves) \
must come from a Desk Note tool result. Don't estimate or calculate figures yourself; quote the tools.
- Use your own web search for broader context: why a stock or sector moved, what a company said, \
what analysts expect. Cite what you find, and keep it separate from Desk Note's numbers.
- Lead with dollars, then %. Separate what the market or sector did from the user's stock picks \
(get_attribution does this split).
- For thesis questions call get_thesis and give a verdict (supports / challenges / no change), \
citing specific headlines, including read-across from watch-list companies.
- If news doesn't explain a move, say so. Never invent a reason.
- This is information, not advice: lay out facts and considerations, don't recommend trades.

Desk Note also remembers things for the user. When they tell you about a trade they've made ("I sold half my AMD at 160"), record it with record_trade (confirm the share count if it's ambiguous), and pass their reason if they gave one. Edit theses with update_thesis when asked. Use review_journal to show how past decisions have worked out.
"""

_CHART_DESCRIPTION = (
    "Chart-ready data: labels plus one or more series, with a unit ($ or %). Kinds: "
    "'portfolio_vs_market' (portfolio value vs SPY scaled to the same start), 'stock_vs_market' "
    "(one symbol vs SPY, both rebased to 0%; needs symbol), 'attribution' (per holding: market, "
    "sector and stock-specific $). Draw it for the user if your interface can render charts."
)

_PROMPTS = [
    types.Prompt(
        name="morning_note",
        title="Morning note",
        description="A short briefing: what moved, why, and what's coming up.",
    ),
    types.Prompt(
        name="check_thesis",
        title="Check a thesis",
        description="Does recent news support or challenge the thesis for one holding?",
        arguments=[types.PromptArgument(name="symbol", description="Ticker, e.g. NVDA", required=True)],
    ),
    types.Prompt(
        name="review_decisions",
        title="Review my decisions",
        description="How have the trades and calls in my journal worked out, luck vs skill?",
    ),
]


def _prompt_text(name: str, args: dict[str, str]) -> str:
    if name == "morning_note":
        return (
            "Give me my morning note. Use get_overview and get_attribution for 1d to say what my "
            "portfolio did and why (market vs sector vs my picks, in dollars). For the two or three "
            "biggest stock-specific movers, check get_news and search the web for what happened. "
            "Then list earnings in the next 7 days with get_upcoming_events, and flag any news that "
            "challenges one of my theses. Keep it to a short paragraph and a few bullets."
        )
    if name == "check_thesis":
        symbol = (args.get("symbol") or "").strip().upper()
        if not symbol:
            raise ValueError("check_thesis needs a symbol")
        return (
            f"Does recent news break my {symbol} thesis? Call get_thesis for {symbol}, search the web "
            "for anything newer than its headlines, and give a verdict (supports / challenges / no "
            "change) citing the specific evidence, including watch-list companies. Say what to watch next."
        )
    if name == "review_decisions":
        return (
            "How have my decisions worked out? Call review_journal and walk me through the biggest wins "
            "and misses in dollars. Separate what the market did from what my picks did, using the "
            "market-adjusted moves, and point out any pattern (e.g. selling winners too early)."
        )
    raise ValueError(f"Unknown prompt '{name}'")


def _tool_list() -> list[types.Tool]:
    tools = []
    for t in TOOLS + MEMORY_TOOLS + WRITE_TOOLS:
        name = t["name"]
        description = _CHART_DESCRIPTION if name == "get_chart" else t["description"]
        writes = name in WRITE_TOOL_NAMES
        tools.append(types.Tool(
            name=name,
            title=name.removeprefix("get_").replace("_", " ").capitalize(),
            description=description,
            inputSchema=t["input_schema"],
            annotations=types.ToolAnnotations(
                read_only_hint=not writes,
                destructive_hint=name == "delete_trade",
                idempotent_hint=not writes,
                open_world_hint=not writes,
            ),
        ))
    return tools


def create_server(holder: AnalyticsHolder) -> Server:
    tools = _tool_list()

    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        def run():
            try:
                analytics = holder.get()
            except Exception as exc:
                return None, json.dumps({"error": f"Portfolio data unavailable: {exc}"}), True
            outcome = run_tool(analytics, params.name, params.arguments or {}, store=holder.store)
            if outcome.changed:
                holder.invalidate()
            # MCP clients render charts themselves, so return the data, not a "shown" note.
            payload = outcome.chart if outcome.chart is not None else outcome.result
            text = json.dumps(payload, default=str) if payload is not None else outcome.content
            return payload, text, outcome.is_error

        payload, text, is_error = await anyio.to_thread.run_sync(run)
        return types.CallToolResult(
            content=[types.TextContent(text=text)],
            structuredContent=payload if isinstance(payload, dict) else None,
            isError=is_error,
        )

    async def list_prompts(ctx, params) -> types.ListPromptsResult:
        return types.ListPromptsResult(prompts=_PROMPTS)

    async def get_prompt(ctx, params: types.GetPromptRequestParams) -> types.GetPromptResult:
        text = _prompt_text(params.name, params.arguments or {})
        return types.GetPromptResult(
            messages=[types.PromptMessage(role="user", content=types.TextContent(text=text))])

    return Server(
        "desk-note",
        version="1.0.0",
        title="Desk Note",
        description="Your portfolio's morning note, on demand.",
        instructions=INSTRUCTIONS,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        on_list_prompts=list_prompts,
        on_get_prompt=get_prompt,
    )


class TokenGate:
    """ASGI middleware: allow ``/<token>/mcp`` (rewritten to ``/mcp``) or a
    matching ``Authorization: Bearer`` header; everything else gets 401."""

    def __init__(self, app, token: str | None):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or self.token is None:
            return await self.app(scope, receive, send)
        path: str = scope.get("path", "")
        _, first, rest = (path.split("/", 2) + ["", ""])[:3]
        if rest and hmac.compare_digest(first.encode(), self.token.encode()):
            inner = "/" + rest
            scope = {**scope, "path": inner, "raw_path": inner.encode()}
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers") or [])
        auth = headers.get(b"authorization", b"").decode()
        if auth.startswith("Bearer ") and hmac.compare_digest(auth[7:].strip(), self.token):
            return await self.app(scope, receive, send)
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{"error":"unauthorized"}'})


def create_http_app(holder: AnalyticsHolder, token: str | None):
    """Streamable HTTP app at /mcp, behind the token gate. ``token=None`` disables
    auth (local testing only)."""
    server = create_server(holder)
    # host="0.0.0.0" skips the SDK's localhost-only Host check: behind a tunnel the
    # Host header is the public hostname. The token gate does the protecting.
    app = server.streamable_http_app(streamable_http_path="/mcp", stateless_http=True,
                                     json_response=True, host="0.0.0.0")
    return TokenGate(app, token)


async def _run_stdio(holder: AnalyticsHolder) -> None:
    from mcp.server.stdio import stdio_server

    server = create_server(holder)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser(description="Run Desk Note as an MCP server.")
    ap.add_argument("--mock", action="store_true", help="synthetic data (no network for prices)")
    ap.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--no-auth", action="store_true", help="HTTP without a token (local testing only)")
    ap.add_argument("--new-token", action="store_true", help="print a random token and exit")
    args = ap.parse_args(argv)

    if args.new_token:
        print(secrets.token_urlsafe(32))
        return 0

    holder = AnalyticsHolder(mock=args.mock)
    if not args.http:
        anyio.run(_run_stdio, holder)
        return 0

    import uvicorn

    token = os.getenv("DESK_NOTE_MCP_TOKEN") or None
    if token is None and not args.no_auth:
        print("Set DESK_NOTE_MCP_TOKEN (generate one with --new-token) or pass --no-auth for local testing.",
              file=sys.stderr)
        return 2
    if token is not None and len(token) < 24:
        print("DESK_NOTE_MCP_TOKEN is too short; use at least 24 random characters (--new-token).", file=sys.stderr)
        return 2
    path = f"/{token}/mcp" if token else "/mcp"
    print(f"Desk Note MCP ({'mock' if args.mock else 'live'} data) on http://{args.host}:{args.port}{path}",
          file=sys.stderr)
    uvicorn.run(create_http_app(holder, token), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
