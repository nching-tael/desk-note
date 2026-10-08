"""Desk Note as an MCP server, so Claude itself (desktop, mobile, Claude Code)
can use the portfolio tools.

    python -m app.mcp_server --mock               # stdio, for Claude Desktop / Claude Code
    python -m app.mcp_server --http --port 8001   # remote connector, e.g. on a Raspberry Pi
    python -m app.mcp_server --new-token          # generate an access token

Claude calls remote connectors from Anthropic's servers, so over HTTP the
endpoint has to be public. DESK_NOTE_MCP_TOKEN keeps it private: send it as
a path prefix (https://host/<token>/mcp, which is what the Claude app's
connector form needs) or as an "Authorization: Bearer" header.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import secrets
import sys

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

Desk Note also remembers things for the user. When they tell you about a trade they've made \
("I sold half my AMD at 160"), record it with record_trade (confirm the share count if it's \
ambiguous), and pass their reason if they gave one. Edit theses with update_thesis when asked. \
Use review_journal to show how past decisions have worked out.
"""

# Claude draws its own charts, so get_chart returns data rather than "chart shown to user".
CHART_DESCRIPTION = (
    "Chart-ready data: labels plus one or more series, with a unit ($ or %). Kinds: "
    "'portfolio_vs_market' (portfolio value vs SPY scaled to the same start), 'stock_vs_market' "
    "(one symbol vs SPY, both rebased to 0%; needs symbol), 'attribution' (per holding: market, "
    "sector and stock-specific $). Draw it for the user if your interface can render charts."
)

PROMPTS = {
    "morning_note": (
        "Morning note",
        "A short briefing: what moved, why, and what's coming up.",
        "Give me my morning note. Use get_overview and get_attribution for 1d to say what my portfolio "
        "did and why (market vs sector vs my picks, in dollars). For the two or three biggest "
        "stock-specific movers, check get_news and search the web for what happened. Then list earnings "
        "in the next 7 days with get_upcoming_events, and flag any news that challenges one of my "
        "theses. Keep it to a short paragraph and a few bullets.",
    ),
    "check_thesis": (
        "Check a thesis",
        "Does recent news support or challenge the thesis for one holding?",
        "Does recent news break my {symbol} thesis? Call get_thesis for {symbol}, search the web for "
        "anything newer than its headlines, and give a verdict (supports / challenges / no change) "
        "citing the specific evidence, including watch-list companies. Say what to watch next.",
    ),
    "review_decisions": (
        "Review my decisions",
        "How have the trades and calls in my journal worked out, luck vs skill?",
        "How have my decisions worked out? Call review_journal and walk me through the biggest wins and "
        "misses in dollars. Separate what the market did from what my picks did, using the "
        "market-adjusted moves, and point out any pattern (e.g. selling winners too early).",
    ),
}


def tool_list():
    tools = []
    for schema in TOOLS + MEMORY_TOOLS + WRITE_TOOLS:
        name = schema["name"]
        writes = name in WRITE_TOOL_NAMES
        tools.append(
            types.Tool(
                name=name,
                title=name.removeprefix("get_").replace("_", " ").capitalize(),
                description=CHART_DESCRIPTION if name == "get_chart" else schema["description"],
                inputSchema=schema["input_schema"],
                annotations=types.ToolAnnotations(
                    read_only_hint=not writes,
                    destructive_hint=name == "delete_trade",
                    idempotent_hint=not writes,
                    open_world_hint=not writes,
                ),
            )
        )
    return tools


def prompt_list():
    prompts = []
    for name, (title, description, template) in PROMPTS.items():
        arguments = None
        if "{symbol}" in template:
            arguments = [types.PromptArgument(name="symbol", description="Ticker, e.g. NVDA", required=True)]
        prompts.append(types.Prompt(name=name, title=title, description=description, arguments=arguments))
    return prompts


def create_server(holder):
    tools = tool_list()
    prompts = prompt_list()

    def call(name, arguments):
        """Runs in a worker thread: the analytics are plain blocking code."""
        try:
            analytics = holder.get()
        except Exception as e:
            return None, json.dumps({"error": f"Portfolio data unavailable: {e}"}), True

        outcome = run_tool(analytics, name, arguments, store=holder.store)
        if outcome.changed:
            holder.invalidate()
        data = outcome.chart or outcome.result
        text = json.dumps(data, default=str) if data is not None else outcome.content
        return data, text, outcome.is_error

    async def list_tools(ctx, params):
        return types.ListToolsResult(tools=tools)

    async def call_tool(ctx, params):
        data, text, is_error = await anyio.to_thread.run_sync(call, params.name, params.arguments or {})
        return types.CallToolResult(
            content=[types.TextContent(text=text)], structuredContent=data, isError=is_error
        )

    async def list_prompts(ctx, params):
        return types.ListPromptsResult(prompts=prompts)

    async def get_prompt(ctx, params):
        if params.name not in PROMPTS:
            raise ValueError(f"Unknown prompt '{params.name}'")
        template = PROMPTS[params.name][2]
        symbol = (params.arguments or {}).get("symbol", "").strip().upper()
        if "{symbol}" in template and not symbol:
            raise ValueError(f"{params.name} needs a symbol")
        text = template.format(symbol=symbol)
        return types.GetPromptResult(
            messages=[types.PromptMessage(role="user", content=types.TextContent(text=text))]
        )

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
    """ASGI middleware that only lets through /<token>/... (rewritten to /...)
    or requests with "Authorization: Bearer <token>". Everything else is a 401."""

    def __init__(self, app, token):
        self.app = app
        self.token = token.encode() if token else None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or self.token is None:
            return await self.app(scope, receive, send)

        # /<token>/mcp
        prefix, _, rest = scope["path"].lstrip("/").partition("/")
        if rest and hmac.compare_digest(prefix.encode(), self.token):
            path = "/" + rest
            return await self.app({**scope, "path": path, "raw_path": path.encode()}, receive, send)

        # Authorization: Bearer <token>
        auth = dict(scope["headers"]).get(b"authorization", b"")
        if auth.startswith(b"Bearer ") and hmac.compare_digest(auth[7:].strip(), self.token):
            return await self.app(scope, receive, send)

        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b'{"error": "unauthorized"}'})


def create_http_app(holder, token):
    """Streamable HTTP at /mcp behind the token gate (token=None: no auth)."""
    server = create_server(holder)
    # host="0.0.0.0" turns off the SDK's localhost-only Host header check. Behind a
    # tunnel the Host is the public hostname; the token is what protects us.
    app = server.streamable_http_app(
        streamable_http_path="/mcp", stateless_http=True, json_response=True, host="0.0.0.0"
    )
    return TokenGate(app, token)


async def serve_stdio(holder):
    from mcp.server.stdio import stdio_server

    server = create_server(holder)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main():
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Run Desk Note as an MCP server.")
    parser.add_argument("--mock", action="store_true", help="use synthetic market data")
    parser.add_argument("--http", action="store_true", help="serve over HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--no-auth", action="store_true", help="HTTP without a token, for local testing")
    parser.add_argument("--new-token", action="store_true", help="print a random token and exit")
    args = parser.parse_args()

    if args.new_token:
        print(secrets.token_urlsafe(32))
        return 0

    holder = AnalyticsHolder(mock=args.mock)
    if not args.http:
        anyio.run(serve_stdio, holder)
        return 0

    token = os.getenv("DESK_NOTE_MCP_TOKEN")
    if not token and not args.no_auth:
        sys.exit("Set DESK_NOTE_MCP_TOKEN (make one with --new-token), or use --no-auth for local testing.")
    if token and len(token) < 24:
        sys.exit("DESK_NOTE_MCP_TOKEN is too short. Use --new-token to generate a proper one.")

    import uvicorn

    path = f"/{token}/mcp" if token else "/mcp"
    data = "mock" if args.mock else "live"
    print(f"Desk Note MCP ({data} data) on http://{args.host}:{args.port}{path}", file=sys.stderr)
    uvicorn.run(create_http_app(holder, token), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
