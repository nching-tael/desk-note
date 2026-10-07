# Desk Note

[![tests](https://github.com/nching-tael/desk-note/actions/workflows/tests.yml/badge.svg)](https://github.com/nching-tael/desk-note/actions/workflows/tests.yml)

**Your portfolio's morning note, on demand.** Desk Note is a personal analyst for your stock portfolio. Ask questions in plain English, like *"Why did I lose money this week?"*, *"What's my biggest risk?"* or *"Does today's news break my NVDA thesis?"*. Every number in the answer comes from a real calculation on your actual holdings, never from the model's own arithmetic.

You can use it two ways:

- **Inside Claude.** Desk Note runs as an MCP server, for example on a Raspberry Pi, and plugs into Claude as a connector. Claude on your phone or desktop searches the web for the story, while Desk Note supplies the numbers, records your trades and keeps a decision journal.
- **As a standalone web app** with its own Claude agent loop, charts, and a "How I got this" trace under every answer.

![The web app answering "Why did I lose money this week?" in mock mode](docs/screenshot.png)

<sub>Screenshots use mock mode (synthetic prices and news). The figures come from the real tool outputs; the answer text was scripted for the capture. [Demo GIF](docs/demo.gif).</sub>

## What you can ask

| You say | Desk Note does |
|---|---|
| *Why did I lose money this week?* | Splits the week's $ change into **market, sector and your stock picks**, using a two-factor model per holding |
| *What's my biggest risk?* | Concentration, correlated groups ("42% of your money is in four chip stocks that move together"), volatility, beta, a −10% market stress test |
| *Does today's news break my NVDA thesis?* | Pulls your thesis, what would break it, the stock-specific move, and news for NVDA **and its watch list** (customers, suppliers, rivals). Gives a verdict: supports, challenges or no change |
| *What earnings are coming up?* | Dates plus the options-implied move, in % and in $ on your position |
| *I sold 20 NVDA at $182 to take profits.* | Records the trade, updates your holdings and cost basis, and journals your reason |
| *Add Oracle to my NVDA watch list.* | Edits the thesis |
| *How have my decisions worked out?* | Reviews your journal. For each buy or sell, how the stock moved since, against the market (beta-adjusted) and in dollars |

In mock mode, the attribution tool's own headline for the first question is:

> Of your -$6,348 over the last 5 trading days, -$4,136 was the market, -$753 was sector moves (Technology -$1,198; Energy +$324), and -$1,459 was your stock picks (stock-specific moves).

## Use it in Claude

The MCP server exposes 14 tools (8 read-only analytics, 2 for trade history and the journal, 4 that record changes) and three ready-made prompts: *Morning note*, *Check a thesis* and *Review my decisions*.

- **Claude app on your phone, from a Raspberry Pi:** `./deploy/install.sh` sets up a hardened systemd service, and Tailscale Funnel gives it a public HTTPS address. See **[docs/raspberry-pi.md](docs/raspberry-pi.md)**.
- **Claude Desktop or Claude Code (local):** run it over stdio. See [the end of the Pi guide](docs/raspberry-pi.md#without-a-pi).

```bash
python -m app.mcp_server --mock                 # stdio
python -m app.mcp_server --http --port 8001     # remote; needs DESK_NOTE_MCP_TOKEN
```

## Run the web app

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add your ANTHROPIC_API_KEY
python run.py --mock          # http://127.0.0.1:8000
```

`--mock` uses deterministic synthetic data and needs no network for prices. Without an API key the page still loads, but chat stays off. Drop `--mock` to use live Yahoo Finance data.

```bash
python -m app.agent "why did I lose money this week?" --mock   # ask from the terminal
python -m app.analytics --mock 1w                              # attribution + risk, no AI involved
```

Settings (`.env`):
- `ANTHROPIC_API_KEY`: web app chat only
- `ANTHROPIC_MODEL`: default `claude-sonnet-5-5`
- `ANTHROPIC_EFFORT`: default `medium`
- `DESK_NOTE_MCP_TOKEN`: remote MCP only
- `DESK_NOTE_DB`: default `data/desknote.db`

## Your own portfolio

1. **`portfolio.csv`:** your positions, with average cost per share (optional).
   ```csv
   symbol,shares,cost_basis
   NVDA,120,48.20
   ```
2. **`theses.yaml`:** why you own each stock, what would prove you wrong, and companies to watch. Watch-list news is how Desk Note spots read-across, such as a big customer cutting spending.
   ```yaml
   NVDA:
     thesis: Hyperscaler data center capex keeps growing through 2027.
     breaks_if: [Big cloud customers cut or slow data center spending plans]
     watch: [MSFT, META, GOOGL, AMZN, TSM]
   ```

On first run these seed a SQLite database (`data/desknote.db`). From then on, tell Claude about trades and thesis changes and it records them. Run `python -m app.store show` to see what's stored.

## How it works

```
                    ┌──────────────── tools.py (14 tools, one dispatcher) ────────────────┐
Claude app ── MCP ──┤                                                                      │
Web UI ── agent.py ─┤   analytics.py (all maths)   store.py (SQLite)   data.py (Yahoo/mock)│
                    └──────────────────────────────────────────────────────────────────────┘
```

**The model never does the maths.** Every figure comes from a Python function. Claude decides which tools to call and explains the results. The system prompt (web app) and the server instructions (MCP) forbid stating any number that didn't come from a tool. Tools return ready-made sentences, like the attribution headline above, so there's nothing left to work out. LLMs are unreliable at arithmetic and confidently recall stale prices. Keeping computation in code makes every figure reproducible and testable.

**Every answer is traceable.** The web app lists each tool call, its inputs and a key result under the answer. Claude's own interface shows the same for the connector.

**Attribution.** For each holding, a two-factor regression is fit on the year of daily returns before the period: `return = a + b_mkt × SPY + b_sec × (sector ETF − SPY)`. Here the sector ETF is the SPDR fund for the stock's sector. Each day's return is split into market, sector and stock-specific parts (the residual, including the intercept). Each part is multiplied by the previous day's position value and summed. The parts add up exactly to the dollar gain or loss. "Stock-specific" is the part of the move the market and sector don't explain: the closest thing to measuring your picks rather than your luck.

**Trades and returns.** Positions are rebuilt day by day from your opening holdings plus recorded trades, and gains come from positions held overnight. That means buying more shares isn't counted as profit, returns are time-weighted, and a stock you sold mid-week still shows what it made before you sold. Cost basis uses average cost, and sells book realised P/L.

**Decision journal.** When you say *why* you traded, the reason is saved with the trade. Reviewing the journal compares each decision with what the stock did since, beta-adjusted for the market, so a buy that rose only because everything rose doesn't look like skill.

**Risk.** One year of daily returns at current weights: annualised volatility, beta to SPY, each holding's share of risk (`wᵢ(Σw)ᵢ / wᵀΣw`), effective number of positions (`1/Σw²`), pairs correlated above 0.7, groups that move together, 1-year max drawdown, and a SPY −10% stress test.

**Web agent loop.** Plain Anthropic SDK tool use: run every requested tool, return all the results in one message, repeat. The loop is capped at 8 rounds, and tool errors go back to the model as results instead of crashing the request.

**Code map:**
- `app/data.py`: providers
- `app/portfolio.py`: files, trades, cost basis
- `app/store.py`: SQLite
- `app/analytics.py`: all the maths
- `app/tools.py`: schemas and dispatch
- `app/mcp_server.py`: MCP over stdio or HTTP
- `app/agent.py`, `app/server.py`, `app/static/index.html`: the web app
- `deploy/`: Pi install and systemd
- `tests/`: 83 tests, all offline: `python -m pytest -q`

## Limitations

- **Yahoo Finance data quality.** yfinance is an unofficial API. Prices can lag, sectors can be missing (those holdings fall back to a market-only model), and options data may be missing (no implied move). Its news feed is often empty, so Desk Note falls back to Yahoo search and then RSS. In Claude, its web search covers the news properly.
- **Trades execute at the close.** A trade affects positions from the close of its date, and same-day intraday gains aren't counted. Opening positions are assumed held through the whole price history. Dividends and cash aren't modelled.
- **Simple models.** Betas are estimates. A move across the whole semiconductor industry counts as "stock-specific" because the model has no industry factor. Implied moves use an at-the-money straddle approximation.
- **Token auth.** The remote MCP endpoint is protected by a secret token in its URL. That's fine for one person; use OAuth before sharing it.
- **Not investment advice.** Desk Note lays out facts and considerations. It doesn't recommend trades, and it never connects to a brokerage.

## Next steps

- OAuth for the remote connector
- Broker CSV import (Schwab, Fidelity, IBKR) to backfill trade history
- ETF look-through, so VOO counts as its underlying sectors
- SEC filings and insider trades as a stored, searchable source
- Thesis alerts when watch-list news challenges a thesis

## License

MIT
