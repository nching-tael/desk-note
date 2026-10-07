# Running Desk Note on a Raspberry Pi

Run Desk Note on a Pi and add it to Claude as a connector. Claude on your phone, desktop or the web can then pull your portfolio's numbers and record your trades, while Claude's own web search covers the news.

```
Claude app ──HTTPS──▶ Tailscale Funnel ──▶ Pi: Desk Note MCP (127.0.0.1:8001) ──▶ Yahoo Finance
 (phone, web,          (public URL,          SQLite: trades, theses, journal
  desktop)              TLS for you)
```

Claude calls connectors from Anthropic's servers, not from your phone, so the endpoint has to be reachable from the internet. Tailscale Funnel provides a public HTTPS address without opening ports on your router. A random token in the URL keeps everyone else out.

## What you need

- A Raspberry Pi 4 or 5 (2 GB RAM is fine) running **64-bit** Raspberry Pi OS Bookworm (Python 3.11)
- A free [Tailscale](https://tailscale.com) account
- A Claude plan that supports custom connectors (Pro, Max, Team or Enterprise)

## 1. Install

```bash
sudo apt update && sudo apt install -y git python3-venv
git clone https://github.com/nching-tael/desk-note.git ~/desk-note
cd ~/desk-note
nano portfolio.csv theses.yaml       # your holdings and theses (first run only, see below)
./deploy/install.sh
```

The script:
1. creates a virtualenv and installs the dependencies;
2. generates an access token into `.env`;
3. seeds `data/desknote.db` from your CSV and YAML;
4. installs and starts the `desk-note-mcp` systemd service, listening on `127.0.0.1:8001` only.

It's safe to re-run: it keeps your token and database.

## 2. Publish it with Tailscale Funnel

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
sudo tailscale funnel --bg 8001
tailscale funnel status              # shows https://<pi>.<tailnet>.ts.net
```

The first time, Tailscale may ask you to enable HTTPS certificates and Funnel for your tailnet. Follow the link it prints.

## 3. Add the connector in Claude

On claude.ai, open **Settings → Connectors → Add custom connector**:

- **Name:** Desk Note
- **URL:** `https://<pi>.<tailnet>.ts.net/<token>/mcp` (the token is `DESK_NOTE_MCP_TOKEN` in `~/desk-note/.env`)

Connectors you add on claude.ai also show up in the Claude mobile and desktop apps. Turn Desk Note on from the tools menu in a chat, then try:

- *Why did I lose money this week?*
- *I sold 20 NVDA at $182 to take profits after the capex news.*
- *Does the latest news break my TSMC thesis?* (Claude searches the web and checks your thesis)
- *How have my decisions worked out this year?*

The connector also offers three ready-made prompts: **Morning note**, **Check a thesis** and **Review my decisions**.

## Day to day

| Task | Command |
|---|---|
| Status / logs | `systemctl status desk-note-mcp` · `journalctl -u desk-note-mcp -f` |
| What's stored | `.venv/bin/python -m app.store show` |
| Update the code | `git pull && ./deploy/install.sh` |
| Back up | copy `data/desknote.db` (trades, theses, journal) and `.env` |
| Rotate the token | edit `DESK_NOTE_MCP_TOKEN` in `.env`, `sudo systemctl restart desk-note-mcp`, update the connector URL |
| Start over from the CSV/YAML | `.venv/bin/python -m app.store reseed --force` (deletes trades and journal) |

`portfolio.csv` and `theses.yaml` are only read the first time, to seed the database. After that, record changes by telling Claude, or edit the files and reseed.

## Security notes

- The token in the URL is the only thing protecting your data. Anyone with the URL can read your holdings and record trades in your database. Desk Note never touches a real brokerage account. Treat the URL like a password, and rotate the token if it leaks.
- The service runs as your user with a read-only filesystem, apart from `data/` and `.cache/`, and listens on localhost only. Only Funnel exposes it.
- Upgrade path: replace the URL token with OAuth, which the MCP spec and Claude connectors support, if you share the Pi with others or want per-device revocation.

## Without a Pi

The same server runs anywhere:

- **Claude Desktop** (local, no network exposure). Add to `claude_desktop_config.json`:
  ```json
  {"mcpServers": {"desk-note": {"command": "/path/to/desk-note/.venv/bin/python",
                                "args": ["-m", "app.mcp_server"],
                                "env": {"PYTHONPATH": "/path/to/desk-note"}}}}
  ```
- **Claude Code:** `claude mcp add desk-note -- /path/to/desk-note/.venv/bin/python -m app.mcp_server`
- **Demo data:** add `--mock` to any of these to use the synthetic portfolio.
