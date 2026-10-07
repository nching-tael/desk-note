#!/usr/bin/env bash
# Set up the Desk Note MCP server on a Raspberry Pi (or any Debian-like Linux).
# Run from the repo root:  ./deploy/install.sh
# Safe to re-run: it keeps an existing .env, token and database.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="$(id -un)"
cd "$DIR"

echo "==> Desk Note in $DIR (user: $USER_NAME)"

if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
  echo "Python 3.11+ is required (Raspberry Pi OS Bookworm ships 3.11)." >&2
  exit 1
fi

echo "==> Python environment"
python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

echo "==> Configuration"
mkdir -p data .cache
touch .env
chmod 600 .env
if ! grep -q '^DESK_NOTE_MCP_TOKEN=.\+' .env; then
  echo "DESK_NOTE_MCP_TOKEN=$(.venv/bin/python -m app.mcp_server --new-token)" >> .env
  echo "    generated a new access token in .env"
fi

echo "==> Database"
.venv/bin/python -m app.store show | head -3

echo "==> systemd service"
sed -e "s|__USER__|$USER_NAME|g" -e "s|__DIR__|$DIR|g" deploy/desk-note-mcp.service \
  | sudo tee /etc/systemd/system/desk-note-mcp.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now desk-note-mcp
sudo systemctl restart desk-note-mcp
sleep 3
if curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:8001/mcp | grep -q 401; then
  echo "    running (unauthenticated requests are refused, as they should be)"
else
  echo "    the service didn't answer; check: journalctl -u desk-note-mcp -n 50" >&2
  exit 1
fi

TOKEN="$(grep '^DESK_NOTE_MCP_TOKEN=' .env | tail -1 | cut -d= -f2-)"
cat <<EOF

Done. Next, publish it with Tailscale Funnel (free, HTTPS, no router setup):

  curl -fsSL https://tailscale.com/install.sh | sh     # if Tailscale isn't installed
  sudo tailscale up
  sudo tailscale funnel --bg 8001

Then add a custom connector in Claude (Settings > Connectors) with this URL,
replacing the host with the one 'tailscale funnel status' prints:

  https://<your-pi>.<your-tailnet>.ts.net/$TOKEN/mcp

Keep that URL private: the token in it is the password to your portfolio.
EOF
