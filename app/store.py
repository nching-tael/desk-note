"""SQLite storage for what Desk Note remembers: opening positions, trades,
theses and the decision journal.

The database is seeded once from portfolio.csv and theses.yaml. After that it's
the source of truth, and changes come in through the tools. Mock mode uses a
fresh in-memory database every time.

    python -m app.store show
    python -m app.store reseed --force    # start over from the CSV/YAML files
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import UTC, date, datetime
from pathlib import Path

from .portfolio import (
    DEFAULT_PORTFOLIO,
    DEFAULT_THESES,
    ROOT,
    Holding,
    PortfolioError,
    Trade,
    load_holdings,
    load_theses,
    replay,
)

DEFAULT_DB = ROOT / "data" / "desknote.db"
JOURNAL_KINDS = ("buy_reason", "sell_reason", "thesis_change", "note")

SCHEMA = """
CREATE TABLE IF NOT EXISTS opening (
    symbol TEXT PRIMARY KEY,
    shares REAL NOT NULL CHECK (shares > 0),
    cost_basis REAL
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    shares REAL NOT NULL CHECK (shares > 0),
    price REAL,
    note TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS theses (
    symbol TEXT PRIMARY KEY,
    thesis TEXT NOT NULL DEFAULT '',
    breaks_if TEXT NOT NULL DEFAULT '[]',
    watch TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    symbol TEXT,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    trade_id INTEGER REFERENCES trades(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS targets (
    symbol TEXT PRIMARY KEY,
    target_pct REAL NOT NULL CHECK (target_pct >= 0),
    band_pct REAL NOT NULL CHECK (band_pct >= 0),
    role TEXT NOT NULL CHECK (role IN ('core', 'satellite')),
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class StoreError(ValueError):
    pass


def now():
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_symbol(symbol):
    cleaned = str(symbol or "").strip().upper()
    if not cleaned or len(cleaned) > 12 or not all(c.isalnum() or c in ".-^=" for c in cleaned):
        raise StoreError(f"'{symbol}' doesn't look like a ticker symbol.")
    return cleaned


class Store:
    def __init__(self, path=":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.RLock()

    # --- seeding

    @property
    def seeded(self):
        return self.db.execute("SELECT 1 FROM meta WHERE key = 'seeded_at'").fetchone() is not None

    def seed(self, holdings, theses, force=False):
        """Load opening positions and theses into an empty store. With
        force=True, wipe everything first, including trades and the journal."""
        with self.lock, self.db:
            if self.seeded and not force:
                return False
            for table in ("journal", "trades", "opening", "theses", "targets"):
                self.db.execute(f"DELETE FROM {table}")
            self.db.executemany(
                "INSERT INTO opening VALUES (?, ?, ?)", [(h.symbol, h.shares, h.cost_basis) for h in holdings]
            )
            for symbol, t in theses.items():
                self.db.execute(
                    "INSERT INTO theses VALUES (?, ?, ?, ?, ?)",
                    (symbol, t["thesis"], json.dumps(t["breaks_if"]), json.dumps(t["watch"]), now()),
                )
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('seeded_at', ?)", (now(),))
        return True

    def seed_from_files(self, portfolio=DEFAULT_PORTFOLIO, theses=DEFAULT_THESES, force=False):
        if self.seeded and not force:
            return False
        return self.seed(load_holdings(portfolio), load_theses(theses), force=force)

    # --- reads

    def opening(self):
        rows = self.db.execute("SELECT * FROM opening ORDER BY symbol")
        return [Holding(r["symbol"], r["shares"], r["cost_basis"]) for r in rows]

    def trades(self, symbol=None):
        if symbol:
            rows = self.db.execute(
                "SELECT * FROM trades WHERE symbol = ? ORDER BY date, id", (symbol.upper(),)
            )
        else:
            rows = self.db.execute("SELECT * FROM trades ORDER BY date, id")
        return [
            Trade(
                date.fromisoformat(r["date"]),
                r["symbol"],
                r["side"],
                r["shares"],
                r["price"],
                r["id"],
                r["note"],
            )
            for r in rows
        ]

    def theses(self):
        rows = self.db.execute("SELECT * FROM theses ORDER BY symbol")
        return {
            r["symbol"]: {
                "thesis": r["thesis"],
                "breaks_if": json.loads(r["breaks_if"]),
                "watch": json.loads(r["watch"]),
            }
            for r in rows
        }

    def journal(self, symbol=None, limit=50):
        query = """
            SELECT j.*, t.side, t.shares, t.price
            FROM journal j LEFT JOIN trades t ON t.id = j.trade_id
            WHERE ? IS NULL OR j.symbol = ?
            ORDER BY j.date DESC, j.id DESC
            LIMIT ?
        """
        symbol = symbol.upper() if symbol else None
        entries = []
        for r in self.db.execute(query, (symbol, symbol, int(limit))):
            entry = {
                "id": r["id"],
                "date": r["date"],
                "symbol": r["symbol"],
                "kind": r["kind"],
                "text": r["text"],
            }
            if r["trade_id"] is not None:
                entry["trade"] = {
                    "id": r["trade_id"],
                    "side": r["side"],
                    "shares": r["shares"],
                    "price": r["price"],
                }
            entries.append(entry)
        return entries

    def targets(self):
        rows = self.db.execute("SELECT * FROM targets ORDER BY target_pct DESC")
        return {
            r["symbol"]: {"target_pct": r["target_pct"], "band_pct": r["band_pct"], "role": r["role"]}
            for r in rows
        }

    # --- writes

    def add_trade(self, trade, reason=None):
        """Save a trade (and the reason for it, if given). Refuses anything that
        would leave the history inconsistent, like selling shares you didn't have."""
        if trade.side not in ("buy", "sell"):
            raise StoreError("side must be 'buy' or 'sell'.")
        if not trade.shares or trade.shares <= 0:
            raise StoreError("shares must be positive.")
        if trade.price is not None and trade.price <= 0:
            raise StoreError("price must be positive.")
        if trade.date > date.today():
            raise StoreError("Trade date can't be in the future.")
        trade = Trade(
            trade.date,
            clean_symbol(trade.symbol),
            trade.side,
            float(trade.shares),
            trade.price,
            None,
            trade.note,
        )

        with self.lock, self.db:
            self.check_history(self.trades() + [trade])
            cursor = self.db.execute(
                "INSERT INTO trades (date, symbol, side, shares, price, note, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    trade.date.isoformat(),
                    trade.symbol,
                    trade.side,
                    trade.shares,
                    trade.price,
                    trade.note,
                    now(),
                ),
            )
            if reason and reason.strip():
                kind = "buy_reason" if trade.side == "buy" else "sell_reason"
                self.insert_journal(trade.date, trade.symbol, kind, reason, cursor.lastrowid)
        return Trade(
            trade.date, trade.symbol, trade.side, trade.shares, trade.price, cursor.lastrowid, trade.note
        )

    def delete_trade(self, trade_id):
        with self.lock, self.db:
            trades = self.trades()
            match = [t for t in trades if t.id == int(trade_id)]
            if not match:
                raise StoreError(f"No trade with id {trade_id}.")
            try:
                self.check_history([t for t in trades if t.id != int(trade_id)])
            except StoreError as e:
                raise StoreError(f"Deleting it would break later trades: {e}") from e
            self.db.execute("DELETE FROM trades WHERE id = ?", (int(trade_id),))
        return match[0]

    def check_history(self, trades):
        try:
            replay(self.opening(), trades)
        except PortfolioError as e:
            raise StoreError(str(e)) from e

    def set_targets(self, targets, replace=True):
        """targets: [{symbol, target_pct, role?, band_pct?}]. With replace, the
        new set must add up to 100%. Otherwise listed symbols are updated and the
        rest kept, as long as the total doesn't go over 100%."""
        if not targets:
            raise StoreError("Give at least one target.")
        cleaned = {}
        for t in targets:
            symbol = clean_symbol(t.get("symbol"))
            target = float(t.get("target_pct", -1))
            if not 0 <= target <= 100:
                raise StoreError(f"{symbol}: target_pct must be between 0 and 100.")
            role = t.get("role") or ("core" if target >= 40 else "satellite")
            if role not in ("core", "satellite"):
                raise StoreError(f"{symbol}: role must be 'core' or 'satellite'.")
            band = t.get("band_pct")
            band = float(band) if band is not None else default_band(target)
            cleaned[symbol] = {"target_pct": target, "band_pct": band, "role": role}

        with self.lock, self.db:
            combined = cleaned if replace else {**self.targets(), **cleaned}
            total = sum(t["target_pct"] for t in combined.values())
            if replace and abs(total - 100) > 0.5:
                raise StoreError(f"Targets add up to {total:g}%, not 100%.")
            if total > 100.5:
                raise StoreError(f"Targets would add up to {total:g}%, over 100%.")
            if replace:
                self.db.execute("DELETE FROM targets")
            for symbol, t in cleaned.items():
                self.db.execute(
                    "INSERT OR REPLACE INTO targets VALUES (?, ?, ?, ?, ?)",
                    (symbol, t["target_pct"], t["band_pct"], t["role"], now()),
                )
        return self.targets()

    def update_thesis(self, symbol, thesis=None, breaks_if=None, watch_add=None, watch_remove=None):
        symbol = clean_symbol(symbol)
        with self.lock, self.db:
            current = self.theses().get(symbol, {"thesis": "", "breaks_if": [], "watch": []})
            if thesis is not None:
                current["thesis"] = " ".join(thesis.split())
            if breaks_if is not None:
                current["breaks_if"] = [b.strip() for b in breaks_if if b and b.strip()]

            watch = current["watch"]
            for s in map(clean_symbol, watch_add or []):
                if s not in watch and s != symbol:
                    watch.append(s)
            remove = {clean_symbol(s) for s in watch_remove or []}
            current["watch"] = [s for s in watch if s not in remove]

            if not (current["thesis"] or current["breaks_if"] or current["watch"]):
                raise StoreError(
                    "A thesis needs at least a description, a breaks_if item or a watch-list symbol."
                )
            self.db.execute(
                "INSERT OR REPLACE INTO theses VALUES (?, ?, ?, ?, ?)",
                (
                    symbol,
                    current["thesis"],
                    json.dumps(current["breaks_if"]),
                    json.dumps(current["watch"]),
                    now(),
                ),
            )
        return {"symbol": symbol, **current}

    def add_journal(self, text, symbol=None, kind="note", on=None):
        if not text or not text.strip():
            raise StoreError("Journal entry text is empty.")
        if kind not in JOURNAL_KINDS:
            raise StoreError(f"kind must be one of: {', '.join(JOURNAL_KINDS)}.")
        symbol = clean_symbol(symbol) if symbol else None
        with self.lock, self.db:
            entry_id = self.insert_journal(on or date.today(), symbol, kind, text, None)
        return next(e for e in self.journal(limit=500) if e["id"] == entry_id)

    def insert_journal(self, on, symbol, kind, text, trade_id):
        cursor = self.db.execute(
            "INSERT INTO journal (date, symbol, kind, text, trade_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (on.isoformat(), symbol, kind, text.strip()[:4000], trade_id, now()),
        )
        return cursor.lastrowid


def default_band(target_pct):
    """How far a position can drift before it's flagged: 5 points for big
    positions, 2 for small ones."""
    return 5.0 if target_pct >= 20 else 2.0


def open_store(mock, path=None):
    """Mock mode: a fresh in-memory store from the sample files. Otherwise the
    database at DESK_NOTE_DB (default data/desknote.db), seeded if it's new."""
    store = Store(":memory:" if mock else path or os.getenv("DESK_NOTE_DB") or DEFAULT_DB)
    store.seed_from_files()
    return store


def main():
    import argparse

    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Inspect or reseed the Desk Note database.")
    parser.add_argument("command", choices=["show", "reseed"])
    parser.add_argument("--force", action="store_true", help="needed for reseed; deletes trades and journal")
    parser.add_argument("--portfolio", default=DEFAULT_PORTFOLIO, help="CSV to reseed from")
    parser.add_argument("--theses", default=DEFAULT_THESES, help="YAML to reseed from")
    args = parser.parse_args()

    store = Store(os.getenv("DESK_NOTE_DB") or DEFAULT_DB)
    if args.command == "reseed":
        if not args.force:
            print(
                "reseed replaces all trades, theses and journal entries with the CSV/YAML files. "
                "Run it again with --force to go ahead."
            )
            return 1
        store.seed_from_files(args.portfolio, args.theses, force=True)
        print(f"Reseeded {store.path} from {args.portfolio} and {args.theses}.")
        return 0

    store.seed_from_files()
    print(f"Database: {store.path}")
    print("Opening positions:", ", ".join(f"{h.symbol} {h.shares:g}" for h in store.opening()))
    for t in store.trades():
        print(f"  trade #{t.id}  {t.date}  {t.side} {t.shares:g} {t.symbol} @ {t.price}")
    print("Theses:", ", ".join(store.theses()))
    print("Journal entries:", len(store.journal(limit=100_000)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
