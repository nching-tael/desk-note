"""SQLite storage for the things Desk Note remembers: opening positions, trades,
theses and the decision journal.

On first run the store is seeded from portfolio.csv (opening positions) and
theses.yaml. After that the database is the source of truth and changes come
through the tools ("I sold half my AMD", "add ORCL to the NVDA watch list").
Mock mode uses an in-memory database seeded fresh on every start.

    python -m app.store show               # print what's stored
    python -m app.store reseed --force     # replace everything with the CSV/YAML files
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .portfolio import (DEFAULT_PORTFOLIO, DEFAULT_THESES, ROOT, Holding, PortfolioError, Trade,
                        load_holdings, load_theses, replay)

DEFAULT_DB = ROOT / "data" / "desknote.db"
JOURNAL_KINDS = ("buy_reason", "sell_reason", "thesis_change", "note")

_SCHEMA = """
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
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class StoreError(ValueError):
    """Invalid write (e.g. selling more than you hold). Message is user-safe."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean_symbol(symbol: Any) -> str:
    sym = str(symbol or "").strip().upper()
    if not sym or len(sym) > 12 or not all(c.isalnum() or c in ".-^=" for c in sym):
        raise StoreError(f"'{symbol}' doesn't look like a ticker symbol.")
    return sym


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(_SCHEMA)

    # ------------------------------------------------------------- seeding

    @property
    def seeded(self) -> bool:
        row = self._db.execute("SELECT value FROM meta WHERE key = 'seeded_at'").fetchone()
        return row is not None

    def seed(self, holdings: list[Holding], theses: dict[str, dict[str, Any]], force: bool = False) -> bool:
        """Load opening positions and theses. Without ``force`` this only runs
        on an empty store; with it, everything (including trades and the
        journal) is replaced."""
        with self._lock, self._db:
            if self.seeded and not force:
                return False
            for table in ("journal", "trades", "opening", "theses"):
                self._db.execute(f"DELETE FROM {table}")
            self._db.executemany("INSERT INTO opening VALUES (?, ?, ?)",
                                 [(h.symbol, h.shares, h.cost_basis) for h in holdings])
            self._db.executemany(
                "INSERT INTO theses VALUES (?, ?, ?, ?, ?)",
                [(s, t["thesis"], json.dumps(t["breaks_if"]), json.dumps(t["watch"]), _now())
                 for s, t in theses.items()])
            self._db.execute("INSERT OR REPLACE INTO meta VALUES ('seeded_at', ?)", (_now(),))
        return True

    def seed_from_files(self, portfolio_path=DEFAULT_PORTFOLIO, theses_path=DEFAULT_THESES,
                        force: bool = False) -> bool:
        if self.seeded and not force:
            return False
        return self.seed(load_holdings(portfolio_path), load_theses(theses_path), force=force)

    # --------------------------------------------------------------- reads

    def opening(self) -> list[Holding]:
        rows = self._db.execute("SELECT symbol, shares, cost_basis FROM opening ORDER BY symbol").fetchall()
        return [Holding(r["symbol"], r["shares"], r["cost_basis"]) for r in rows]

    def trades(self, symbol: str | None = None) -> list[Trade]:
        sql = "SELECT * FROM trades" + (" WHERE symbol = ?" if symbol else "") + " ORDER BY date, id"
        rows = self._db.execute(sql, (symbol.upper(),) if symbol else ()).fetchall()
        return [Trade(date.fromisoformat(r["date"]), r["symbol"], r["side"], r["shares"], r["price"],
                      r["id"], r["note"]) for r in rows]

    def theses(self) -> dict[str, dict[str, Any]]:
        rows = self._db.execute("SELECT * FROM theses ORDER BY symbol").fetchall()
        return {r["symbol"]: {"thesis": r["thesis"], "breaks_if": json.loads(r["breaks_if"]),
                              "watch": json.loads(r["watch"])} for r in rows}

    def journal(self, symbol: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql = """SELECT j.*, t.side, t.shares, t.price FROM journal j
                 LEFT JOIN trades t ON t.id = j.trade_id"""
        args: tuple = ()
        if symbol:
            sql += " WHERE j.symbol = ?"
            args = (symbol.upper(),)
        sql += " ORDER BY j.date DESC, j.id DESC LIMIT ?"
        rows = self._db.execute(sql, args + (int(limit),)).fetchall()
        out = []
        for r in rows:
            entry = {"id": r["id"], "date": r["date"], "symbol": r["symbol"], "kind": r["kind"],
                     "text": r["text"]}
            if r["trade_id"] is not None:
                entry["trade"] = {"id": r["trade_id"], "side": r["side"], "shares": r["shares"], "price": r["price"]}
            out.append(entry)
        return out

    # -------------------------------------------------------------- writes

    def add_trade(self, trade: Trade, reason: str | None = None) -> Trade:
        """Record a trade; optionally journal the reason. Validates that the
        full history still makes sense (no selling shares you didn't have)."""
        symbol = _clean_symbol(trade.symbol)
        if trade.side not in ("buy", "sell"):
            raise StoreError("side must be 'buy' or 'sell'.")
        if not trade.shares or trade.shares <= 0:
            raise StoreError("shares must be positive.")
        if trade.price is not None and trade.price <= 0:
            raise StoreError("price must be positive.")
        if trade.date > date.today():
            raise StoreError("Trade date can't be in the future.")
        trade = Trade(trade.date, symbol, trade.side, float(trade.shares), trade.price, None, trade.note)
        with self._lock, self._db:
            try:
                replay(self.opening(), self.trades() + [trade])
            except PortfolioError as exc:
                raise StoreError(str(exc)) from exc
            cur = self._db.execute(
                "INSERT INTO trades (date, symbol, side, shares, price, note, created_at) VALUES (?,?,?,?,?,?,?)",
                (trade.date.isoformat(), symbol, trade.side, trade.shares, trade.price, trade.note, _now()))
            saved = Trade(trade.date, symbol, trade.side, trade.shares, trade.price, cur.lastrowid, trade.note)
            if reason and reason.strip():
                self._insert_journal(trade.date, symbol, "buy_reason" if trade.side == "buy" else "sell_reason",
                                     reason, cur.lastrowid)
        return saved

    def delete_trade(self, trade_id: int) -> Trade:
        with self._lock, self._db:
            existing = [t for t in self.trades() if t.id == int(trade_id)]
            if not existing:
                raise StoreError(f"No trade with id {trade_id}.")
            remaining = [t for t in self.trades() if t.id != int(trade_id)]
            try:
                replay(self.opening(), remaining)
            except PortfolioError as exc:
                raise StoreError(f"Deleting it would break later trades: {exc}") from exc
            self._db.execute("DELETE FROM trades WHERE id = ?", (int(trade_id),))
        return existing[0]

    def update_thesis(self, symbol: str, thesis: str | None = None, breaks_if: list[str] | None = None,
                      watch_add: list[str] | None = None, watch_remove: list[str] | None = None) -> dict[str, Any]:
        sym = _clean_symbol(symbol)
        with self._lock, self._db:
            current = self.theses().get(sym, {"thesis": "", "breaks_if": [], "watch": []})
            if thesis is not None:
                current["thesis"] = " ".join(thesis.split())
            if breaks_if is not None:
                current["breaks_if"] = [b.strip() for b in breaks_if if b and b.strip()]
            watch = list(current["watch"])
            for w in watch_add or []:
                w = _clean_symbol(w)
                if w not in watch and w != sym:
                    watch.append(w)
            removed = {_clean_symbol(w) for w in watch_remove or []}
            current["watch"] = [w for w in watch if w not in removed]
            if not current["thesis"] and not current["breaks_if"] and not current["watch"]:
                raise StoreError("A thesis needs at least a description, a breaks_if item or a watch-list symbol.")
            self._db.execute("INSERT OR REPLACE INTO theses VALUES (?, ?, ?, ?, ?)",
                             (sym, current["thesis"], json.dumps(current["breaks_if"]),
                              json.dumps(current["watch"]), _now()))
        return {"symbol": sym, **current}

    def add_journal(self, text: str, symbol: str | None = None, kind: str = "note",
                    on: date | None = None) -> dict[str, Any]:
        if not text or not text.strip():
            raise StoreError("Journal entry text is empty.")
        if kind not in JOURNAL_KINDS:
            raise StoreError(f"kind must be one of: {', '.join(JOURNAL_KINDS)}.")
        sym = _clean_symbol(symbol) if symbol else None
        with self._lock, self._db:
            entry_id = self._insert_journal(on or date.today(), sym, kind, text, None)
        return next(e for e in self.journal(limit=500) if e["id"] == entry_id)

    def _insert_journal(self, on: date, symbol: str | None, kind: str, text: str, trade_id: int | None) -> int:
        cur = self._db.execute(
            "INSERT INTO journal (date, symbol, kind, text, trade_id, created_at) VALUES (?,?,?,?,?,?)",
            (on.isoformat(), symbol, kind, text.strip()[:4000], trade_id, _now()))
        return cur.lastrowid


def open_store(mock: bool, path: str | Path | None = None) -> Store:
    """Mock: fresh in-memory store from the sample files. Live: the database at
    DESK_NOTE_DB (default data/desknote.db), seeded from the files if empty."""
    if mock:
        store = Store(":memory:")
    else:
        store = Store(path or os.getenv("DESK_NOTE_DB") or DEFAULT_DB)
    store.seed_from_files()
    return store


def main(argv: list[str] | None = None) -> int:
    import argparse

    from dotenv import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser(description="Inspect or reseed the Desk Note database.")
    ap.add_argument("command", choices=["show", "reseed"])
    ap.add_argument("--force", action="store_true", help="required for reseed: replaces trades and journal too")
    args = ap.parse_args(argv)
    store = Store(os.getenv("DESK_NOTE_DB") or DEFAULT_DB)
    if args.command == "reseed":
        if not args.force:
            print("reseed replaces all trades, theses and journal entries with the CSV/YAML files. "
                  "Re-run with --force to do it.")
            return 1
        store.seed_from_files(force=True)
        print(f"Reseeded {store.path} from portfolio.csv and theses.yaml.")
        return 0
    store.seed_from_files()
    print(f"Database: {store.path}")
    print(f"Opening positions: {', '.join(f'{h.symbol} {h.shares:g}' for h in store.opening())}")
    for t in store.trades():
        print(f"  trade #{t.id} {t.date} {t.side} {t.shares:g} {t.symbol} @ {t.price}")
    print(f"Theses: {', '.join(store.theses())}")
    print(f"Journal entries: {len(store.journal(limit=10000))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
