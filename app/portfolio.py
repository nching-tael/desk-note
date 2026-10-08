"""Holdings, theses and trades: loading the seed files and replaying trades."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORTFOLIO = ROOT / "portfolio.csv"
DEFAULT_THESES = ROOT / "theses.yaml"


class PortfolioError(ValueError):
    pass


@dataclass(frozen=True)
class Holding:
    symbol: str
    shares: float
    cost_basis: float | None  # average cost per share


@dataclass(frozen=True)
class Trade:
    """A buy or sell after the opening positions. It takes effect at the
    close of its date."""

    date: date
    symbol: str
    side: str  # "buy" or "sell"
    shares: float
    price: float | None = None
    id: int | None = None
    note: str | None = None


@dataclass
class Position:
    shares: float
    avg_cost: float | None
    realised: float = 0.0


def parse_number(text):
    text = (text or "").strip().replace(",", "").replace("$", "")
    return float(text) if text else None


def load_holdings(path=DEFAULT_PORTFOLIO):
    """Read portfolio.csv (symbol, shares, cost_basis). Rows for the same
    symbol are merged."""
    path = Path(path)
    if not path.exists():
        raise PortfolioError(f"Portfolio file not found: {path}")

    positions = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = {c.strip().lower() for c in reader.fieldnames or []}
        if not {"symbol", "shares"} <= columns:
            raise PortfolioError("portfolio.csv needs at least 'symbol' and 'shares' columns")

        for line, row in enumerate(reader, start=2):
            row = {(k or "").strip().lower(): v for k, v in row.items()}
            symbol = (row.get("symbol") or "").strip().upper()
            if not symbol or symbol.startswith("#"):
                continue
            try:
                shares = parse_number(row.get("shares"))
                cost = parse_number(row.get("cost_basis"))
            except ValueError as e:
                raise PortfolioError(f"Line {line}: {e}") from e
            if not shares:
                continue

            if symbol in positions:
                prev = positions[symbol]
                total = prev.shares + shares
                if prev.avg_cost is not None and cost is not None:
                    cost = (prev.shares * prev.avg_cost + shares * cost) / total
                else:
                    cost = None
                shares = total
            positions[symbol] = Position(shares, cost)

    if not positions:
        raise PortfolioError("portfolio.csv has no holdings")
    return [Holding(symbol, p.shares, p.avg_cost) for symbol, p in positions.items()]


def load_theses(path=DEFAULT_THESES):
    """Read theses.yaml into {SYMBOL: {"thesis", "breaks_if", "watch"}}."""
    path = Path(path)
    if not path.exists():
        return {}

    theses = {}
    for symbol, entry in (yaml.safe_load(path.read_text()) or {}).items():
        entry = entry or {}
        breaks_if = entry.get("breaks_if") or []
        if isinstance(breaks_if, str):
            breaks_if = [breaks_if]
        watch = entry.get("watch") or []
        if isinstance(watch, str):
            watch = watch.split(",")

        theses[str(symbol).upper()] = {
            "thesis": " ".join(str(entry.get("thesis") or "").split()),
            "breaks_if": [str(b).strip() for b in breaks_if if str(b).strip()],
            "watch": [str(w).strip().upper() for w in watch if str(w).strip()],
        }
    return theses


def replay(opening, trades):
    """Apply trades in date order to the opening positions, using average
    cost. Returns {symbol: Position}. Raises if a sell exceeds what was held."""
    book = {h.symbol: Position(h.shares, h.cost_basis) for h in opening}

    for t in sorted(trades, key=lambda t: (t.date, t.id or 0)):
        pos = book.setdefault(t.symbol, Position(0.0, None))

        if t.side == "buy":
            if pos.shares == 0:
                pos.avg_cost = t.price
            elif pos.avg_cost is not None and t.price is not None:
                pos.avg_cost = (pos.avg_cost * pos.shares + t.price * t.shares) / (pos.shares + t.shares)
            else:
                pos.avg_cost = None
            pos.shares += t.shares

        elif t.side == "sell":
            if t.shares > pos.shares + 1e-9:
                raise PortfolioError(
                    f"Can't sell {t.shares:g} {t.symbol} on {t.date}: only {pos.shares:g} held then."
                )
            if pos.avg_cost is not None and t.price is not None:
                pos.realised += (t.price - pos.avg_cost) * t.shares
            pos.shares = max(pos.shares - t.shares, 0.0)
            if pos.shares < 1e-9:
                pos.shares = 0.0

        else:
            raise PortfolioError(f"Unknown trade side '{t.side}'")

    return book
