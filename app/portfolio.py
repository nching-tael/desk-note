"""Load holdings (portfolio.csv) and investment theses (theses.yaml)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORTFOLIO = ROOT / "portfolio.csv"
DEFAULT_THESES = ROOT / "theses.yaml"


@dataclass(frozen=True)
class Holding:
    symbol: str
    shares: float
    cost_basis: float | None  # average cost per share


class PortfolioError(ValueError):
    pass


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("$", "")
    if not text:
        return None
    return float(text)


def load_holdings(path: str | Path = DEFAULT_PORTFOLIO) -> list[Holding]:
    """Read ``symbol,shares,cost_basis`` rows. ``cost_basis`` is the average
    cost per share and may be blank. Duplicate symbols are merged (cost is
    share-weighted)."""
    path = Path(path)
    if not path.exists():
        raise PortfolioError(f"Portfolio file not found: {path}")
    merged: dict[str, tuple[float, float | None]] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = {f.strip().lower() for f in reader.fieldnames or []}
        if not {"symbol", "shares"} <= fields:
            raise PortfolioError("portfolio.csv needs at least 'symbol' and 'shares' columns")
        for lineno, row in enumerate(reader, start=2):
            row = {(k or "").strip().lower(): v for k, v in row.items()}
            symbol = (row.get("symbol") or "").strip().upper()
            if not symbol or symbol.startswith("#"):
                continue
            try:
                shares = _to_float(row.get("shares"))
                cost = _to_float(row.get("cost_basis"))
            except ValueError as exc:
                raise PortfolioError(f"Line {lineno}: {exc}") from exc
            if not shares:
                continue
            if symbol in merged:
                old_shares, old_cost = merged[symbol]
                total = old_shares + shares
                if old_cost is not None and cost is not None and total:
                    cost = (old_shares * old_cost + shares * cost) / total
                else:
                    cost = None
                shares = total
            merged[symbol] = (shares, cost)
    if not merged:
        raise PortfolioError("portfolio.csv has no holdings")
    return [Holding(sym, sh, cb) for sym, (sh, cb) in merged.items()]


def load_theses(path: str | Path = DEFAULT_THESES) -> dict[str, dict[str, Any]]:
    """Return ``{SYMBOL: {"thesis": str, "breaks_if": [str], "watch": [str]}}``.
    A missing file yields an empty dict."""
    path = Path(path)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    theses: dict[str, dict[str, Any]] = {}
    for symbol, entry in raw.items():
        entry = entry or {}
        breaks = entry.get("breaks_if") or []
        if isinstance(breaks, str):
            breaks = [breaks]
        watch = entry.get("watch") or []
        if isinstance(watch, str):
            watch = [w.strip() for w in watch.split(",")]
        theses[str(symbol).upper()] = {
            "thesis": " ".join(str(entry.get("thesis") or "").split()),
            "breaks_if": [str(b).strip() for b in breaks if str(b).strip()],
            "watch": [str(w).strip().upper() for w in watch if str(w).strip()],
        }
    return theses


@dataclass(frozen=True)
class Trade:
    """A buy or sell after the opening positions in portfolio.csv. Trades take
    effect at the close of ``date`` (or the next session if it isn't one)."""
    date: date
    symbol: str
    side: str  # "buy" | "sell"
    shares: float
    price: float | None = None
    id: int | None = None
    note: str | None = None


@dataclass
class Position:
    shares: float
    avg_cost: float | None  # average cost per share; None if unknown
    realised: float = 0.0  # realised P/L from sells, in $


def replay(opening: list[Holding], trades: list[Trade]) -> dict[str, Position]:
    """Apply trades (in date order) to the opening positions using average cost.
    Raises PortfolioError if a sell exceeds the shares held at that point."""
    book: dict[str, Position] = {h.symbol: Position(h.shares, h.cost_basis) for h in opening}
    for t in sorted(trades, key=lambda t: (t.date, t.id or 0)):
        pos = book.setdefault(t.symbol, Position(0.0, None))
        if t.side == "buy":
            if pos.shares <= 1e-9:
                pos.avg_cost = t.price
            elif pos.avg_cost is not None and t.price is not None:
                pos.avg_cost = (pos.avg_cost * pos.shares + t.price * t.shares) / (pos.shares + t.shares)
            else:
                pos.avg_cost = None
            pos.shares += t.shares
        elif t.side == "sell":
            if t.shares > pos.shares + 1e-9:
                raise PortfolioError(
                    f"Can't sell {t.shares:g} {t.symbol} on {t.date}: only {pos.shares:g} held then.")
            if pos.avg_cost is not None and t.price is not None:
                pos.realised += (t.price - pos.avg_cost) * t.shares
            pos.shares -= t.shares
            if pos.shares <= 1e-9:
                pos.shares = 0.0
        else:
            raise PortfolioError(f"Unknown trade side '{t.side}'")
    return book
