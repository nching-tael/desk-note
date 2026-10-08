"""All of Desk Note's maths.

Every number the agent reports comes from one of these functions. They return
plain JSON-friendly dicts: dollars rounded to whole numbers, percentages to one
decimal place (expressed as percent, e.g. -2.1 means -2.1%).

Positions are valued at *current* share counts throughout, so historical
figures answer "what did my current portfolio do", not "what did my account do".
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .data import MARKET, DataProvider, sector_etf
from .portfolio import Holding, Trade, replay

TRADING_DAYS = 252
FIT_WINDOW = 252  # sessions of returns used to fit attribution betas
MIN_FIT_OBS = 40

PERIOD_SESSIONS = {"1d": 1, "1w": 5, "1m": 21, "3m": 63, "1y": 252}
PERIODS = ["1d", "1w", "1m", "3m", "ytd", "1y"]
PERIOD_LABELS = {
    "1d": "last trading day",
    "1w": "last 5 trading days",
    "1m": "last month",
    "3m": "last 3 months",
    "ytd": "year to date",
    "1y": "last year",
}
_PERIOD_ALIASES = {
    "day": "1d", "today": "1d", "1day": "1d", "d": "1d",
    "week": "1w", "1week": "1w", "w": "1w", "5d": "1w",
    "month": "1m", "1month": "1m", "m": "1m",
    "3month": "3m", "3months": "3m", "quarter": "3m",
    "year": "1y", "1year": "1y", "12m": "1y", "y": "1y",
    "year_to_date": "ytd",
}


class AnalyticsError(ValueError):
    """Raised for bad inputs (unknown period or symbol). Message is user-safe."""


def normalise_period(period: str | None, default: str = "1w") -> str:
    p = (period or default).strip().lower().replace(" ", "").replace("-", "")
    p = _PERIOD_ALIASES.get(p, p)
    if p not in PERIODS:
        raise AnalyticsError(f"Unknown period '{period}'. Use one of: {', '.join(PERIODS)}.")
    return p


def usd(x: float) -> int:
    return int(round(float(x))) if np.isfinite(x) else 0


def pct(fraction: float) -> float | None:
    """Fraction (0.021) -> percent rounded to 1 decimal (2.1)."""
    if fraction is None or not np.isfinite(fraction):
        return None
    value = round(float(fraction) * 100, 1)
    return 0.0 if value == 0 else value


def fmt_usd(x: float) -> str:
    """-1240.4 -> '-$1,240'."""
    v = usd(x)
    return f"-${abs(v):,}" if v < 0 else f"+${v:,}" if v > 0 else "$0"


@dataclass
class _Window:
    period: str
    start: pd.Timestamp  # close the period is measured from
    end: pd.Timestamp
    start_pos: int
    end_pos: int
    sessions: int

    @property
    def label(self) -> str:
        return PERIOD_LABELS[self.period]


class Analytics:
    def __init__(self, provider: DataProvider, holdings: list[Holding],
                 theses: dict[str, dict[str, Any]] | None = None, trades: list[Trade] | None = None):
        """``holdings`` are the opening positions (assumed held throughout the
        price history); ``trades`` are later buys and sells."""
        self.provider = provider
        self.holdings = holdings
        self.theses = theses or {}
        self.trades = sorted(trades or [], key=lambda t: (t.date, t.id or 0))
        self._lock = threading.Lock()
        self._loaded = False
        self.warnings: list[str] = []

    # ------------------------------------------------------------------ data

    def load(self) -> "Analytics":
        with self._lock:
            if not self._loaded:
                self._load()
                self._loaded = True
        return self

    def _load(self) -> None:
        as_of = self.provider.as_of()
        start = as_of - timedelta(days=2 * 365 + 45)
        symbols = list(dict.fromkeys([h.symbol for h in self.holdings] + [t.symbol for t in self.trades]))

        self.profiles: dict[str, dict[str, Any]] = {}
        self.sector_of: dict[str, str | None] = {}
        self.etf_of: dict[str, str | None] = {}
        for sym in symbols:
            prof = self.provider.profile(sym)
            self.profiles[sym] = prof
            sector = prof.get("sector")
            if not sector and (prof.get("quote_type") or "").upper() == "ETF":
                sector = "ETF"
            self.sector_of[sym] = sector or "Unknown"
            self.etf_of[sym] = sector_etf(sector)
        etfs = sorted({e for e in self.etf_of.values() if e})

        closes = self.provider.prices(sorted(set(symbols) | set(etfs) | {MARKET}), start)
        closes = closes[closes.index <= pd.Timestamp(as_of)]
        if MARKET not in closes or closes[MARKET].dropna().empty:
            raise RuntimeError("No market (SPY) price data available; cannot run analytics.")
        closes = closes[closes[MARKET].notna()].sort_index()
        closes = closes.ffill(limit=5)

        held = []
        listed: dict[str, int] = {}  # position of each symbol's first real price
        for sym in symbols:
            series = closes.get(sym)
            if series is None or series.dropna().empty:
                self.warnings.append(f"No price data for {sym}; it is excluded from the analysis.")
                continue
            if series.isna().any():
                first = series.first_valid_index()
                if first is not None and first > closes.index[0] + pd.Timedelta(days=400):
                    self.warnings.append(f"{sym} has less than a year of price history.")
                closes[sym] = series.bfill()  # flat before listing: no fake returns
            listed[sym] = int(closes.index.get_loc(series.first_valid_index()))
            held.append(sym)
        for etf in etfs:
            if closes.get(etf) is None or closes[etf].dropna().empty:
                self.warnings.append(f"No data for sector ETF {etf}; using market-only model for its holdings.")
                for sym, e in self.etf_of.items():
                    if e == etf:
                        self.etf_of[sym] = None
            else:
                closes[etf] = closes[etf].bfill()
        if not held:
            raise RuntimeError("None of the holdings have price data.")

        # Shares held at each close: opening positions throughout, then each
        # trade from the close of its date onwards.
        shares = pd.DataFrame(0.0, index=closes.index, columns=held)
        for h in self.holdings:
            if h.symbol in shares:
                shares[h.symbol] += h.shares
        for t in self.trades:
            if t.symbol in shares:
                sign = 1.0 if t.side == "buy" else -1.0
                shares.loc[shares.index >= pd.Timestamp(t.date), t.symbol] += sign * t.shares
        shares = shares.clip(lower=0.0)
        book = replay(self.holdings, [t for t in self.trades if t.symbol in held])

        self.listed_pos = listed
        self.all_symbols = held  # everything held at some point in the history
        self.shares_df = shares
        self.shares = shares.iloc[-1]
        self.symbols = [s for s in held if self.shares[s] > 1e-9]  # held today
        if not self.symbols:
            raise RuntimeError("No current holdings with price data.")
        self.closes = closes
        self.returns = closes.pct_change().fillna(0.0)
        self.cost = {s: book[s].avg_cost if s in book else None for s in held}
        self.realised = {s: book[s].realised for s in held if s in book and book[s].realised}
        self.values = closes[held] * shares
        self.port = self.values.sum(axis=1)
        # Daily P/L from positions held overnight; trades don't count as gains.
        self.pnl_df = (shares.shift(1) * closes[held].diff()).fillna(0.0)
        self.pnl = self.pnl_df.sum(axis=1)
        prev = self.port.shift(1)
        self.port_ret = (self.pnl / prev).where(prev > 0, 0.0).fillna(0.0)

    @property
    def as_of(self) -> str:
        self.load()
        return self.closes.index[-1].date().isoformat()

    def _days(self, w: _Window) -> slice:
        """Positions of the sessions whose returns make up the window."""
        return slice(w.start_pos + 1, w.end_pos + 1)

    def _window_pnl(self, w: _Window) -> float:
        return float(self.pnl.iloc[self._days(w)].sum())

    def _window_twr(self, w: _Window) -> float:
        """Time-weighted return: unaffected by buying or selling during the window."""
        return float((1 + self.port_ret.iloc[self._days(w)]).prod() - 1)

    def name(self, symbol: str) -> str:
        return (self.profiles.get(symbol) or {}).get("name") or symbol

    def window(self, period: str) -> _Window:
        self.load()
        p = normalise_period(period)
        idx = self.closes.index
        last = len(idx) - 1
        if p == "ytd":
            year = idx[-1].year
            n = int((idx.year == year).sum())
        else:
            n = PERIOD_SESSIONS[p]
        n = max(1, min(n, last))
        return _Window(p, idx[last - n], idx[last], last - n, last, n)

    # -------------------------------------------------------------- overview

    def overview(self) -> dict[str, Any]:
        self.load()
        v = self.values
        total = float(self.port.iloc[-1])
        day = self.window("1d")
        week = self.window("1w")

        def change(w: _Window) -> dict[str, Any]:
            return {"dollars": usd(self._window_pnl(w)), "pct": pct(self._window_twr(w))}

        positions = []
        cost_total = 0.0
        pl_total = 0.0
        for sym in self.symbols:
            value = float(v[sym].iloc[-1])
            price = float(self.closes[sym].iloc[-1])
            prev = float(self.closes[sym].iloc[-2])
            cost = self.cost.get(sym)
            pl = pl_pct = None
            if cost:
                basis = cost * self.shares[sym]
                pl = value - basis
                pl_pct = pct(pl / basis)
                cost_total += basis
                pl_total += pl
            positions.append({
                "symbol": sym,
                "name": self.name(sym),
                "sector": self.sector_of[sym],
                "shares": float(self.shares[sym]),
                "price": round(price, 2),
                "value": usd(value),
                "weight_pct": pct(value / total),
                "day_dollars": usd(float(self.pnl_df[sym].iloc[-1])),
                "day_pct": pct(price / prev - 1),
                "unrealised_pl": usd(pl) if pl is not None else None,
                "unrealised_pl_pct": pl_pct,
            })
        positions.sort(key=lambda p: p["value"], reverse=True)
        out = {
            "as_of": self.as_of,
            "data_source": self.provider.name,
            "total_value": usd(total),
            "day_change": change(day),
            "week_change": change(week),
            "unrealised_pl": {
                "dollars": usd(pl_total),
                "pct": pct(pl_total / cost_total) if cost_total else None,
                "cost_basis_total": usd(cost_total),
            },
            "positions": positions,
            "warnings": list(self.warnings),
        }
        if self.realised:
            out["realised_pl"] = {"dollars": usd(sum(self.realised.values())),
                                  "by_symbol": {s: usd(v) for s, v in self.realised.items()}}
        return out

    # ----------------------------------------------------------- performance

    def _held_during(self, w: _Window) -> list[str]:
        """Symbols with a position at some close the window's returns start from."""
        prev = self.shares_df.iloc[w.start_pos:w.end_pos]
        return [s for s in self.all_symbols if (prev[s] > 1e-9).any()]

    def performance(self, period: str = "1w") -> dict[str, Any]:
        w = self.window(period)
        days = self._days(w)
        start_total = float(self.port.iloc[w.start_pos])
        end_total = float(self.port.iloc[w.end_pos])
        pnl_total = self._window_pnl(w)
        positions = []
        for sym in self._held_during(w):
            held = self.shares_df[sym].shift(1).iloc[days] > 1e-9
            r = self.returns[sym].iloc[days].where(held, 0.0)
            positions.append({
                "symbol": sym,
                "name": self.name(sym),
                "dollars": usd(float(self.pnl_df[sym].iloc[days].sum())),
                "pct": pct(float((1 + r).prod() - 1)),
            })
        positions.sort(key=lambda p: p["dollars"])
        spy = self.closes[MARKET]
        spy_ret = float(spy.iloc[w.end_pos] / spy.iloc[w.start_pos] - 1)
        port_ret = self._window_twr(w)
        out = {
            "period": w.period,
            "period_label": w.label,
            "start_date": w.start.date().isoformat(),
            "end_date": w.end.date().isoformat(),
            "sessions": w.sessions,
            "start_value": usd(start_total),
            "end_value": usd(end_total),
            "change_dollars": usd(pnl_total),
            "change_pct": pct(port_ret),
            "market_spy_pct": pct(spy_ret),
            "vs_market_pct_points": round((port_ret - spy_ret) * 100, 1),
            "if_tracked_market_dollars": usd(start_total * spy_ret),
            "best": positions[-1],
            "worst": positions[0],
            "positions": sorted(positions, key=lambda p: p["dollars"], reverse=True),
        }
        flows = end_total - start_total - pnl_total
        if abs(flows) >= 1:
            out["net_purchases_dollars"] = usd(flows)
            out["note"] = ("change_dollars is gain/loss only; the value also changed by net_purchases_dollars "
                           "from buying and selling. change_pct is time-weighted.")
        return out

    # ----------------------------------------------------------- attribution

    def _fit_betas(self, sym: str, w: _Window) -> dict[str, Any]:
        """Two-factor fit on the FIT_WINDOW sessions before the period:
        r = a + b_mkt * SPY + b_sec * (sectorETF - SPY)."""
        r = self.returns
        # Skip the back-filled flat stretch before a stock started trading; a
        # genuine unchanged close (return 0) is real data and stays in.
        lo = max(1, w.start_pos - FIT_WINDOW + 1, self.listed_pos.get(sym, 0) + 1)
        hi = w.start_pos + 1  # returns up to and including the period's start close
        y = r[sym].iloc[lo:hi].to_numpy()
        m = r[MARKET].iloc[lo:hi].to_numpy()
        etf = self.etf_of.get(sym)
        cols = [np.ones_like(m), m]
        if etf:
            cols.append(r[etf].iloc[lo:hi].to_numpy() - m)
        X = np.column_stack(cols)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if ok.sum() < MIN_FIT_OBS:
            return {"alpha": 0.0, "beta_mkt": 1.0, "beta_sector": 0.0, "etf": etf,
                    "obs": int(ok.sum()), "fallback": True}
        coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        return {"alpha": float(coef[0]), "beta_mkt": float(coef[1]),
                "beta_sector": float(coef[2]) if etf else 0.0, "etf": etf,
                "obs": int(ok.sum()), "fallback": False}

    def attribution_frame(self, period: str = "1w") -> tuple[pd.DataFrame, _Window]:
        """Unrounded $ attribution per holding: columns market, sector,
        stock_specific, total (plus betas)."""
        w = self.window(period)
        r = self.returns
        rows = []
        days = slice(w.start_pos + 1, w.end_pos + 1)
        m = r[MARKET].iloc[days].to_numpy()
        for sym in self._held_during(w):
            fit = self._fit_betas(sym, w)
            prev_value = self.values[sym].iloc[w.start_pos:w.end_pos].to_numpy()
            exposure = prev_value[0] if prev_value[0] > 0 else prev_value[prev_value > 0].mean()
            ret = r[sym].iloc[days].to_numpy()
            sec = (r[fit["etf"]].iloc[days].to_numpy() - m) if fit["etf"] else np.zeros_like(m)
            market_ret = fit["beta_mkt"] * m
            sector_ret = fit["beta_sector"] * sec
            specific_ret = ret - market_ret - sector_ret  # includes the intercept
            rows.append({
                "symbol": sym,
                "sector": self.sector_of[sym],
                "market": float((market_ret * prev_value).sum()),
                "sector_move": float((sector_ret * prev_value).sum()),
                "stock_specific": float((specific_ret * prev_value).sum()),
                "total": float((ret * prev_value).sum()),
                "start_value": float(exposure),  # first (or average) position value, for %
                "beta_mkt": fit["beta_mkt"],
                "beta_sector": fit["beta_sector"],
                "sector_etf": fit["etf"],
                "fallback": fit["fallback"],
            })
        return pd.DataFrame(rows).set_index("symbol"), w

    def attribution(self, period: str = "1w") -> dict[str, Any]:
        df, w = self.attribution_frame(period)
        total = df["total"].sum()
        mkt = df["market"].sum()
        sec = df["sector_move"].sum()
        spec = df["stock_specific"].sum()

        by_sector = (df.groupby("sector")["sector_move"].sum()
                     .sort_values(key=lambda s: -s.abs()))
        top_sectors = [(k, v) for k, v in by_sector.head(2).items() if abs(v) >= 1]
        sector_note = f" ({'; '.join(f'{k} {fmt_usd(v)}' for k, v in top_sectors)})" if top_sectors else ""
        headline = (f"Of your {fmt_usd(total)} over the {w.label}, {fmt_usd(mkt)} was the market, "
                    f"{fmt_usd(sec)} was sector moves{sector_note}, and {fmt_usd(spec)} was your "
                    f"stock picks (stock-specific moves).")

        holdings = []
        for sym, row in df.sort_values("total").iterrows():
            holdings.append({
                "symbol": sym,
                "sector": row["sector"],
                "total": usd(row["total"]),
                "market": usd(row["market"]),
                "sector_move": usd(row["sector_move"]),
                "stock_specific": usd(row["stock_specific"]),
                "stock_specific_pct": pct(row["stock_specific"] / row["start_value"]),
                "beta_mkt": round(row["beta_mkt"], 2),
                "beta_sector": round(row["beta_sector"], 2),
                "sector_etf": row["sector_etf"] if isinstance(row["sector_etf"], str) else None,
            })
        top_specific = df.reindex(df["stock_specific"].abs().sort_values(ascending=False).index)
        out = {
            "period": w.period,
            "period_label": w.label,
            "start_date": w.start.date().isoformat(),
            "end_date": w.end.date().isoformat(),
            "sessions": w.sessions,
            "headline": headline,
            "total": usd(total),
            "market": usd(mkt),
            "sector": usd(sec),
            "stock_specific": usd(spec),
            "sector_by_name": {k: usd(v) for k, v in by_sector.items()},
            "biggest_stock_specific": [
                {"symbol": s, "dollars": usd(r["stock_specific"]),
                 "pct": pct(r["stock_specific"] / r["start_value"])}
                for s, r in top_specific.head(3).iterrows()
            ],
            "holdings": holdings,
            "method": ("Per holding: return = a + b_mkt*SPY + b_sec*(sector ETF - SPY), fit on the "
                       "year before the period. Daily components x previous day's position value, summed. "
                       "Stock-specific = residual including intercept. Uses current share counts."),
        }
        fallbacks = df.index[df["fallback"]].tolist()
        if fallbacks:
            out["note"] = f"Not enough history to fit betas for {', '.join(fallbacks)}; assumed beta 1 to the market."
        return out

    # ------------------------------------------------------------------ risk

    def _weights(self) -> pd.Series:
        v = self.values.iloc[-1][self.symbols]
        return v / v.sum()

    def risk(self) -> dict[str, Any]:
        self.load()
        w = self._weights()
        rets = self.returns[self.symbols].iloc[-TRADING_DAYS:]
        spy = self.returns[MARKET].iloc[-TRADING_DAYS:]
        cov = rets.cov() * TRADING_DAYS
        wv = w.to_numpy()
        port_var = float(wv @ cov.to_numpy() @ wv)
        port_vol = float(np.sqrt(port_var))
        port_daily = rets @ w
        beta = float(np.cov(port_daily, spy)[0, 1] / np.var(spy, ddof=1))
        contrib = w * (cov.to_numpy() @ wv) / port_var
        total_value = float(self.port.iloc[-1])

        corr = rets.corr()
        pairs = []
        syms = self.symbols
        for i, a in enumerate(syms):
            for b in syms[i + 1:]:
                c = corr.loc[a, b]
                if np.isfinite(c) and c > 0.7:
                    pairs.append({"pair": [a, b], "correlation": round(float(c), 2)})
        pairs.sort(key=lambda p: -p["correlation"])

        clusters = self._clusters(corr, w, total_value, threshold=0.6)

        sector_w = w.groupby(pd.Series(self.sector_of)).sum().sort_values(ascending=False)

        # Max drawdown of the current portfolio (today's share counts) over the past year.
        v = (self.closes[self.symbols] * self.shares[self.symbols]).sum(axis=1).iloc[-(TRADING_DAYS + 1):]
        peak = v.cummax()
        dd = v / peak - 1
        trough_date = dd.idxmin()
        peak_date = v.loc[:trough_date].idxmax()
        drawdown = {
            "pct": pct(float(dd.min())),
            "dollars": usd(float(v.loc[trough_date] - v.loc[peak_date])),
            "peak_date": peak_date.date().isoformat(),
            "trough_date": trough_date.date().isoformat(),
            "recovered": bool(v.iloc[-1] >= v.loc[peak_date]),
        }

        # Stress: SPY -10%, each holding moves by its own beta.
        spy_var = float(np.var(spy, ddof=1))
        stress_rows = []
        for sym in syms:
            b = float(np.cov(rets[sym], spy)[0, 1] / spy_var) if spy_var else 1.0
            value = float(self.values[sym].iloc[-1])
            stress_rows.append((sym, b, value * b * -0.10))
        stress_total = sum(r[2] for r in stress_rows)
        stress_rows.sort(key=lambda r: r[2])

        positions = sorted(
            ({"symbol": s, "weight_pct": pct(w[s]), "risk_share_pct": pct(contrib[s])} for s in syms),
            key=lambda p: -p["risk_share_pct"],
        )
        top = w.sort_values(ascending=False)
        return {
            "as_of": self.as_of,
            "total_value": usd(total_value),
            "volatility_annual_pct": pct(port_vol),
            "typical_daily_move_dollars": usd(total_value * port_vol / np.sqrt(TRADING_DAYS)),
            "market_volatility_annual_pct": pct(float(spy.std() * np.sqrt(TRADING_DAYS))),
            "beta_to_spy": round(beta, 2),
            "effective_number_of_positions": round(float(1 / (wv ** 2).sum()), 1),
            "number_of_positions": len(syms),
            "largest_position": {"symbol": top.index[0], "weight_pct": pct(top.iloc[0])},
            "top3_weight_pct": pct(float(top.head(3).sum())),
            "risk_contributions": positions,
            "sector_weights_pct": {k: pct(v) for k, v in sector_w.items()},
            "correlated_pairs_above_0_7": pairs[:10],
            "correlated_groups": clusters,
            "max_drawdown_1y": drawdown,
            "stress_test_spy_down_10pct": {
                "dollars": usd(stress_total),
                "pct": pct(stress_total / total_value),
                "by_holding": [{"symbol": s, "beta": round(b, 2), "dollars": usd(d)} for s, b, d in stress_rows],
            },
            "method": "One year of daily returns at current weights. Risk share = w_i*(Σw)_i / wᵀΣw.",
        }

    def _clusters(self, corr: pd.DataFrame, w: pd.Series, total_value: float,
                  threshold: float) -> list[dict[str, Any]]:
        """Groups of holdings linked by pairwise correlation >= threshold."""
        syms = list(corr.index)
        parent = {s: s for s in syms}

        def find(s: str) -> str:
            while parent[s] != s:
                parent[s] = parent[parent[s]]
                s = parent[s]
            return s

        for i, a in enumerate(syms):
            for b in syms[i + 1:]:
                if corr.loc[a, b] >= threshold:
                    parent[find(a)] = find(b)
        groups: dict[str, list[str]] = {}
        for s in syms:
            groups.setdefault(find(s), []).append(s)
        out = []
        for members in groups.values():
            if len(members) < 2:
                continue
            sub = corr.loc[members, members].to_numpy()
            avg = sub[np.triu_indices(len(members), 1)].mean()
            weight = float(w[members].sum())
            out.append({
                "symbols": sorted(members, key=lambda s: -w[s]),
                "sectors": sorted({self.sector_of[s] for s in members}),
                "weight_pct": pct(weight),
                "value_dollars": usd(weight * total_value),
                "average_correlation": round(float(avg), 2),
            })
        out.sort(key=lambda g: -g["weight_pct"])
        return out

    # ---------------------------------------------------------------- thesis

    def news(self, symbols: list[str], days: int = 7, limit: int = 8) -> dict[str, Any]:
        days = int(max(1, min(int(days), 30)))
        out = {}
        for sym in symbols:
            sym = str(sym).strip().upper()
            if not sym:
                continue
            try:
                items = self.provider.news(sym, days)
            except Exception:
                items = []
            out[sym] = [{
                "title": n["title"],
                "publisher": n.get("publisher"),
                "published": n["published"][:16].replace("T", " ") + " UTC",
                "summary": (n.get("summary") or "")[:280],
                "url": n.get("url"),
            } for n in items[:limit]]
        return {"days": days, "news": out,
                "note": "Headlines only cover what the provider returns; absence of news is not proof nothing happened."}

    def thesis(self, symbol: str) -> dict[str, Any]:
        self.load()
        sym = str(symbol).strip().upper()
        entry = self.theses.get(sym)
        if sym not in self.symbols and entry is None:
            raise AnalyticsError(f"{sym} is not in the portfolio and has no thesis in theses.yaml.")
        out: dict[str, Any] = {"symbol": sym, "name": self.name(sym) if sym in self.symbols else sym}
        if entry:
            out.update({"thesis": entry["thesis"], "breaks_if": entry["breaks_if"], "watch_list": entry["watch"]})
        else:
            out.update({"thesis": None, "breaks_if": [], "watch_list": [],
                        "note": "No thesis recorded in theses.yaml for this holding."})

        if sym in self.symbols:
            perf = {}
            for p in ("1w", "1m"):
                df, w = self.attribution_frame(p)
                if sym not in df.index:  # bought too recently to have a move
                    continue
                row = df.loc[sym]
                perf[p] = {
                    "dollars": usd(row["total"]),
                    "pct": pct(row["total"] / row["start_value"]),
                    "market_dollars": usd(row["market"]),
                    "sector_dollars": usd(row["sector_move"]),
                    "stock_specific_dollars": usd(row["stock_specific"]),
                    "stock_specific_pct": pct(row["stock_specific"] / row["start_value"]),
                    "spy_pct": pct(float(self.closes[MARKET].iloc[w.end_pos] / self.closes[MARKET].iloc[w.start_pos] - 1)),
                }
            out["performance"] = perf
            out["position_value"] = usd(float(self.values[sym].iloc[-1]))
            out["weight_pct"] = pct(float(self._weights()[sym]))

        news = self.news([sym] + out["watch_list"], days=14, limit=6)["news"]
        out["news_14d"] = {"own": news.get(sym, []),
                           "watch_list": {s: news.get(s, []) for s in out["watch_list"]}}
        out["next_earnings"] = self._earnings_entry(sym)
        return out

    # ---------------------------------------------------------------- events

    def _earnings_entry(self, sym: str) -> dict[str, Any] | None:
        try:
            e = self.provider.earnings(sym)
        except Exception:
            e = None
        if not e or not e.get("date"):
            return None
        as_of = date.fromisoformat(self.as_of)
        when = date.fromisoformat(e["date"])
        entry = {"date": e["date"], "days_until": (when - as_of).days,
                 "implied_move_pct": e.get("implied_move_pct")}
        if sym in self.symbols and e.get("implied_move_pct") is not None:
            value = float(self.values[sym].iloc[-1])
            entry["implied_move_dollars"] = usd(value * e["implied_move_pct"] / 100)
        return entry

    def events(self, days: int = 30) -> dict[str, Any]:
        self.load()
        days = int(max(1, min(int(days), 120)))
        upcoming, unknown = [], []
        for sym in self.symbols:
            e = self._earnings_entry(sym)
            if e is None:
                unknown.append(sym)
                continue
            if 0 <= e["days_until"] <= days:
                upcoming.append({"symbol": sym, "name": self.name(sym), "event": "earnings",
                                 "position_value": usd(float(self.values[sym].iloc[-1])), **e})
        upcoming.sort(key=lambda e: e["date"])
        return {
            "as_of": self.as_of,
            "window_days": days,
            "events": upcoming,
            "no_date_available": unknown,
            "note": "Implied move = options market's expected +/- move through earnings (either direction).",
        }

    # --------------------------------------------------------------- journal

    def _price_series(self, sym: str) -> pd.Series | None:
        """Closes for any symbol, aligned to the analysis calendar."""
        if sym in self.closes:
            return self.closes[sym]
        cache = self.__dict__.setdefault("_extra_series", {})
        if sym not in cache:
            try:
                df = self.provider.prices([sym], self.closes.index[0].date())
                s = df[sym].reindex(self.closes.index).ffill() if sym in df else None
                cache[sym] = s if s is not None and not s.dropna().empty else None
            except Exception:
                cache[sym] = None
        return cache[sym]

    def trades_report(self, symbol: str | None = None) -> dict[str, Any]:
        self.load()
        sym = symbol.strip().upper() if symbol else None
        trades = [t for t in self.trades if sym is None or t.symbol == sym]
        return {
            "trades": [{"id": t.id, "date": t.date.isoformat(), "symbol": t.symbol, "side": t.side,
                        "shares": t.shares, "price": t.price,
                        "value_dollars": usd(t.shares * t.price) if t.price else None} for t in trades[-50:]],
            "count": len(trades),
            "realised_pl_dollars": {s: usd(v) for s, v in self.realised.items() if sym is None or s == sym},
            "note": "Opening positions from portfolio.csv aren't trades; they're assumed held throughout.",
        }

    def review_journal(self, entries: list[dict[str, Any]]) -> dict[str, Any]:
        """How each journaled decision has worked out since: the stock's move,
        the market's move, and the market-adjusted move (stock - beta x SPY).
        For trades, the $ effect on the shares traded."""
        self.load()
        idx = self.closes.index
        spy = self.closes[MARKET]
        spy_r = self.returns[MARKET]
        reviewed, buys, sells = [], [], []
        for e in entries:
            item = {k: e.get(k) for k in ("id", "date", "symbol", "kind")}
            item["text"] = (e.get("text") or "")[:300]
            if e.get("trade"):
                item["trade"] = e["trade"]
            sym = e.get("symbol")
            series = self._price_series(sym) if sym else None
            when = pd.Timestamp(e["date"])
            pos = int(idx.searchsorted(when))
            if series is None or when < idx[0] or pos >= len(idx) - 1:
                item["outcome"] = None
                item["outcome_note"] = ("No symbol." if not sym else "Too recent or outside the price "
                                        "history to evaluate." if series is not None else f"No price data for {sym}.")
                reviewed.append(item)
                continue
            p0, p1 = float(series.iloc[pos]), float(series.iloc[-1])
            stock_ret = p1 / p0 - 1
            spy_ret = float(spy.iloc[-1] / spy.iloc[pos] - 1)
            r = series.pct_change().iloc[max(1, pos - TRADING_DAYS + 1):pos + 1]
            m = spy_r.iloc[max(1, pos - TRADING_DAYS + 1):pos + 1]
            beta = float(np.cov(r, m)[0, 1] / np.var(m, ddof=1)) if len(r) >= MIN_FIT_OBS else 1.0
            adjusted = stock_ret - beta * spy_ret
            outcome = {
                "from": idx[pos].date().isoformat(),
                "sessions": len(idx) - 1 - pos,
                "stock_pct": pct(stock_ret),
                "spy_pct": pct(spy_ret),
                "beta": round(beta, 2),
                "market_adjusted_pct": pct(adjusted),
            }
            trade = e.get("trade") or {}
            if trade.get("price") and trade.get("shares"):
                move = trade["shares"] * (p1 - trade["price"])
                if trade.get("side") == "buy":
                    outcome["gain_since_buy_dollars"] = usd(move)
                else:
                    outcome["move_since_sale_dollars"] = usd(move)
            kind = e.get("kind")
            if kind == "buy_reason":
                buys.append(adjusted)
            elif kind == "sell_reason":
                sells.append(adjusted)
            item["outcome"] = outcome
            reviewed.append(item)
        summary = {"entries": len(reviewed)}
        if buys:
            summary["buys_avg_market_adjusted_pct"] = pct(float(np.mean(buys)))
            summary["buys_beating_market_adjusted"] = f"{sum(b > 0 for b in buys)} of {len(buys)}"
        if sells:
            summary["sells_avg_market_adjusted_pct_since"] = pct(float(np.mean(sells)))
            summary["sells_followed_by_underperformance"] = f"{sum(x < 0 for x in sells)} of {len(sells)}"
        return {
            "as_of": self.as_of,
            "summary": summary,
            "entries": reviewed,
            "how_to_read": ("market_adjusted_pct = stock move minus beta x S&P 500 move since the entry. "
                            "For a buy, positive is good. For a sell, negative means the stock lagged after "
                            "you sold (good timing); move_since_sale_dollars > 0 is gain you gave up."),
        }

    # ---------------------------------------------------------------- charts

    def chart(self, kind: str, symbol: str | None = None, period: str | None = None) -> dict[str, Any]:
        kind = (kind or "").strip().lower()
        if kind == "portfolio_vs_market":
            return self.chart_portfolio_vs_market(period or "1m")
        if kind == "stock_vs_market":
            if not symbol:
                raise AnalyticsError("stock_vs_market needs a symbol.")
            return self.chart_stock_vs_market(symbol, period or "1m")
        if kind == "attribution":
            return self.chart_attribution(period or "1w")
        raise AnalyticsError("Unknown chart kind. Use portfolio_vs_market, stock_vs_market or attribution.")

    def _labels(self, w: _Window) -> list[str]:
        return [d.date().isoformat() for d in self.closes.index[w.start_pos:w.end_pos + 1]]

    def chart_portfolio_vs_market(self, period: str = "1m") -> dict[str, Any]:
        w = self.window(period)
        start_value = float(self.port.iloc[w.start_pos])
        # Growth of the starting value (time-weighted), so buying and selling
        # don't show up as gains or losses.
        growth = (1 + self.port_ret.iloc[self._days(w)]).cumprod()
        port = pd.concat([pd.Series([start_value]), start_value * growth])
        spy = self.closes[MARKET].iloc[w.start_pos:w.end_pos + 1]
        scaled = spy / spy.iloc[0] * start_value
        return {
            "kind": "portfolio_vs_market",
            "type": "line",
            "title": f"Your portfolio vs the S&P 500, {w.label}",
            "unit": "$",
            "labels": self._labels(w),
            "series": [
                {"name": "Your portfolio", "role": "primary", "data": [usd(x) for x in port]},
                {"name": "S&P 500 (SPY), same start value", "role": "benchmark", "data": [usd(x) for x in scaled]},
            ],
        }

    def chart_stock_vs_market(self, symbol: str, period: str = "1m") -> dict[str, Any]:
        w = self.window(period)
        sym = symbol.strip().upper()
        if sym in self.closes:
            series = self.closes[sym].iloc[w.start_pos:w.end_pos + 1]
        else:
            fetched = self.provider.prices([sym], w.start.date())
            series = fetched[sym].reindex(self.closes.index[w.start_pos:w.end_pos + 1]).ffill() if sym in fetched else None
            if series is None or series.dropna().empty:
                raise AnalyticsError(f"No price data for {sym}.")
        spy = self.closes[MARKET].iloc[w.start_pos:w.end_pos + 1]

        def rebase(s: pd.Series) -> list[float | None]:
            base = s.dropna().iloc[0]
            return [round(float((x / base - 1) * 100), 2) if np.isfinite(x) else None for x in s]

        return {
            "kind": "stock_vs_market",
            "type": "line",
            "title": f"{sym} vs the S&P 500, {w.label}",
            "unit": "%",
            "labels": self._labels(w),
            "series": [
                {"name": sym, "role": "primary", "data": rebase(series)},
                {"name": "S&P 500 (SPY)", "role": "benchmark", "data": rebase(spy)},
            ],
        }

    def chart_attribution(self, period: str = "1w") -> dict[str, Any]:
        df, w = self.attribution_frame(period)
        df = df.reindex(df["total"].abs().sort_values(ascending=False).index)
        return {
            "kind": "attribution",
            "type": "bar",
            "stacked": True,
            "title": f"What drove each holding, {w.label}",
            "unit": "$",
            "labels": df.index.tolist(),
            "series": [
                {"name": "Market", "role": "market", "data": [usd(x) for x in df["market"]]},
                {"name": "Sector", "role": "sector", "data": [usd(x) for x in df["sector_move"]]},
                {"name": "Stock-specific", "role": "specific", "data": [usd(x) for x in df["stock_specific"]]},
            ],
        }


def build(mock: bool = True, portfolio_path=None, theses_path=None, provider: DataProvider | None = None,
          store=None) -> Analytics:
    """Build Analytics from a Store (opening positions, trades, theses) or,
    without one, straight from portfolio.csv and theses.yaml."""
    from .data import make_provider
    from .portfolio import DEFAULT_PORTFOLIO, DEFAULT_THESES, load_holdings, load_theses

    provider = provider or make_provider(mock)
    if store is not None:
        return Analytics(provider, store.opening(), store.theses(), store.trades())
    return Analytics(provider, load_holdings(portfolio_path or DEFAULT_PORTFOLIO),
                     load_theses(theses_path or DEFAULT_THESES))


class AnalyticsHolder:
    """Keeps one loaded Analytics and rebuilds it when prices go stale (live
    mode) or after a write to the store, so a long-running server (e.g. on a
    Raspberry Pi) always answers from current data. The provider and its disk
    cache are reused across rebuilds."""

    def __init__(self, mock: bool = False, ttl_seconds: float = 15 * 60,
                 analytics: Analytics | None = None, store=None):
        import time

        from .data import make_provider

        self._time = time.monotonic
        self.mock = mock
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        self._fixed = analytics is not None
        if store is None and not self._fixed:
            from .store import open_store

            store = open_store(mock)
        self.store = store
        self._provider = analytics.provider if analytics else None
        self._make_provider = lambda: make_provider(mock)
        self._current = analytics
        self._built_at = self._time()
        self._dirty = False

    def invalidate(self) -> None:
        """Call after writing to the store."""
        with self._lock:
            self._dirty = True

    def get(self) -> Analytics:
        with self._lock:
            expired = not self.mock and self._time() - self._built_at > self.ttl
            stale = not self._fixed and self._current is not None and (self._dirty or expired)
            if self._current is None or stale:
                self._provider = self._provider or self._make_provider()
                fresh = build(provider=self._provider, store=self.store)
                try:
                    fresh.load()
                except Exception:
                    if self._current is None or self._dirty:
                        raise
                    return self._current  # keep serving the last good data
                self._current, self._built_at, self._dirty = fresh, self._time(), False
            return self._current


if __name__ == "__main__":  # python -m app.analytics [--mock] [period]
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Print a sample attribution and risk report.")
    ap.add_argument("period", nargs="?", default="1w")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    a = AnalyticsHolder(mock=args.mock).get()
    att = a.attribution(args.period)
    print(att["headline"], "\n")
    for h in att["holdings"]:
        print(f"  {h['symbol']:<6} total {h['total']:>8,}  market {h['market']:>7,}  "
              f"sector {h['sector_move']:>7,}  stock {h['stock_specific']:>7,}")
    print("\nRisk report:")
    print(json.dumps(a.risk(), indent=2))
