"""The Analytics class: loads prices for a portfolio and computes gains,
performance and attribution. Risk, research, journal and charts live in
their own modules and are exposed as methods here."""

from __future__ import annotations

import threading
from datetime import timedelta

import numpy as np
import pandas as pd

from ..data import MARKET, sector_etf
from ..portfolio import replay
from . import charts, journal, research, risk
from .periods import (
    PERIOD_SESSIONS,
    TRADING_DAYS,
    Window,
    fmt_usd,
    normalise_period,
    pct,
    usd,
)

MIN_FIT_DAYS = 40  # below this, assume beta 1 to the market


class Analytics:
    def __init__(self, provider, holdings, theses=None, trades=None):
        """holdings are the opening positions, assumed held throughout the
        price history; trades are buys and sells after that."""
        self.provider = provider
        self.holdings = holdings
        self.theses = theses or {}
        self.trades = sorted(trades or [], key=lambda t: (t.date, t.id or 0))
        self.warnings = []
        self.extra_prices = {}
        self._loaded = False
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if not self._loaded:
                self._load()
                self._loaded = True
        return self

    def _load(self):
        today = self.provider.as_of()
        symbols = list(dict.fromkeys([h.symbol for h in self.holdings] + [t.symbol for t in self.trades]))

        self.profiles = {s: self.provider.profile(s) for s in symbols}
        self.sector_of = {}
        self.etf_of = {}
        for symbol, profile in self.profiles.items():
            sector = profile.get("sector")
            if not sector and profile.get("quote_type") == "ETF":
                sector = "ETF"
            self.sector_of[symbol] = sector or "Unknown"
            self.etf_of[symbol] = sector_etf(sector)
        etfs = sorted({e for e in self.etf_of.values() if e})

        start = today - timedelta(days=2 * 365 + 45)
        closes = self.provider.prices(sorted(set(symbols) | set(etfs) | {MARKET}), start)
        closes = closes[closes.index <= pd.Timestamp(today)]
        if MARKET not in closes or closes[MARKET].isna().all():
            raise RuntimeError("No market (SPY) price data available.")
        closes = closes[closes[MARKET].notna()].sort_index().ffill(limit=5)

        priced = []
        self.listed_pos = {}  # index of each symbol's first real price
        for symbol in symbols:
            series = closes.get(symbol)
            if series is None or series.isna().all():
                self.warnings.append(f"No price data for {symbol}; it is excluded from the analysis.")
                continue
            first = series.first_valid_index()
            if first > closes.index[0] + pd.Timedelta(days=400):
                self.warnings.append(f"{symbol} has less than a year of price history.")
            # Before a stock listed, hold its first price flat so it has no returns.
            closes[symbol] = series.bfill()
            self.listed_pos[symbol] = closes.index.get_loc(first)
            priced.append(symbol)
        if not priced:
            raise RuntimeError("None of the holdings have price data.")

        for etf in etfs:
            if etf not in closes or closes[etf].isna().all():
                self.warnings.append(f"No data for sector ETF {etf}; using a market-only model instead.")
                self.etf_of = {s: (None if e == etf else e) for s, e in self.etf_of.items()}
            else:
                closes[etf] = closes[etf].bfill()

        # Shares held at each close: the opening positions throughout, then
        # each trade from the close of its date onwards.
        shares = pd.DataFrame(0.0, index=closes.index, columns=priced)
        for h in self.holdings:
            if h.symbol in shares:
                shares[h.symbol] += h.shares
        for t in self.trades:
            if t.symbol in shares:
                change = t.shares if t.side == "buy" else -t.shares
                shares.loc[shares.index >= pd.Timestamp(t.date), t.symbol] += change
        shares = shares.clip(lower=0).round(9)  # 10 - 3.3 - 6.7 should be exactly 0

        book = replay(self.holdings, [t for t in self.trades if t.symbol in priced])
        self.cost = {s: book[s].avg_cost if s in book else None for s in priced}
        self.realised = {s: book[s].realised for s in priced if s in book and book[s].realised}

        self.closes = closes
        self.returns = closes.pct_change().fillna(0.0)
        self.all_symbols = priced
        self.shares_df = shares
        self.shares = shares.iloc[-1]
        self.symbols = [s for s in priced if self.shares[s] > 0]  # held today
        if not self.symbols:
            raise RuntimeError("No current holdings with price data.")

        self.values = closes[priced] * shares
        self.port = self.values.sum(axis=1)
        # A day's gain comes from the shares held at the previous close, so
        # buying more shares never counts as a gain.
        self.pnl_df = (shares.shift(1) * closes[priced].diff()).fillna(0.0)
        self.pnl = self.pnl_df.sum(axis=1)
        prev_value = self.port.shift(1)
        self.port_ret = (self.pnl / prev_value).where(prev_value > 0, 0.0).fillna(0.0)

    # --- basics

    @property
    def as_of(self):
        self.load()
        return self.closes.index[-1].date().isoformat()

    def name(self, symbol):
        return self.profiles.get(symbol, {}).get("name") or symbol

    def window(self, period):
        self.load()
        period = normalise_period(period)
        index = self.closes.index
        last = len(index) - 1
        if period == "ytd":
            n = int((index.year == index[-1].year).sum())
        else:
            n = PERIOD_SESSIONS[period]
        n = max(1, min(n, last))
        return Window(period, index[last - n], index[last], last - n, last)

    def weights(self):
        values = self.values.iloc[-1][self.symbols]
        return values / values.sum()

    def held_during(self, w):
        """Symbols held at any of the closes the period's returns start from."""
        held = self.shares_df.iloc[w.start_pos : w.end_pos] > 0
        return [s for s in self.all_symbols if held[s].any()]

    def gain(self, w):
        return float(self.pnl.iloc[w.days].sum())

    def time_weighted_return(self, w):
        return float((1 + self.port_ret.iloc[w.days]).prod() - 1)

    def market_return(self, w):
        spy = self.closes[MARKET]
        return float(spy.iloc[w.end_pos] / spy.iloc[w.start_pos] - 1)

    def price_series(self, symbol):
        """Closes for any symbol, held or not, on the same calendar."""
        if symbol in self.closes:
            return self.closes[symbol]
        if symbol not in self.extra_prices:
            series = None
            try:
                prices = self.provider.prices([symbol], self.closes.index[0].date())
                if symbol in prices and not prices[symbol].isna().all():
                    series = prices[symbol].reindex(self.closes.index).ffill()
            except Exception:
                pass
            self.extra_prices[symbol] = series
        return self.extra_prices[symbol]

    # --- overview and performance

    def overview(self):
        self.load()
        total = float(self.port.iloc[-1])
        positions = []
        cost_total = gain_total = 0.0

        for symbol in self.symbols:
            shares = float(self.shares[symbol])
            price = float(self.closes[symbol].iloc[-1])
            value = shares * price
            position = {
                "symbol": symbol,
                "name": self.name(symbol),
                "sector": self.sector_of[symbol],
                "shares": shares,
                "price": round(price, 2),
                "value": usd(value),
                "weight_pct": pct(value / total),
                "day_dollars": usd(self.pnl_df[symbol].iloc[-1]),
                "day_pct": pct(price / self.closes[symbol].iloc[-2] - 1),
                "unrealised_pl": None,
                "unrealised_pl_pct": None,
            }
            cost = self.cost.get(symbol)
            if cost:
                basis = cost * shares
                position["unrealised_pl"] = usd(value - basis)
                position["unrealised_pl_pct"] = pct(value / basis - 1)
                cost_total += basis
                gain_total += value - basis
            positions.append(position)

        def change(period):
            w = self.window(period)
            return {"dollars": usd(self.gain(w)), "pct": pct(self.time_weighted_return(w))}

        result = {
            "as_of": self.as_of,
            "data_source": self.provider.name,
            "total_value": usd(total),
            "day_change": change("1d"),
            "week_change": change("1w"),
            "unrealised_pl": {
                "dollars": usd(gain_total),
                "pct": pct(gain_total / cost_total) if cost_total else None,
                "cost_basis_total": usd(cost_total),
            },
            "positions": sorted(positions, key=lambda p: p["value"], reverse=True),
            "warnings": list(self.warnings),
        }
        if self.realised:
            result["realised_pl"] = {
                "dollars": usd(sum(self.realised.values())),
                "by_symbol": {s: usd(v) for s, v in self.realised.items()},
            }
        return result

    def performance(self, period="1w"):
        w = self.window(period)
        positions = []
        for symbol in self.held_during(w):
            held = self.shares_df[symbol].shift(1).iloc[w.days] > 0
            returns = self.returns[symbol].iloc[w.days].where(held, 0.0)
            positions.append(
                {
                    "symbol": symbol,
                    "name": self.name(symbol),
                    "dollars": usd(self.pnl_df[symbol].iloc[w.days].sum()),
                    "pct": pct((1 + returns).prod() - 1),
                }
            )
        positions.sort(key=lambda p: p["dollars"], reverse=True)

        start_value = float(self.port.iloc[w.start_pos])
        end_value = float(self.port.iloc[w.end_pos])
        gain = self.gain(w)
        portfolio_return = self.time_weighted_return(w)
        market_return = self.market_return(w)

        result = {
            "period": w.period,
            "period_label": w.label,
            "start_date": w.start.date().isoformat(),
            "end_date": w.end.date().isoformat(),
            "sessions": w.sessions,
            "start_value": usd(start_value),
            "end_value": usd(end_value),
            "change_dollars": usd(gain),
            "change_pct": pct(portfolio_return),
            "market_spy_pct": pct(market_return),
            "vs_market_pct_points": round((portfolio_return - market_return) * 100, 1),
            "if_tracked_market_dollars": usd(start_value * market_return),
            "best": positions[0],
            "worst": positions[-1],
            "positions": positions,
        }
        purchases = end_value - start_value - gain
        if abs(purchases) >= 1:
            result["net_purchases_dollars"] = usd(purchases)
            result["note"] = (
                "change_dollars is gain or loss only; the value also changed by "
                "net_purchases_dollars from buying and selling. change_pct is time-weighted."
            )
        return result

    # --- attribution

    def fit_betas(self, symbol, w):
        """Fit r = a + b_mkt * SPY + b_sec * (sector ETF - SPY) on the year of
        returns before the period."""
        lo = max(1, w.start_pos - TRADING_DAYS + 1, self.listed_pos.get(symbol, 0) + 1)
        days = slice(lo, w.start_pos + 1)
        y = self.returns[symbol].iloc[days].to_numpy()
        market = self.returns[MARKET].iloc[days].to_numpy()
        etf = self.etf_of.get(symbol)

        columns = [np.ones_like(market), market]
        if etf:
            columns.append(self.returns[etf].iloc[days].to_numpy() - market)
        X = np.column_stack(columns)

        if len(y) < MIN_FIT_DAYS:
            return {"beta_mkt": 1.0, "beta_sector": 0.0, "etf": etf, "fallback": True}
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        return {
            "beta_mkt": float(coef[1]),
            "beta_sector": float(coef[2]) if etf else 0.0,
            "etf": etf,
            "fallback": False,
        }

    def attribution_frame(self, period="1w"):
        """Unrounded $ split per holding into market, sector_move and
        stock_specific (which add up to total)."""
        w = self.window(period)
        market = self.returns[MARKET].iloc[w.days].to_numpy()
        rows = []

        for symbol in self.held_during(w):
            fit = self.fit_betas(symbol, w)
            returns = self.returns[symbol].iloc[w.days].to_numpy()
            if fit["etf"]:
                sector = self.returns[fit["etf"]].iloc[w.days].to_numpy() - market
            else:
                sector = np.zeros_like(market)

            market_part = fit["beta_mkt"] * market
            sector_part = fit["beta_sector"] * sector
            stock_part = returns - market_part - sector_part
            prev_value = self.values[symbol].iloc[w.start_pos : w.end_pos].to_numpy()
            # for percentages: the starting value, or the average if bought mid-period
            base = prev_value[0] if prev_value[0] > 0 else prev_value[prev_value > 0].mean()

            rows.append(
                {
                    "symbol": symbol,
                    "sector": self.sector_of[symbol],
                    "market": float(market_part @ prev_value),
                    "sector_move": float(sector_part @ prev_value),
                    "stock_specific": float(stock_part @ prev_value),
                    "total": float(returns @ prev_value),
                    "start_value": float(base),
                    "beta_mkt": fit["beta_mkt"],
                    "beta_sector": fit["beta_sector"],
                    "sector_etf": fit["etf"],
                    "fallback": fit["fallback"],
                }
            )
        return pd.DataFrame(rows).set_index("symbol"), w

    def attribution(self, period="1w"):
        df, w = self.attribution_frame(period)
        total, market = df["total"].sum(), df["market"].sum()
        sector, stock = df["sector_move"].sum(), df["stock_specific"].sum()

        by_sector = df.groupby("sector")["sector_move"].sum()
        by_sector = by_sector.reindex(by_sector.abs().sort_values(ascending=False).index)
        top = [f"{name} {fmt_usd(v)}" for name, v in by_sector.head(2).items() if abs(v) >= 1]
        sector_detail = f" ({'; '.join(top)})" if top else ""
        headline = (
            f"Of your {fmt_usd(total)} over the {w.label}, {fmt_usd(market)} was the market, "
            f"{fmt_usd(sector)} was sector moves{sector_detail}, and {fmt_usd(stock)} was "
            f"your stock picks (stock-specific moves)."
        )

        holdings = [
            {
                "symbol": symbol,
                "sector": row["sector"],
                "total": usd(row["total"]),
                "market": usd(row["market"]),
                "sector_move": usd(row["sector_move"]),
                "stock_specific": usd(row["stock_specific"]),
                "stock_specific_pct": pct(row["stock_specific"] / row["start_value"]),
                "beta_mkt": round(row["beta_mkt"], 2),
                "beta_sector": round(row["beta_sector"], 2),
                "sector_etf": row["sector_etf"] if isinstance(row["sector_etf"], str) else None,
            }
            for symbol, row in df.sort_values("total").iterrows()
        ]

        biggest = df.loc[df["stock_specific"].abs().sort_values(ascending=False).index[:3]]
        result = {
            "period": w.period,
            "period_label": w.label,
            "start_date": w.start.date().isoformat(),
            "end_date": w.end.date().isoformat(),
            "sessions": w.sessions,
            "headline": headline,
            "total": usd(total),
            "market": usd(market),
            "sector": usd(sector),
            "stock_specific": usd(stock),
            "sector_by_name": {name: usd(v) for name, v in by_sector.items()},
            "biggest_stock_specific": [
                {
                    "symbol": s,
                    "dollars": usd(r["stock_specific"]),
                    "pct": pct(r["stock_specific"] / r["start_value"]),
                }
                for s, r in biggest.iterrows()
            ],
            "holdings": holdings,
            "method": (
                "Per holding: return = a + b_mkt*SPY + b_sec*(sector ETF - SPY), fit on the year "
                "before the period. Each day's parts x the previous day's position value, summed. "
                "Stock-specific is what's left over, including the intercept."
            ),
        }
        fallbacks = df.index[df["fallback"]].tolist()
        if fallbacks:
            result["note"] = (
                f"Not enough history to fit betas for {', '.join(fallbacks)}; assumed beta 1 to the market."
            )
        return result

    # --- everything else lives in its own module

    def risk(self):
        return risk.report(self)

    def news(self, symbols, days=7, limit=8):
        return research.news(self, symbols, days, limit)

    def thesis(self, symbol):
        return research.thesis(self, symbol)

    def events(self, days=30):
        return research.events(self, days)

    def trades_report(self, symbol=None):
        return journal.trades_report(self, symbol)

    def review_journal(self, entries):
        return journal.review(self, entries)

    def chart(self, kind, symbol=None, period=None):
        return charts.chart(self, kind, symbol, period)

    def chart_portfolio_vs_market(self, period="1m"):
        return charts.portfolio_vs_market(self, period)
