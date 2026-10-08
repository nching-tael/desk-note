"""Synthetic market data for demos and tests, so nothing depends on the network.

Prices come from a simple factor model with a fixed seed. The last five
sessions are scripted: the market falls about 2%, Nvidia drops after
Microsoft signals slower data center spending, Eli Lilly jumps on trial data,
oil rises, AMD slides on the last day with no news, and TSMC reports in four
days with a ~6% implied move.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .base import DataProvider, sector_etf


@dataclass(frozen=True)
class MockStock:
    name: str
    sector: str
    price: float  # last close
    beta_mkt: float
    beta_sec: float
    idio_vol: float  # daily stock-specific volatility
    semis: float  # exposure to a shared chip-industry factor
    drift: float


STOCKS = {
    "NVDA": MockStock("NVIDIA Corporation", "Technology", 182, 1.7, 1.0, 0.011, 1.0, 0.0012),
    "TSM": MockStock("Taiwan Semiconductor Manufacturing", "Technology", 291, 1.3, 1.0, 0.010, 1.0, 0.0010),
    "AVGO": MockStock("Broadcom Inc.", "Technology", 334, 1.4, 1.0, 0.011, 1.0, 0.0011),
    "AMD": MockStock("Advanced Micro Devices", "Technology", 161, 1.6, 1.0, 0.013, 1.0, 0.0007),
    "MSFT": MockStock("Microsoft Corporation", "Technology", 508, 1.0, 0.8, 0.010, 0, 0.0006),
    "AAPL": MockStock("Apple Inc.", "Technology", 252, 1.1, 0.8, 0.011, 0, 0.0005),
    "AMZN": MockStock("Amazon.com, Inc.", "Consumer Cyclical", 221, 1.2, 0.9, 0.014, 0, 0.0006),
    "LLY": MockStock("Eli Lilly and Company", "Healthcare", 806, 0.5, 1.0, 0.015, 0, 0.0007),
    "JPM": MockStock("JPMorgan Chase & Co.", "Financial Services", 302, 1.0, 1.0, 0.009, 0, 0.0007),
    "XOM": MockStock("Exxon Mobil Corporation", "Energy", 116, 0.7, 1.0, 0.010, 0, 0.0002),
    "COST": MockStock("Costco Wholesale Corporation", "Consumer Defensive", 921, 0.7, 0.9, 0.010, 0, 0.0006),
    # watch-list names
    "META": MockStock("Meta Platforms, Inc.", "Communication Services", 715, 1.3, 1.0, 0.016, 0, 0.0008),
    "GOOGL": MockStock("Alphabet Inc.", "Communication Services", 243, 1.1, 1.0, 0.013, 0, 0.0006),
    "ASML": MockStock("ASML Holding N.V.", "Technology", 980, 1.3, 1.0, 0.013, 0.6, 0.0004),
    "INTC": MockStock("Intel Corporation", "Technology", 36, 1.2, 1.0, 0.020, 0.6, 0.0),
    "MRVL": MockStock("Marvell Technology", "Technology", 84, 1.6, 1.0, 0.020, 0.8, 0.0004),
    "QCOM": MockStock("QUALCOMM Incorporated", "Technology", 168, 1.2, 1.0, 0.013, 0.5, 0.0003),
    "ORCL": MockStock("Oracle Corporation", "Technology", 290, 1.1, 1.0, 0.016, 0, 0.0007),
    "NVO": MockStock("Novo Nordisk A/S", "Healthcare", 58, 0.6, 1.0, 0.017, 0, -0.0004),
    "AMGN": MockStock("Amgen Inc.", "Healthcare", 295, 0.6, 1.0, 0.011, 0, 0.0002),
    "BAC": MockStock("Bank of America", "Financial Services", 51, 1.1, 1.0, 0.010, 0, 0.0005),
    "GS": MockStock("The Goldman Sachs Group", "Financial Services", 780, 1.2, 1.0, 0.011, 0, 0.0008),
    "C": MockStock("Citigroup Inc.", "Financial Services", 101, 1.2, 1.0, 0.012, 0, 0.0007),
    "CVX": MockStock("Chevron Corporation", "Energy", 158, 0.7, 1.0, 0.010, 0, 0.0001),
    "COP": MockStock("ConocoPhillips", "Energy", 96, 0.8, 1.0, 0.012, 0, 0.0001),
    "WMT": MockStock("Walmart Inc.", "Consumer Defensive", 102, 0.6, 0.9, 0.009, 0, 0.0006),
    "TGT": MockStock("Target Corporation", "Consumer Defensive", 92, 0.9, 0.9, 0.016, 0, -0.0003),
}

# ETF: (last close, beta to SPY, daily volatility relative to SPY)
ETFS = {
    "SPY": (668, 1.0, 0.0),
    "XLK": (281, 1.15, 0.005),
    "XLV": (141, 0.7, 0.005),
    "XLF": (53, 1.0, 0.005),
    "XLE": (90, 0.8, 0.011),
    "XLY": (238, 1.1, 0.005),
    "XLP": (79, 0.6, 0.005),
    "XLI": (152, 1.0, 0.004),
    "XLU": (85, 0.5, 0.006),
    "XLRE": (42, 0.8, 0.006),
    "XLB": (90, 0.9, 0.006),
    "XLC": (115, 1.0, 0.005),
}

# The scripted last week, oldest day first.
WEEK_SPY = [-0.004, -0.006, 0.002, -0.009, -0.003]
WEEK_SECTOR = {  # sector ETF minus SPY
    "XLK": [-0.002, -0.003, -0.006, -0.002, 0.001],
    "XLE": [0.008, 0.012, 0.004, 0.009, 0.003],
    "XLV": [0.000, 0.006, 0.001, 0.001, 0.000],
}
WEEK_SEMIS = [-0.002, 0.000, -0.010, -0.004, -0.001]
WEEK_STOCK = {  # stock-specific moves
    "NVDA": [0.002, 0.001, -0.062, -0.012, 0.003],
    "LLY": [0.003, 0.088, 0.012, -0.004, 0.002],
    "XOM": [0.002, 0.006, 0.001, 0.004, 0.000],
    "CVX": [0.001, 0.005, 0.000, 0.003, 0.000],
    "NVO": [0.000, -0.071, -0.008, 0.002, 0.000],
    "AMD": [0.001, 0.002, -0.004, 0.000, -0.034],  # no news explains the last day
    "TSM": [0.002, 0.003, 0.004, 0.006, 0.002],
    "MSFT": [0.000, 0.001, -0.018, 0.004, 0.001],
    "META": [0.001, 0.000, -0.006, 0.014, 0.000],
    "AAPL": [0.002, -0.001, 0.003, 0.001, 0.002],
}

NEWS = yaml.safe_load((Path(__file__).parent / "mock_news.yaml").read_text())

# symbol: (calendar days until earnings, implied move %)
EARNINGS = {
    "TSM": (4, 6.1),
    "JPM": (8, 3.0),
    "MSFT": (21, 4.5),
    "GOOGL": (21, 5.9),
    "META": (22, 7.2),
    "AMZN": (23, 6.8),
    "AAPL": (23, 4.2),
    "XOM": (24, 2.9),
    "AMD": (27, 8.4),
    "LLY": (29, 5.5),
    "NVDA": (43, 7.9),
    "COST": (64, 3.4),
    "AVGO": (64, 6.5),
}

PUBLISHER = "Demo Wire (synthetic)"

# Funds: a few ETFs with sector weights and top holdings, for look-through.
# Their prices are built from their holdings (see generate), XLE's already exists.
FUNDS = {
    "VOO": {
        "name": "Vanguard S&P 500 ETF",
        "price": 615,
        "category": "Large Blend",
        "sectors": {
            "Technology": 0.38,
            "Financial Services": 0.12,
            "Communication Services": 0.10,
            "Consumer Cyclical": 0.10,
            "Healthcare": 0.09,
            "Industrials": 0.08,
            "Consumer Defensive": 0.05,
            "Energy": 0.03,
            "Utilities": 0.02,
            "Real Estate": 0.02,
            "Basic Materials": 0.01,
        },
        "holdings": {
            "NVDA": 0.081,
            "AAPL": 0.070,
            "MSFT": 0.057,
            "AMZN": 0.038,
            "GOOGL": 0.030,
            "META": 0.029,
            "AVGO": 0.026,
            "JPM": 0.014,
            "LLY": 0.012,
            "XOM": 0.009,
        },
    },
    "QQQ": {
        "name": "Invesco QQQ Trust",
        "price": 602,
        "category": "Large Growth",
        "sectors": {
            "Technology": 0.60,
            "Communication Services": 0.13,
            "Consumer Cyclical": 0.10,
            "Consumer Defensive": 0.06,
            "Healthcare": 0.05,
            "Industrials": 0.04,
            "Utilities": 0.02,
        },
        "holdings": {
            "NVDA": 0.083,
            "AAPL": 0.074,
            "MSFT": 0.058,
            "AMZN": 0.052,
            "AVGO": 0.050,
            "META": 0.045,
            "GOOGL": 0.044,
            "AMD": 0.042,
            "COST": 0.025,
            "QCOM": 0.015,
        },
    },
    "SMH": {
        "name": "VanEck Semiconductor ETF",
        "price": 331,
        "category": "Technology",
        "sectors": {"Technology": 1.0},
        "holdings": {
            "NVDA": 0.193,
            "TSM": 0.093,
            "AMD": 0.055,
            "AVGO": 0.053,
            "ASML": 0.045,
            "QCOM": 0.040,
            "INTC": 0.035,
            "MRVL": 0.030,
        },
    },
    "XLE": {
        "name": "Energy Select Sector SPDR Fund",
        "price": 90,
        "category": "Equity Energy",
        "sectors": {"Energy": 1.0},
        "holdings": {"XOM": 0.24, "CVX": 0.18, "COP": 0.067},
    },
    "BND": {
        "name": "Vanguard Total Bond Market ETF",
        "price": 74,
        "category": "Intermediate Core Bond",
        "sectors": {},
        "holdings": {},
        "bonds": True,
    },
}


def prices_from_returns(returns, last_price):
    """Price path with the given daily returns that ends at last_price.
    The first return is ignored since there's no earlier close."""
    growth = np.cumprod(1 + np.asarray(returns, dtype=float))
    return last_price * growth / growth[-1]


def last_weekday(d):
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


class MockProvider(DataProvider):
    name = "mock"
    is_mock = True
    SESSIONS = 560  # a bit over two years

    def __init__(self, as_of=None, seed=35):
        self.today = last_weekday(as_of or date.today())
        self.seed = seed
        self.sessions = pd.bdate_range(end=self.today, periods=self.SESSIONS)
        self.closes = self.generate()

    def as_of(self):
        return self.today

    def generate(self):
        n = self.SESSIONS
        rng = np.random.default_rng(self.seed)
        week = slice(n - 5, n)

        spy = rng.normal(0.0004, 0.0095, n)
        spy[week] = WEEK_SPY
        semis = rng.normal(0, 0.012, n)
        semis[week] = WEEK_SEMIS

        returns = {"SPY": spy}
        sector_excess = {}
        for etf, (_, beta, vol) in ETFS.items():
            if etf == "SPY":
                continue
            excess = rng.normal(0, vol, n)
            excess[week] = WEEK_SECTOR.get(etf, excess[week] * 0.3)
            returns[etf] = beta * spy + excess
            returns[etf][week] = spy[week] + excess[week]
            sector_excess[etf] = returns[etf] - spy

        for symbol, s in STOCKS.items():
            own = rng.normal(0, s.idio_vol, n)
            own[week] = WEEK_STOCK.get(symbol, own[week] * 0.3)
            sector = sector_excess.get(sector_etf(s.sector), np.zeros(n))
            r = s.drift + s.beta_mkt * spy + s.beta_sec * sector + s.semis * semis + own
            r[week] -= s.drift  # keep the scripted week exactly as written
            returns[symbol] = r

        # Funds get their own random stream so adding one never changes the
        # prices above. Holdings move the fund; the rest follows a rough proxy.
        fund_rng = np.random.default_rng(self.seed + 1)
        rest_of_fund = {
            "VOO": spy,
            "QQQ": 1.1 * spy + 0.6 * sector_excess["XLK"],
            "SMH": 1.4 * spy + sector_excess["XLK"] + semis,
        }
        for symbol, fund in FUNDS.items():
            if symbol in returns:
                continue
            if fund.get("bonds"):
                returns[symbol] = fund_rng.normal(0.00015, 0.003, n)
                continue
            held = sum(fund["holdings"].values())
            r = sum(w * returns[s] for s, w in fund["holdings"].items()) + (1 - held) * rest_of_fund[symbol]
            returns[symbol] = r + fund_rng.normal(0, 0.0005, n)

        last_prices = {sym: s.price for sym, s in STOCKS.items()}
        last_prices.update({sym: fund["price"] for sym, fund in FUNDS.items()})
        last_prices.update({etf: params[0] for etf, params in ETFS.items()})
        closes = {sym: prices_from_returns(r, last_prices[sym]) for sym, r in returns.items()}
        return pd.DataFrame(closes, index=self.sessions)

    def made_up_prices(self, symbol):
        """Plausible prices for a symbol the mock doesn't know (e.g. after
        someone edits portfolio.csv). Seeded by the symbol so it's stable."""
        rng = np.random.default_rng(int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16))
        spy = self.closes["SPY"].pct_change().fillna(0).to_numpy()
        r = 0.0003 + rng.uniform(0.6, 1.4) * spy + rng.normal(0, 0.014, len(spy))
        return prices_from_returns(r, rng.uniform(20, 400))

    def prices(self, symbols, start):
        columns = {}
        for symbol in (s.upper() for s in symbols):
            if symbol in self.closes:
                columns[symbol] = self.closes[symbol].to_numpy()
            else:
                columns[symbol] = self.made_up_prices(symbol)
        df = pd.DataFrame(columns, index=self.sessions)
        return df[df.index >= pd.Timestamp(start)]

    def profile(self, symbol):
        symbol = symbol.upper()
        if symbol in STOCKS:
            stock = STOCKS[symbol]
            return {
                "symbol": symbol,
                "name": stock.name,
                "sector": stock.sector,
                "industry": None,
                "quote_type": "EQUITY",
            }
        if symbol in FUNDS:
            return {
                "symbol": symbol,
                "name": FUNDS[symbol]["name"],
                "sector": None,
                "industry": None,
                "quote_type": "ETF",
            }
        quote_type = "ETF" if symbol in ETFS else None
        return {"symbol": symbol, "name": symbol, "sector": None, "industry": None, "quote_type": quote_type}

    def fund(self, symbol):
        fund = FUNDS.get(symbol.upper())
        if fund is None:
            return None
        holdings = [
            {"symbol": s, "name": STOCKS[s].name if s in STOCKS else s, "weight": w}
            for s, w in fund["holdings"].items()
        ]
        if fund.get("bonds"):
            classes = {"stocks": 0.0, "bonds": 0.986, "cash": 0.014, "other": 0.0}
        else:
            classes = {"stocks": 0.995, "bonds": 0.0, "cash": 0.005, "other": 0.0}
        return {
            "category": fund["category"],
            "sector_weights": dict(fund["sectors"]),
            "top_holdings": holdings,
            "asset_classes": classes,
        }

    def news(self, symbol, days=14):
        symbol = symbol.upper()
        midnight = datetime.min.time()
        cutoff = datetime.combine(self.today, midnight, tzinfo=UTC) - timedelta(days=days)

        items = []
        for story in NEWS:
            if story["symbol"] != symbol:
                continue
            day = self.sessions[-1 - story["days_ago"]].date()
            published = datetime.combine(day, midnight, tzinfo=UTC) + timedelta(hours=story["hour"])
            if published >= cutoff:
                items.append(
                    {
                        "symbol": symbol,
                        "title": story["title"],
                        "publisher": PUBLISHER,
                        "published": published.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "url": None,
                        "summary": story["summary"],
                    }
                )
        return sorted(items, key=lambda item: item["published"], reverse=True)

    def earnings(self, symbol):
        symbol = symbol.upper()
        if symbol not in EARNINGS:
            return None
        days, move = EARNINGS[symbol]
        return {
            "symbol": symbol,
            "date": (self.today + timedelta(days=days)).isoformat(),
            "implied_move_pct": move,
            "implied_move_source": "synthetic",
        }
