from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

MARKET = "SPY"

# Yahoo sector name -> SPDR sector ETF
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Energy": "XLE",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
}

# Yahoo isn't always consistent about sector names.
SECTOR_ALIASES = {
    "Information Technology": "Technology",
    "Health Care": "Healthcare",
    "Financials": "Financial Services",
    "Financial": "Financial Services",
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Staples": "Consumer Defensive",
    "Materials": "Basic Materials",
    "Communication": "Communication Services",
}


# Yahoo's keys for fund sector weightings
FUND_SECTOR_KEYS = {
    "technology": "Technology",
    "healthcare": "Healthcare",
    "financial_services": "Financial Services",
    "energy": "Energy",
    "consumer_cyclical": "Consumer Cyclical",
    "consumer_defensive": "Consumer Defensive",
    "industrials": "Industrials",
    "utilities": "Utilities",
    "realestate": "Real Estate",
    "basic_materials": "Basic Materials",
    "communication_services": "Communication Services",
}


def normalise_sector(sector):
    if not sector or not str(sector).strip():
        return None
    sector = str(sector).strip()
    return SECTOR_ALIASES.get(sector, sector)


def sector_etf(sector):
    return SECTOR_ETFS.get(normalise_sector(sector) or "")


class DataProvider:
    """Interface for market data. Missing data should come back as None, an
    empty list or NaN columns, never an exception."""

    name = "base"
    is_mock = False

    def as_of(self) -> date:
        raise NotImplementedError

    def prices(self, symbols: list[str], start: date) -> pd.DataFrame:
        """Adjusted daily closes, one column per symbol (all-NaN if unknown)."""
        raise NotImplementedError

    def profile(self, symbol: str) -> dict[str, Any]:
        """symbol, name, sector, industry, quote_type."""
        raise NotImplementedError

    def news(self, symbol: str, days: int = 14) -> list[dict[str, Any]]:
        """Newest first: symbol, title, publisher, published (ISO UTC), url, summary."""
        raise NotImplementedError

    def earnings(self, symbol: str) -> dict[str, Any] | None:
        """symbol, date, implied_move_pct, implied_move_source."""
        raise NotImplementedError

    def fund(self, symbol: str) -> dict[str, Any] | None:
        """For ETFs and mutual funds: category, sector_weights ({sector: fraction}),
        top_holdings ([{symbol, name, weight}]) and asset_classes ({stocks, bonds,
        cash, other}). None for anything that isn't a fund."""
        return None
