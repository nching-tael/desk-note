"""Shared test helpers: a provider that serves fixed prices."""

from datetime import date

import numpy as np
import pandas as pd

from app.data import DataProvider


class DummyProvider(DataProvider):
    """Serves fixed closes. ``sectors`` maps symbol -> Yahoo sector name."""

    name = "dummy"
    is_mock = True

    def __init__(self, closes: pd.DataFrame, sectors: dict[str, str] | None = None):
        self.closes = closes
        self.sectors = sectors or {}

    def as_of(self) -> date:
        return self.closes.index[-1].date()

    def prices(self, symbols, start):
        return self.closes.reindex(columns=[s.upper() for s in symbols])

    def profile(self, symbol):
        return {"symbol": symbol, "name": symbol, "sector": self.sectors.get(symbol), "quote_type": "EQUITY"}

    def news(self, symbol, days=14):
        return []

    def earnings(self, symbol):
        return None


def frame(columns: dict[str, list[float]], end: str = "2026-10-07") -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=pd.bdate_range(end=end, periods=n))


def prices_from_returns(returns: np.ndarray, end_price: float = 100.0) -> list[float]:
    """Price path whose day-on-day returns are exactly ``returns`` (first is ignored)."""
    growth = np.cumprod(1 + np.asarray(returns, dtype=float))
    growth = growth / growth[0]
    return list(end_price * growth / growth[-1])
