"""Periods and number formatting shared by the analytics modules.

Results are rounded for display: dollars to whole numbers, percentages to one
decimal place (2.1 means 2.1%).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

TRADING_DAYS = 252

PERIODS = ["1d", "1w", "1m", "3m", "ytd", "1y"]
PERIOD_SESSIONS = {"1d": 1, "1w": 5, "1m": 21, "3m": 63, "1y": 252}
PERIOD_LABELS = {
    "1d": "last trading day",
    "1w": "last 5 trading days",
    "1m": "last month",
    "3m": "last 3 months",
    "ytd": "year to date",
    "1y": "last year",
}
# fmt: off
PERIOD_ALIASES = {
    "day": "1d", "today": "1d", "1day": "1d", "d": "1d",
    "week": "1w", "1week": "1w", "w": "1w", "5d": "1w",
    "month": "1m", "1month": "1m", "m": "1m",
    "3month": "3m", "3months": "3m", "quarter": "3m",
    "year": "1y", "1year": "1y", "12m": "1y", "y": "1y",
    "year_to_date": "ytd",
}
# fmt: on


class AnalyticsError(ValueError):
    """Bad input, e.g. an unknown period or symbol. The message is shown to the model."""


def normalise_period(period, default="1w"):
    p = (period or default).strip().lower().replace(" ", "").replace("-", "")
    p = PERIOD_ALIASES.get(p, p)
    if p not in PERIODS:
        raise AnalyticsError(f"Unknown period '{period}'. Use one of: {', '.join(PERIODS)}.")
    return p


@dataclass
class Window:
    """A period measured from the close at start_pos to the close at end_pos."""

    period: str
    start: pd.Timestamp
    end: pd.Timestamp
    start_pos: int
    end_pos: int

    @property
    def sessions(self):
        return self.end_pos - self.start_pos

    @property
    def days(self):
        """Index positions of the sessions whose returns make up the period."""
        return slice(self.start_pos + 1, self.end_pos + 1)

    @property
    def label(self):
        return PERIOD_LABELS[self.period]


def usd(x):
    x = float(x)
    return round(x) if math.isfinite(x) else 0


def pct(fraction):
    if fraction is None or not math.isfinite(fraction):
        return None
    return round(float(fraction) * 100, 1) + 0.0  # + 0.0 turns -0.0 into 0.0


def fmt_usd(x):
    """-1240.4 -> '-$1,240', 12 -> '+$12'."""
    x = usd(x)
    if x < 0:
        return f"-${-x:,}"
    return f"+${x:,}" if x > 0 else "$0"
