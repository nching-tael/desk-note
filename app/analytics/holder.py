"""Building an Analytics object and keeping it fresh in a long-running server."""

from __future__ import annotations

import threading
import time

from ..data import make_provider
from ..portfolio import DEFAULT_PORTFOLIO, DEFAULT_THESES, load_holdings, load_theses
from .core import Analytics


def build(mock=True, portfolio_path=None, theses_path=None, provider=None, store=None):
    """From a Store (positions, trades, theses) if given, otherwise straight
    from portfolio.csv and theses.yaml."""
    provider = provider or make_provider(mock)
    if store is not None:
        return Analytics(provider, store.opening(), store.theses(), store.trades())
    return Analytics(
        provider,
        load_holdings(portfolio_path or DEFAULT_PORTFOLIO),
        load_theses(theses_path or DEFAULT_THESES),
    )


class AnalyticsHolder:
    """Holds the current Analytics and rebuilds it when prices are older than
    ttl (live mode) or after something is written to the store."""

    def __init__(self, mock=False, ttl_seconds=15 * 60, analytics=None, store=None):
        self.mock = mock
        self.ttl = ttl_seconds
        self.fixed = analytics is not None  # tests pass one in and it never changes
        if store is None and not self.fixed:
            from ..store import open_store

            store = open_store(mock)
        self.store = store
        self.provider = analytics.provider if analytics else None
        self.current = analytics
        self.built_at = time.monotonic()
        self.dirty = False
        self.lock = threading.Lock()

    def invalidate(self):
        with self.lock:
            self.dirty = True

    def get(self):
        with self.lock:
            if self.current is None or self._stale():
                self._rebuild()
            return self.current

    def _stale(self):
        if self.fixed:
            return False
        expired = not self.mock and time.monotonic() - self.built_at > self.ttl
        return self.dirty or expired

    def _rebuild(self):
        self.provider = self.provider or make_provider(self.mock)
        fresh = build(provider=self.provider, store=self.store)
        try:
            fresh.load()
        except Exception:
            # if a refresh fails (e.g. Yahoo is down), keep serving the old data
            if self.current is None or self.dirty:
                raise
            return
        self.current = fresh
        self.built_at = time.monotonic()
        self.dirty = False
