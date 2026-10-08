"""Live market data from Yahoo Finance, via yfinance, with a small disk cache."""

from __future__ import annotations

import hashlib
import logging
import pickle
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd

from .base import DataProvider, normalise_sector

log = logging.getLogger(__name__)

ISO = "%Y-%m-%dT%H:%M:%SZ"


class DiskCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key):
        return self.root / (hashlib.sha1(key.encode()).hexdigest() + ".pkl")

    def get(self, key, ttl):
        path = self._path(key)
        try:
            if time.time() - path.stat().st_mtime > ttl:
                return None
            with path.open("rb") as f:
                return pickle.load(f)
        except (OSError, pickle.PickleError, EOFError):
            return None

    def set(self, key, value):
        try:
            with self._path(key).open("wb") as f:
                pickle.dump(value, f)
        except OSError:
            pass


def to_iso(value):
    """Epoch seconds or a date string -> ISO UTC string (None if unparseable)."""
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(value, tz=UTC)
        else:
            dt = pd.Timestamp(value).to_pydatetime()
            dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
        return dt.strftime(ISO)
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def normalise_news_item(symbol, item):
    """yfinance has two news formats: the old flat one and a newer one nested
    under "content". Returns our common shape, or None if it's unusable."""
    if not isinstance(item, dict):
        return None

    content = item.get("content")
    if isinstance(content, dict):
        title = content.get("title")
        publisher = (content.get("provider") or {}).get("displayName")
        url = (content.get("canonicalUrl") or content.get("clickThroughUrl") or {}).get("url")
        published = to_iso(content.get("pubDate") or content.get("displayTime"))
        summary = content.get("summary") or content.get("description") or ""
    else:
        title = item.get("title")
        publisher = item.get("publisher")
        url = item.get("link") or item.get("url")
        published = to_iso(item.get("providerPublishTime") or item.get("published"))
        summary = item.get("summary") or ""

    if not title or not published:
        return None
    return {
        "symbol": symbol,
        "title": title.strip(),
        "publisher": publisher or "Unknown",
        "published": published,
        "url": url,
        "summary": summary.strip()[:400],
    }


def company_name(info):
    short, full = info.get("shortName"), info.get("longName")
    # Yahoo cuts shortName off at 30 characters ("Taiwan Semiconductor Manufactur")
    if short and len(short) < 30:
        return short
    return full or short


class LiveProvider(DataProvider):
    name = "yahoo"

    PRICE_TTL = 30 * 60
    PROFILE_TTL = 7 * 24 * 3600
    NEWS_TTL = 30 * 60
    EARNINGS_TTL = 6 * 3600

    def __init__(self, cache_dir=".cache"):
        import yfinance  # only needed in live mode

        self.yf = yfinance
        # yfinance logs every unknown symbol at ERROR level; we handle those ourselves
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        self.cache = DiskCache(cache_dir)
        self.tickers = {}

    def ticker(self, symbol):
        if symbol not in self.tickers:
            self.tickers[symbol] = self.yf.Ticker(symbol)
        return self.tickers[symbol]

    def as_of(self):
        return date.today()

    def prices(self, symbols, start):
        symbols = sorted({s.upper() for s in symbols})
        key = f"prices:{','.join(symbols)}:{start}"
        cached = self.cache.get(key, self.PRICE_TTL)
        if cached is not None:
            return cached

        try:
            raw = self.yf.download(
                symbols, start=start.isoformat(), auto_adjust=True, progress=False, group_by="column"
            )
        except Exception as e:
            log.warning("price download failed: %s", e)
            raw = pd.DataFrame()

        closes = extract_closes(raw, symbols)
        if not closes.dropna(how="all").empty:
            self.cache.set(key, closes)
        return closes

    def profile(self, symbol):
        symbol = symbol.upper()
        cached = self.cache.get(f"profile:{symbol}", self.PROFILE_TTL)
        if cached is not None:
            return cached

        try:
            info = self.ticker(symbol).get_info() or {}
        except Exception as e:
            log.warning("profile lookup failed for %s: %s", symbol, e)
            info = {}

        profile = {
            "symbol": symbol,
            "name": company_name(info) or symbol,
            "sector": normalise_sector(info.get("sector")),
            "industry": info.get("industry"),
            "quote_type": info.get("quoteType"),
        }
        if info:
            self.cache.set(f"profile:{symbol}", profile)
        return profile

    def news(self, symbol, days=14):
        symbol = symbol.upper()
        items = self.cache.get(f"news:{symbol}", self.NEWS_TTL)
        if items is None:
            items = self.fetch_news(symbol)
            if items:
                self.cache.set(f"news:{symbol}", items)

        cutoff = (datetime.now(UTC) - timedelta(days=days)).strftime(ISO)
        seen = set()
        result = []
        for item in sorted(items, key=lambda n: n["published"], reverse=True):
            # the same story is often syndicated under several publishers
            if item["published"] >= cutoff and item["title"].lower() not in seen:
                seen.add(item["title"].lower())
                result.append(item)
        return result

    def fetch_news(self, symbol):
        # The ticker news endpoint is often empty, so fall back to search, then RSS.
        for source in (self.news_from_ticker, self.news_from_search, self.news_from_rss):
            try:
                items = [normalise_news_item(symbol, raw) for raw in source(symbol) or []]
                items = [i for i in items if i]
            except Exception as e:
                log.info("%s failed for %s: %s", source.__name__, symbol, e)
                continue
            if items:
                return items
        return []

    def news_from_ticker(self, symbol):
        return self.ticker(symbol).get_news(count=30)

    def news_from_search(self, symbol):
        results = self.yf.Search(symbol, news_count=20, max_results=1).news or []
        # search also returns general market stories; keep the ones tagged with this symbol
        tagged = [r for r in results if symbol in r.get("relatedTickers", [])]
        return tagged or [r for r in results if "relatedTickers" not in r]

    def news_from_rss(self, symbol):
        query = urllib.parse.urlencode({"s": symbol, "region": "US", "lang": "en-US"})
        req = urllib.request.Request(
            "https://feeds.finance.yahoo.com/rss/2.0/headline?" + query,
            headers={"User-Agent": "Mozilla/5.0 (DeskNote)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            root = ET.fromstring(resp.read())

        items = []
        for node in root.iter("item"):
            try:
                published = parsedate_to_datetime(node.findtext("pubDate")).timestamp()
            except (TypeError, ValueError):
                published = None
            items.append(
                {
                    "title": node.findtext("title"),
                    "link": node.findtext("link"),
                    "summary": node.findtext("description") or "",
                    "providerPublishTime": published,
                    "publisher": "Yahoo Finance RSS",
                }
            )
        return items

    def earnings(self, symbol):
        symbol = symbol.upper()
        cached = self.cache.get(f"earnings:{symbol}", self.EARNINGS_TTL)
        if cached is not None:
            return cached or None  # {} means we looked and there's no date

        result = None
        ticker = self.ticker(symbol)
        when = next_earnings_date(ticker)
        if when:
            try:
                move = implied_move(ticker, when)
            except Exception as e:
                log.info("no implied move for %s: %s", symbol, e)
                move = None
            result = {
                "symbol": symbol,
                "date": when.isoformat(),
                "implied_move_pct": move,
                "implied_move_source": "ATM straddle / spot" if move is not None else None,
            }
        self.cache.set(f"earnings:{symbol}", result or {})
        return result


def next_earnings_date(ticker):
    today = date.today()
    dates = []
    try:
        calendar = ticker.calendar
        if isinstance(calendar, dict):
            raw = calendar.get("Earnings Date") or []
            dates += [pd.Timestamp(d).date() for d in (raw if isinstance(raw, list) else [raw])]
    except Exception:
        pass

    if not any(d >= today for d in dates):
        try:
            history = ticker.get_earnings_dates(limit=8)
            if history is not None:
                dates += [ts.date() for ts in history.index]
        except Exception:
            pass

    upcoming = sorted(d for d in dates if d >= today)
    return upcoming[0] if upcoming else None


def implied_move(ticker, earnings_date):
    """Expected % move through earnings: at-the-money call + put (mid prices)
    divided by the stock price, using the first expiry in the 10 days after."""
    expiries = [pd.Timestamp(e).date() for e in ticker.options or ()]
    expiries = [e for e in expiries if earnings_date <= e <= earnings_date + timedelta(days=10)]
    if not expiries:
        return None

    closes = ticker.history(period="5d", auto_adjust=True)["Close"].dropna()
    if closes.empty:
        return None
    spot = float(closes.iloc[-1])  # today's row can be NaN mid-session
    chain = ticker.option_chain(expiries[0].isoformat())

    def atm_price(options):
        if options is None or options.empty:
            return None
        row = options.loc[(options["strike"] - spot).abs().idxmin()]
        if row["bid"] > 0 and row["ask"] > 0:
            return (row["bid"] + row["ask"]) / 2
        return row["lastPrice"] if row["lastPrice"] > 0 else None

    call, put = atm_price(chain.calls), atm_price(chain.puts)
    if call is None or put is None:
        return None
    return round(float(call + put) / spot * 100, 1)


def extract_closes(raw, symbols):
    """yf.download returns differently shaped frames depending on how many
    symbols you ask for. Normalise to one close column per symbol."""
    empty = pd.DataFrame(columns=symbols, dtype=float)
    if raw is None or raw.empty:
        return empty

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            closes = raw["Close"]
        else:
            closes = raw.xs("Close", axis=1, level=1)
    elif "Close" in raw.columns:
        closes = raw[["Close"]].rename(columns={"Close": symbols[0]})
    else:
        return empty

    closes = closes.copy()
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).normalize()
    return closes.reindex(columns=symbols).astype(float).sort_index()
