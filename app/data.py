"""Market data providers.

Two implementations share one interface:

* ``LiveProvider`` pulls prices, company profiles, news, earnings dates and
  options-implied earnings moves from Yahoo Finance (via yfinance), with a small
  on-disk cache so repeated questions don't re-download everything.
* ``MockProvider`` generates deterministic synthetic data with a scripted week of
  stories, so the app, tests and screenshots work with no network at all.

Every method degrades gracefully: missing data comes back as ``None``, an empty
list or NaN columns rather than an exception.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MARKET = "SPY"

# Yahoo Finance sector names -> SPDR sector ETFs.
SECTOR_ETFS: dict[str, str] = {
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

# Yahoo occasionally uses alternative spellings.
_SECTOR_ALIASES = {
    "Information Technology": "Technology",
    "Health Care": "Healthcare",
    "Financials": "Financial Services",
    "Financial": "Financial Services",
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Staples": "Consumer Defensive",
    "Materials": "Basic Materials",
    "Communication": "Communication Services",
}


def normalise_sector(sector: str | None) -> str | None:
    if not sector:
        return None
    sector = str(sector).strip()
    return _SECTOR_ALIASES.get(sector, sector) if sector else None


def sector_etf(sector: str | None) -> str | None:
    """The SPDR ETF for a Yahoo sector name, or None if unknown."""
    return SECTOR_ETFS.get(normalise_sector(sector) or "")


class DataProvider:
    """Interface shared by the live and mock providers."""

    name = "base"
    is_mock = False

    def as_of(self) -> date:
        """The date the provider considers 'today'."""
        raise NotImplementedError

    def prices(self, symbols: list[str], start: date) -> pd.DataFrame:
        """Daily closes (adjusted for splits and dividends) from ``start``.

        Index: tz-naive DatetimeIndex of sessions. Columns: one per requested
        symbol; symbols with no data are present as all-NaN columns.
        """
        raise NotImplementedError

    def profile(self, symbol: str) -> dict[str, Any]:
        """``{"symbol", "name", "sector", "industry"}``; unknown fields are None."""
        raise NotImplementedError

    def news(self, symbol: str, days: int = 14) -> list[dict[str, Any]]:
        """Recent headlines, newest first, each normalised to
        ``{"symbol", "title", "publisher", "published", "url", "summary"}``
        where ``published`` is an ISO-8601 UTC timestamp."""
        raise NotImplementedError

    def earnings(self, symbol: str) -> dict[str, Any] | None:
        """Next earnings date and options-implied move, or None if unknown.

        ``{"symbol", "date": "YYYY-MM-DD", "implied_move_pct": float | None,
        "implied_move_source": str | None}``
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Live provider (Yahoo Finance)
# ---------------------------------------------------------------------------


class _DiskCache:
    """Tiny pickle cache with per-entry TTL. Failures are ignored."""

    def __init__(self, root: Path):
        self.root = root
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def _path(self, key: str) -> Path:
        return self.root / (hashlib.sha1(key.encode()).hexdigest() + ".pkl")

    def get(self, key: str, ttl: float) -> Any:
        path = self._path(key)
        try:
            if time.time() - path.stat().st_mtime > ttl:
                return None
            with path.open("rb") as fh:
                return pickle.load(fh)
        except (OSError, pickle.PickleError, EOFError, AttributeError):
            return None

    def set(self, key: str, value: Any) -> None:
        try:
            with self._path(key).open("wb") as fh:
                pickle.dump(value, fh)
        except (OSError, pickle.PickleError):
            pass


def _iso_utc(value: Any) -> str | None:
    """Normalise yfinance timestamps (epoch seconds or ISO strings) to ISO UTC."""
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        else:
            dt = pd.Timestamp(value).to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def normalise_news_item(symbol: str, item: dict[str, Any]) -> dict[str, Any] | None:
    """Convert either yfinance news schema (pre-0.2.48 flat, or newer nested
    ``content``) into the provider's common shape. Returns None if unusable."""
    if not isinstance(item, dict):
        return None
    content = item.get("content") if isinstance(item.get("content"), dict) else None
    if content is not None:
        provider = content.get("provider") or {}
        url = (content.get("canonicalUrl") or {}).get("url") or (
            content.get("clickThroughUrl") or {}
        ).get("url")
        title = content.get("title")
        publisher = provider.get("displayName") if isinstance(provider, dict) else None
        published = _iso_utc(content.get("pubDate") or content.get("displayTime"))
        summary = content.get("summary") or content.get("description") or ""
    else:
        title = item.get("title")
        publisher = item.get("publisher")
        url = item.get("link") or item.get("url")
        published = _iso_utc(item.get("providerPublishTime") or item.get("published"))
        summary = item.get("summary") or ""
    if not title or not published:
        return None
    return {
        "symbol": symbol,
        "title": str(title).strip(),
        "publisher": publisher or "Unknown",
        "published": published,
        "url": url,
        "summary": str(summary).strip()[:400],
    }


class LiveProvider(DataProvider):
    name = "yahoo"
    is_mock = False

    PRICE_TTL = 30 * 60
    PROFILE_TTL = 7 * 24 * 3600
    NEWS_TTL = 30 * 60
    EARNINGS_TTL = 6 * 3600

    def __init__(self, cache_dir: str | Path = ".cache"):
        import yfinance  # imported lazily so mock mode never needs it

        self._yf = yfinance
        # yfinance logs every missing symbol/field at ERROR; we handle those ourselves.
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        self.cache = _DiskCache(Path(cache_dir))
        self._tickers: dict[str, Any] = {}

    def _ticker(self, symbol: str):
        if symbol not in self._tickers:
            self._tickers[symbol] = self._yf.Ticker(symbol)
        return self._tickers[symbol]

    def as_of(self) -> date:
        return date.today()

    def prices(self, symbols: list[str], start: date) -> pd.DataFrame:
        symbols = sorted({s.upper() for s in symbols})
        key = f"prices:{','.join(symbols)}:{start.isoformat()}"
        cached = self.cache.get(key, self.PRICE_TTL)
        if isinstance(cached, pd.DataFrame):
            return cached
        try:
            raw = self._yf.download(
                tickers=symbols,
                start=start.isoformat(),
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="column",
            )
        except Exception as exc:  # network errors, rate limits, parser changes
            log.warning("price download failed: %s", exc)
            raw = pd.DataFrame()
        closes = _extract_closes(raw, symbols)
        if not closes.dropna(how="all").empty:
            self.cache.set(key, closes)
        return closes

    def profile(self, symbol: str) -> dict[str, Any]:
        symbol = symbol.upper()
        key = f"profile:{symbol}"
        cached = self.cache.get(key, self.PROFILE_TTL)
        if isinstance(cached, dict):
            return cached
        info: dict[str, Any] = {}
        try:
            info = self._ticker(symbol).get_info() or {}
        except Exception as exc:
            log.warning("profile lookup failed for %s: %s", symbol, exc)
        result = {
            "symbol": symbol,
            "name": _company_name(info) or symbol,
            "sector": normalise_sector(info.get("sector")),
            "industry": info.get("industry"),
            "quote_type": info.get("quoteType"),
        }
        if info:
            self.cache.set(key, result)
        return result

    def news(self, symbol: str, days: int = 14) -> list[dict[str, Any]]:
        symbol = symbol.upper()
        key = f"news:{symbol}"
        items = self.cache.get(key, self.NEWS_TTL)
        if not isinstance(items, list):
            items = []
            # Yahoo's ticker news endpoint is often empty; fall back to search, then RSS.
            for source in (self._news_ticker, self._news_search, self._news_rss):
                try:
                    items = [n for n in (normalise_news_item(symbol, r) for r in source(symbol) or []) if n]
                except Exception as exc:
                    log.info("news source %s failed for %s: %s", source.__name__, symbol, exc)
                if items:
                    break
            if items:
                self.cache.set(key, items)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent = [n for n in items if _parse_iso(n["published"]) >= cutoff]
        # De-duplicate syndicated headlines.
        seen: set[str] = set()
        unique = []
        for n in sorted(recent, key=lambda n: n["published"], reverse=True):
            k = n["title"].lower()
            if k not in seen:
                seen.add(k)
                unique.append(n)
        return unique

    def _news_ticker(self, symbol: str) -> list[dict[str, Any]]:
        ticker = self._ticker(symbol)
        return ticker.get_news(count=30) if hasattr(ticker, "get_news") else ticker.news

    def _news_search(self, symbol: str) -> list[dict[str, Any]]:
        results = self._yf.Search(symbol, news_count=20, max_results=1).news or []
        # Search returns market-wide stories too; keep the ones tagged with this symbol.
        tagged = [r for r in results if symbol in (r.get("relatedTickers") or [])]
        return tagged if tagged else [r for r in results if "relatedTickers" not in r]

    @staticmethod
    def _news_rss(symbol: str) -> list[dict[str, Any]]:
        import urllib.parse
        import urllib.request
        import xml.etree.ElementTree as ET
        from email.utils import parsedate_to_datetime

        url = ("https://feeds.finance.yahoo.com/rss/2.0/headline?"
               + urllib.parse.urlencode({"s": symbol, "region": "US", "lang": "en-US"}))
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (DeskNote)"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            root = ET.fromstring(resp.read())
        items = []
        for it in root.iter("item"):
            pub = it.findtext("pubDate")
            try:
                ts = parsedate_to_datetime(pub).timestamp() if pub else None
            except (TypeError, ValueError):
                ts = None
            items.append({"title": it.findtext("title"), "link": it.findtext("link"),
                          "summary": it.findtext("description") or "", "providerPublishTime": ts,
                          "publisher": "Yahoo Finance RSS"})
        return items

    def earnings(self, symbol: str) -> dict[str, Any] | None:
        symbol = symbol.upper()
        key = f"earnings:{symbol}"
        cached = self.cache.get(key, self.EARNINGS_TTL)
        if cached is not None:
            return cached or None  # {} caches "no earnings date"
        result = self._fetch_earnings(symbol)
        self.cache.set(key, result or {})
        return result

    def _fetch_earnings(self, symbol: str) -> dict[str, Any] | None:
        ticker = self._ticker(symbol)
        next_date = self._next_earnings_date(ticker)
        if next_date is None:
            return None
        move, source = None, None
        try:
            move = self._implied_move(ticker, next_date)
            source = "ATM straddle / spot, first expiry after earnings" if move is not None else None
        except Exception as exc:
            log.info("implied move unavailable for %s: %s", symbol, exc)
        return {
            "symbol": symbol,
            "date": next_date.isoformat(),
            "implied_move_pct": move,
            "implied_move_source": source,
        }

    @staticmethod
    def _next_earnings_date(ticker) -> date | None:
        today = date.today()
        candidates: list[date] = []
        try:
            cal = ticker.calendar
            if isinstance(cal, dict):
                raw = cal.get("Earnings Date") or []
                raw = raw if isinstance(raw, (list, tuple)) else [raw]
                candidates += [pd.Timestamp(d).date() for d in raw if d is not None]
            elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                candidates += [pd.Timestamp(d).date() for d in cal.loc["Earnings Date"].dropna()]
        except Exception:
            pass
        if not any(d >= today for d in candidates):
            try:
                df = ticker.get_earnings_dates(limit=8)
                if df is not None and not df.empty:
                    candidates += [pd.Timestamp(d).date() for d in df.index]
            except Exception:
                pass
        upcoming = sorted(d for d in candidates if d >= today)
        return upcoming[0] if upcoming else None

    @staticmethod
    def _implied_move(ticker, earnings_date: date) -> float | None:
        """Options-implied earnings move: ATM straddle mid / spot, using the
        first expiry on or after the earnings date (within 10 days of it)."""
        expiries = [pd.Timestamp(e).date() for e in (ticker.options or ())]
        after = [e for e in expiries if earnings_date <= e <= earnings_date + timedelta(days=10)]
        if not after:
            return None
        expiry = after[0]
        chain = ticker.option_chain(expiry.isoformat())
        hist = ticker.history(period="5d", auto_adjust=True)
        if hist is None or hist.empty:
            return None
        spot = float(hist["Close"].iloc[-1])

        def mid(df: pd.DataFrame) -> float | None:
            if df is None or df.empty:
                return None
            row = df.iloc[(df["strike"] - spot).abs().argsort()[:1]].iloc[0]
            bid, ask, last = row.get("bid"), row.get("ask"), row.get("lastPrice")
            if bid and ask and bid > 0 and ask > 0:
                return (float(bid) + float(ask)) / 2
            return float(last) if last and last > 0 else None

        call, put = mid(chain.calls), mid(chain.puts)
        if call is None or put is None or spot <= 0:
            return None
        return round((call + put) / spot * 100, 1)


def _company_name(info: dict[str, Any]) -> str | None:
    short, long_ = info.get("shortName"), info.get("longName")
    # Yahoo truncates shortName at 30 characters ("Taiwan Semiconductor Manufactur").
    if short and len(short) < 30:
        return short
    return long_ or short


def _parse_iso(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _extract_closes(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Pull a symbols-by-column close frame out of whatever yf.download returned."""
    if raw is None or raw.empty:
        closes = pd.DataFrame(columns=symbols, dtype=float)
    elif isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "Close" in level0:
            closes = raw["Close"]
        else:  # grouped by ticker
            closes = raw.xs("Close", axis=1, level=1)
    else:
        closes = raw[["Close"]].rename(columns={"Close": symbols[0]}) if "Close" in raw else raw
    closes = closes.copy()
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).normalize()
    closes = closes.reindex(columns=symbols).astype(float)
    return closes.sort_index()


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MockStock:
    name: str
    sector: str | None
    price: float  # price on the as-of date
    beta_mkt: float
    beta_sec: float
    idio_vol: float
    semis: float = 0.0  # loading on a shared semiconductor factor
    mu: float = 0.0003


_MOCK_STOCKS: dict[str, _MockStock] = {
    "NVDA": _MockStock("NVIDIA Corporation", "Technology", 182.0, 1.7, 1.0, 0.011, 1.0, 0.0012),
    "TSM": _MockStock("Taiwan Semiconductor Manufacturing", "Technology", 291.0, 1.3, 1.0, 0.010, 1.0, 0.0010),
    "AVGO": _MockStock("Broadcom Inc.", "Technology", 334.0, 1.4, 1.0, 0.011, 1.0, 0.0011),
    "AMD": _MockStock("Advanced Micro Devices", "Technology", 161.0, 1.6, 1.0, 0.013, 1.0, 0.0007),
    "MSFT": _MockStock("Microsoft Corporation", "Technology", 508.0, 1.0, 0.8, 0.010, 0.0, 0.0006),
    "AAPL": _MockStock("Apple Inc.", "Technology", 252.0, 1.1, 0.8, 0.011, 0.0, 0.0005),
    "AMZN": _MockStock("Amazon.com, Inc.", "Consumer Cyclical", 221.0, 1.2, 0.9, 0.014, 0.0, 0.0006),
    "LLY": _MockStock("Eli Lilly and Company", "Healthcare", 806.0, 0.5, 1.0, 0.015, 0.0, 0.0007),
    "JPM": _MockStock("JPMorgan Chase & Co.", "Financial Services", 302.0, 1.0, 1.0, 0.009, 0.0, 0.0007),
    "XOM": _MockStock("Exxon Mobil Corporation", "Energy", 116.0, 0.7, 1.0, 0.010, 0.0, 0.0002),
    "COST": _MockStock("Costco Wholesale Corporation", "Consumer Defensive", 921.0, 0.7, 0.9, 0.010, 0.0, 0.0006),
    # Watch-list names (news and prices available, not held by default).
    "META": _MockStock("Meta Platforms, Inc.", "Communication Services", 715.0, 1.3, 1.0, 0.016, 0.0, 0.0008),
    "GOOGL": _MockStock("Alphabet Inc.", "Communication Services", 243.0, 1.1, 1.0, 0.013, 0.0, 0.0006),
    "ASML": _MockStock("ASML Holding N.V.", "Technology", 980.0, 1.3, 1.0, 0.013, 0.6, 0.0004),
    "INTC": _MockStock("Intel Corporation", "Technology", 36.0, 1.2, 1.0, 0.020, 0.6, 0.0),
    "MRVL": _MockStock("Marvell Technology", "Technology", 84.0, 1.6, 1.0, 0.020, 0.8, 0.0004),
    "QCOM": _MockStock("QUALCOMM Incorporated", "Technology", 168.0, 1.2, 1.0, 0.013, 0.5, 0.0003),
    "ORCL": _MockStock("Oracle Corporation", "Technology", 290.0, 1.1, 1.0, 0.016, 0.0, 0.0007),
    "NVO": _MockStock("Novo Nordisk A/S", "Healthcare", 58.0, 0.6, 1.0, 0.017, 0.0, -0.0004),
    "AMGN": _MockStock("Amgen Inc.", "Healthcare", 295.0, 0.6, 1.0, 0.011, 0.0, 0.0002),
    "BAC": _MockStock("Bank of America", "Financial Services", 51.0, 1.1, 1.0, 0.010, 0.0, 0.0005),
    "GS": _MockStock("The Goldman Sachs Group", "Financial Services", 780.0, 1.2, 1.0, 0.011, 0.0, 0.0008),
    "C": _MockStock("Citigroup Inc.", "Financial Services", 101.0, 1.2, 1.0, 0.012, 0.0, 0.0007),
    "CVX": _MockStock("Chevron Corporation", "Energy", 158.0, 0.7, 1.0, 0.010, 0.0, 0.0001),
    "COP": _MockStock("ConocoPhillips", "Energy", 96.0, 0.8, 1.0, 0.012, 0.0, 0.0001),
    "WMT": _MockStock("Walmart Inc.", "Consumer Defensive", 102.0, 0.6, 0.9, 0.009, 0.0, 0.0006),
    "TGT": _MockStock("Target Corporation", "Consumer Defensive", 92.0, 0.9, 0.9, 0.016, 0.0, -0.0003),
}

# ETF parameters: (price on as-of date, beta to SPY, excess-return vol).
_MOCK_ETFS: dict[str, tuple[float, float, float]] = {
    "SPY": (668.0, 1.0, 0.0),
    "XLK": (281.0, 1.15, 0.005),
    "XLV": (141.0, 0.7, 0.005),
    "XLF": (53.0, 1.0, 0.005),
    "XLE": (90.0, 0.8, 0.011),
    "XLY": (238.0, 1.1, 0.005),
    "XLP": (79.0, 0.6, 0.005),
    "XLI": (152.0, 1.0, 0.004),
    "XLU": (85.0, 0.5, 0.006),
    "XLRE": (42.0, 0.8, 0.006),
    "XLB": (90.0, 0.9, 0.006),
    "XLC": (115.0, 1.0, 0.005),
}

# The scripted final week (5 sessions, oldest first).
_WEEK_SPY = [-0.004, -0.006, 0.002, -0.009, -0.003]  # about -2.0%
_WEEK_SECTOR_EXCESS = {  # sector ETF return minus SPY return
    "XLK": [-0.002, -0.003, -0.006, -0.002, 0.001],
    "XLE": [0.008, 0.012, 0.004, 0.009, 0.003],
    "XLV": [0.000, 0.006, 0.001, 0.001, 0.000],
}
_WEEK_SEMIS = [-0.002, 0.000, -0.010, -0.004, -0.001]
_WEEK_IDIO = {  # stock-specific shocks
    "NVDA": [0.002, 0.001, -0.062, -0.012, 0.003],
    "LLY": [0.003, 0.088, 0.012, -0.004, 0.002],
    "XOM": [0.002, 0.006, 0.001, 0.004, 0.000],
    "CVX": [0.001, 0.005, 0.000, 0.003, 0.000],
    "NVO": [0.000, -0.071, -0.008, 0.002, 0.000],
    "AMD": [0.001, 0.002, -0.004, 0.000, -0.034],  # falls with no news
    "TSM": [0.002, 0.003, 0.004, 0.006, 0.002],
    "MSFT": [0.000, 0.001, -0.018, 0.004, 0.001],
    "META": [0.001, 0.000, -0.006, 0.014, 0.000],
    "AAPL": [0.002, -0.001, 0.003, 0.001, 0.002],
}

# (symbol, session offset from the last session (0 = as-of day), UTC hour, title, summary)
_MOCK_NEWS: list[tuple[str, int, int, str, str]] = [
    ("MSFT", -3, 21, "Microsoft signals slower growth in data center spending next year",
     "On its investor call the CFO said capital spending growth will 'moderate meaningfully' in "
     "fiscal 2027 as the company digests AI capacity built over the past two years. Azure growth "
     "guidance was unchanged."),
    ("MSFT", -2, 13, "Analysts split on whether Microsoft's capex comments mark a peak in AI spending",
     "Some analysts read the remarks as a digestion pause; others warned other hyperscalers could follow."),
    ("NVDA", -2, 16, "Nvidia shares slide as Microsoft's capex comments rattle AI chip stocks",
     "Nvidia fell the most among large chipmakers. Microsoft is estimated to be one of its largest "
     "customers for data center GPUs."),
    ("NVDA", -1, 15, "Nvidia extends losses; options traders pay up for protection",
     "Implied volatility rose as investors weighed whether other cloud customers will slow orders."),
    ("META", -1, 12, "Meta reiterates plan to raise AI infrastructure spending next year",
     "Meta said it still expects capital expenditure to rise 'significantly' in 2027 to support model training."),
    ("GOOGL", -4, 14, "Alphabet expands TPU access to outside cloud customers",
     "Google Cloud will rent its in-house AI chips to more third-party developers, a potential "
     "alternative to Nvidia GPUs for some workloads."),
    ("AVGO", -2, 17, "Broadcom slips with chip stocks after Microsoft spending comments",
     "Broadcom fell alongside other AI-exposed semiconductor names."),
    ("TSM", -1, 9, "TSMC to report third-quarter results next week; AI demand in focus",
     "Investors will look for commentary on 2027 AI accelerator demand and 2nm ramp timing."),
    ("TSM", -9, 8, "TSMC September revenue rises 38% from a year earlier",
     "Monthly sales came in ahead of analyst estimates on strong advanced-node demand."),
    ("LLY", -3, 12, "Eli Lilly's oral GLP-1 pill meets primary goal in late-stage obesity trial",
     "Patients on the highest dose lost an average of 13.6% of body weight at 72 weeks; "
     "discontinuation rates due to side effects were in line with injectables."),
    ("LLY", -2, 14, "Lilly plans to file oral obesity drug with regulators by year-end",
     "The company said it has built manufacturing capacity ahead of a potential launch."),
    ("NVO", -3, 13, "Novo Nordisk shares fall after rival's obesity pill trial data",
     "Investors worry about competition for Novo's own oral and injectable obesity drugs."),
    ("XOM", -3, 15, "Oil climbs to seven-week high as OPEC+ extends supply cuts",
     "Brent crude rose above $78 a barrel after the group extended voluntary cuts through Q1."),
    ("XOM", -1, 16, "Exxon gains with crude; Guyana output running ahead of schedule",
     "Exxon said its newest Guyana production vessel reached full capacity early."),
    ("CVX", -3, 15, "Chevron and other oil majors rise as crude rallies", "Energy was the best-performing sector."),
    ("AAPL", -4, 11, "Supplier checks point to steady iPhone demand into the holidays",
     "Analysts said orders at assemblers were tracking in line with last year."),
    ("AMZN", -2, 18, "AWS adds new AI infrastructure regions in Europe and Asia",
     "Amazon said the regions will offer both Nvidia GPUs and its own Trainium chips."),
    ("JPM", 0, 12, "JPMorgan to report earnings next week; investment banking fees seen higher",
     "Analysts expect advisory and underwriting fees to rise from a year earlier."),
    ("COST", -4, 21, "Costco September comparable sales rise 6.1%", "E-commerce sales grew 13%."),
    ("AMD", -12, 14, "AMD shows next-generation MI accelerator roadmap at developer event",
     "The company reiterated its AI GPU revenue targets for next year."),
    ("INTC", -5, 13, "Intel names new head of foundry business", "The appointment follows a reorganisation."),
]

# (calendar days from as-of, implied move %)
_MOCK_EARNINGS: dict[str, tuple[int, float]] = {
    "TSM": (4, 6.1),
    "JPM": (8, 3.0),
    "MSFT": (21, 4.5),
    "AMZN": (23, 6.8),
    "AAPL": (23, 4.2),
    "XOM": (24, 2.9),
    "AMD": (27, 8.4),
    "LLY": (29, 5.5),
    "NVDA": (43, 7.9),
    "COST": (64, 3.4),
    "AVGO": (64, 6.5),
    "META": (22, 7.2),
    "GOOGL": (21, 5.9),
}

MOCK_PUBLISHER = "Demo Wire (synthetic)"


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _last_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


class MockProvider(DataProvider):
    """Deterministic synthetic market with a scripted final week.

    The final week: the market falls about 2%; Nvidia drops sharply after
    Microsoft signals slower data center spending; Eli Lilly jumps on trial data;
    oil (and Exxon) rise; AMD slides on the last day with no news; and TSMC
    reports in 4 days with a ~6% implied move.
    """

    name = "mock"
    is_mock = True
    SESSIONS = 560  # a little over two years

    def __init__(self, as_of: date | None = None, seed: int = 35):
        self._as_of = _last_weekday(as_of or date.today())
        self.seed = seed
        self._sessions = pd.bdate_range(end=pd.Timestamp(self._as_of), periods=self.SESSIONS)
        self._closes = self._generate()

    def as_of(self) -> date:
        return self._as_of

    # -- prices -------------------------------------------------------------

    def _generate(self) -> pd.DataFrame:
        n = self.SESSIONS
        rng = np.random.default_rng(self.seed)
        week = slice(n - 5, n)

        spy = rng.normal(0.0004, 0.0095, n)
        spy[week] = _WEEK_SPY
        semis = rng.normal(0.0, 0.012, n)
        semis[week] = _WEEK_SEMIS

        returns: dict[str, np.ndarray] = {"SPY": spy}
        excess: dict[str, np.ndarray] = {}
        for etf, (_, beta, vol) in _MOCK_ETFS.items():
            if etf == "SPY":
                continue
            ex = rng.normal(0.0, vol, n)
            if etf in _WEEK_SECTOR_EXCESS:
                ex[week] = _WEEK_SECTOR_EXCESS[etf]
            else:
                ex[week] *= 0.3
            # Sector ETF return = beta * SPY + noise; "excess" is relative to SPY.
            sec = beta * spy + ex
            sec[week] = spy[week] + ex[week]
            returns[etf] = sec
            excess[etf] = sec - spy

        for sym, st in _MOCK_STOCKS.items():
            idio = rng.normal(0.0, st.idio_vol, n)
            idio[week] = _WEEK_IDIO.get(sym, list(idio[week] * 0.3))
            sec_ex = excess.get(sector_etf(st.sector) or "", np.zeros(n))
            returns[sym] = st.mu + st.beta_mkt * spy + st.beta_sec * sec_ex + st.semis * semis + idio
            returns[sym][week] -= st.mu  # keep the scripted week clean

        closes = {}
        for sym, r in returns.items():
            end_price = _MOCK_STOCKS[sym].price if sym in _MOCK_STOCKS else _MOCK_ETFS[sym][0]
            closes[sym] = self._to_prices(r, end_price)
        return pd.DataFrame(closes, index=self._sessions)

    @staticmethod
    def _to_prices(returns: np.ndarray, end_price: float) -> np.ndarray:
        """Turn daily returns into a price path ending at ``end_price``.
        The first session's return is ignored (it has no prior close)."""
        growth = np.cumprod(1.0 + np.asarray(returns, dtype=float))
        growth = growth / growth[0]
        return end_price * growth / growth[-1]

    def _unknown_series(self, symbol: str) -> np.ndarray:
        """Plausible prices for symbols the mock doesn't script (e.g. after the
        user edits portfolio.csv)."""
        rng = np.random.default_rng(_stable_seed(symbol))
        spy_r = self._closes["SPY"].pct_change().fillna(0).to_numpy()
        beta = rng.uniform(0.6, 1.4)
        r = 0.0003 + beta * spy_r + rng.normal(0, 0.014, len(spy_r))
        return self._to_prices(r, float(rng.uniform(20, 400)))

    def prices(self, symbols: list[str], start: date) -> pd.DataFrame:
        out = {}
        for s in symbols:
            s = s.upper()
            out[s] = self._closes[s].to_numpy() if s in self._closes else self._unknown_series(s)
        df = pd.DataFrame(out, index=self._sessions)
        return df.loc[df.index >= pd.Timestamp(start)]

    # -- reference data -----------------------------------------------------

    def profile(self, symbol: str) -> dict[str, Any]:
        symbol = symbol.upper()
        st = _MOCK_STOCKS.get(symbol)
        if st is not None:
            return {"symbol": symbol, "name": st.name, "sector": st.sector,
                    "industry": None, "quote_type": "EQUITY"}
        if symbol in _MOCK_ETFS:
            return {"symbol": symbol, "name": f"{symbol} ETF", "sector": None,
                    "industry": None, "quote_type": "ETF"}
        return {"symbol": symbol, "name": symbol, "sector": None, "industry": None, "quote_type": None}

    def _session(self, offset: int) -> date:
        return self._sessions[len(self._sessions) - 1 + offset].date()

    def news(self, symbol: str, days: int = 14) -> list[dict[str, Any]]:
        symbol = symbol.upper()
        cutoff = datetime.combine(self._as_of, datetime.min.time(), tzinfo=timezone.utc) - timedelta(days=days)
        items = []
        for sym, offset, hour, title, summary in _MOCK_NEWS:
            if sym != symbol:
                continue
            published = datetime.combine(self._session(offset), datetime.min.time(),
                                         tzinfo=timezone.utc) + timedelta(hours=hour)
            if published < cutoff:
                continue
            items.append({
                "symbol": symbol,
                "title": title,
                "publisher": MOCK_PUBLISHER,
                "published": published.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "url": None,
                "summary": summary,
            })
        return sorted(items, key=lambda n: n["published"], reverse=True)

    def earnings(self, symbol: str) -> dict[str, Any] | None:
        symbol = symbol.upper()
        if symbol not in _MOCK_EARNINGS:
            return None
        days, move = _MOCK_EARNINGS[symbol]
        return {
            "symbol": symbol,
            "date": (self._as_of + timedelta(days=days)).isoformat(),
            "implied_move_pct": move,
            "implied_move_source": "synthetic",
        }


def make_provider(mock: bool, **kwargs) -> DataProvider:
    return MockProvider(**kwargs) if mock else LiveProvider(**kwargs)


__all__ = [
    "DataProvider", "LiveProvider", "MockProvider", "SECTOR_ETFS", "MARKET",
    "sector_etf", "normalise_sector", "normalise_news_item", "make_provider",
]
