"""LiveProvider logic that can be tested offline (no network)."""
from types import SimpleNamespace

from app.data import LiveProvider, _company_name


def offline_provider(tmp_path, ticker_news=(), search_news=()):
    p = LiveProvider.__new__(LiveProvider)
    from app.data import _DiskCache
    p.cache = _DiskCache(tmp_path)
    p._tickers = {}
    p._yf = SimpleNamespace(
        Ticker=lambda s: SimpleNamespace(get_news=lambda count: list(ticker_news)),
        Search=lambda s, news_count, max_results: SimpleNamespace(news=list(search_news)),
    )
    p._news_rss = lambda s: []
    return p


def test_news_falls_back_to_search_and_filters_by_symbol(tmp_path):
    import time
    now = int(time.time())
    search = [
        {"title": "About NVDA", "publisher": "A", "providerPublishTime": now, "relatedTickers": ["NVDA"]},
        {"title": "Market wrap", "publisher": "B", "providerPublishTime": now, "relatedTickers": ["SPY"]},
    ]
    p = offline_provider(tmp_path, ticker_news=[], search_news=search)
    news = p.news("NVDA", 7)
    assert [n["title"] for n in news] == ["About NVDA"]


def test_news_empty_everywhere_is_empty_list(tmp_path):
    assert offline_provider(tmp_path).news("NVDA", 7) == []


def test_company_name_avoids_truncated_short_name():
    assert _company_name({"shortName": "Taiwan Semiconductor Manufactur",
                          "longName": "Taiwan Semiconductor Manufacturing Company Limited"}).startswith("Taiwan Semiconductor Manufacturing")
    assert _company_name({"shortName": "Apple Inc.", "longName": "Apple Inc."}) == "Apple Inc."
