"""Yahoo provider logic that can be tested without the network."""

import time
from types import SimpleNamespace

from app.data.yahoo import DiskCache, LiveProvider, company_name


def offline_provider(tmp_path, ticker_news=(), search_news=()):
    provider = LiveProvider.__new__(LiveProvider)
    provider.cache = DiskCache(tmp_path)
    provider.tickers = {}
    provider.yf = SimpleNamespace(
        Ticker=lambda s: SimpleNamespace(get_news=lambda count: list(ticker_news)),
        Search=lambda s, news_count, max_results: SimpleNamespace(news=list(search_news)),
    )
    provider.news_from_rss = lambda s: []
    return provider


def test_news_falls_back_to_search_and_keeps_tagged_stories(tmp_path):
    now = int(time.time())
    search = [
        {"title": "About NVDA", "publisher": "A", "providerPublishTime": now, "relatedTickers": ["NVDA"]},
        {"title": "Market wrap", "publisher": "B", "providerPublishTime": now, "relatedTickers": ["SPY"]},
    ]
    news = offline_provider(tmp_path, search_news=search).news("NVDA", 7)
    assert [n["title"] for n in news] == ["About NVDA"]


def test_no_news_anywhere_is_an_empty_list(tmp_path):
    assert offline_provider(tmp_path).news("NVDA", 7) == []


def test_company_name_avoids_truncated_short_name():
    tsmc = {
        "shortName": "Taiwan Semiconductor Manufactur",
        "longName": "Taiwan Semiconductor Manufacturing Company Limited",
    }
    assert company_name(tsmc) == "Taiwan Semiconductor Manufacturing Company Limited"
    assert company_name({"shortName": "Apple Inc.", "longName": "Apple Inc."}) == "Apple Inc."
