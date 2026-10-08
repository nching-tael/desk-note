from .base import MARKET, SECTOR_ETFS, DataProvider, normalise_sector, sector_etf
from .mock import MockProvider
from .yahoo import LiveProvider, normalise_news_item


def make_provider(mock):
    return MockProvider() if mock else LiveProvider()
