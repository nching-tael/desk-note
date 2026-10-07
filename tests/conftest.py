from datetime import date

import pytest

from app.analytics import Analytics
from app.data import MockProvider
from app.portfolio import load_holdings, load_theses

AS_OF = date(2026, 10, 7)  # a Wednesday


@pytest.fixture(scope="session")
def provider():
    return MockProvider(as_of=AS_OF)


@pytest.fixture(scope="session")
def analytics(provider):
    return Analytics(provider, load_holdings(), load_theses()).load()
