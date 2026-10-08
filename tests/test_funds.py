"""ETFs: classification for attribution, and look-through exposure."""

import numpy as np
import pytest

from app.analytics import Analytics
from app.analytics.exposure import classify_fund
from app.portfolio import Holding


@pytest.fixture(scope="module")
def etfs(provider):
    holdings = [
        Holding("VOO", 100, 500.0),
        Holding("SMH", 50, 250.0),
        Holding("NVDA", 20, 150.0),
        Holding("BND", 100, 72.0),
    ]
    return Analytics(provider, holdings, {}).load()


def value(a, symbol):
    return float(a.values[symbol].iloc[-1])


def test_funds_are_classified(provider):
    assert classify_fund(provider.fund("VOO")) == ("Diversified fund", None)
    assert classify_fund(provider.fund("SMH")) == ("Technology", "XLK")
    assert classify_fund(provider.fund("XLE")) == ("Energy", "XLE")
    assert classify_fund(provider.fund("BND")) == ("Bonds", None)
    assert provider.fund("NVDA") is None


def test_sector_fund_gets_a_sector_factor(etfs):
    assert etfs.etf_of["SMH"] == "XLK" and etfs.sector_of["SMH"] == "Technology"
    assert etfs.etf_of["VOO"] is None and etfs.etf_of["BND"] is None
    df, _ = etfs.attribution_frame("1w")
    np.testing.assert_allclose(
        df["market"] + df["sector_move"] + df["stock_specific"], df["total"], atol=1e-6
    )
    assert df.loc["SMH", "sector_move"] != 0
    assert df.loc["VOO", "sector_move"] == 0  # broad fund: market only
    assert df.loc["SMH", "beta_sector"] > 0.5


def test_broad_fund_is_mostly_market(etfs):
    df, _ = etfs.attribution_frame("1m")
    voo = df.loc["VOO"]
    assert voo["beta_mkt"] == pytest.approx(1.1, abs=0.2)
    assert abs(voo["market"]) > abs(voo["stock_specific"])


def test_look_through_nvidia_by_hand(etfs):
    # NVDA: 20 shares held directly + 8.1% of VOO + 19.3% of SMH.
    expected = value(etfs, "NVDA") + 0.081 * value(etfs, "VOO") + 0.193 * value(etfs, "SMH")
    nvda = next(e for e in etfs.exposure()["top_exposures"] if e["symbol"] == "NVDA")
    assert nvda["total_dollars"] == pytest.approx(expected, abs=1)
    assert nvda["direct_dollars"] == round(value(etfs, "NVDA"))
    assert set(nvda["via_funds"]) == {"VOO", "SMH"}


def test_look_through_overlaps_and_sectors(etfs):
    result = etfs.exposure()
    assert result["top_exposures"][0]["symbol"] == "NVDA"
    assert {e["symbol"] for e in result["overlaps"]} >= {"NVDA", "AVGO"}  # AVGO is in both VOO and SMH
    sectors = result["sector_exposure_pct"]
    assert sum(sectors.values()) == pytest.approx(100, abs=0.5)
    total = float(etfs.values.iloc[-1].sum())
    assert sectors["Bonds"] == round(value(etfs, "BND") * 0.986 / total * 100, 1)
    assert "Diversified fund" not in sectors  # VOO is spread across its real sectors
    smh = next(f for f in result["funds"] if f["symbol"] == "SMH")
    assert smh["top_holdings_cover_pct"] == 54.4 and smh["treated_as"] == "Technology"


def test_risk_uses_look_through_sectors(etfs):
    assert "Bonds" in etfs.risk()["sector_weights_pct"]


def test_stock_only_portfolio_looks_through_to_itself(analytics):
    result = analytics.exposure()
    assert result["funds"] == [] and result["overlaps"] == []
    assert result["named_stocks_cover_pct"] == 100.0
    assert sum(result["sector_exposure_pct"].values()) == pytest.approx(100, abs=0.5)


def test_fund_with_no_data_is_unclassified(provider, monkeypatch):
    real = provider.fund
    monkeypatch.setattr(
        provider,
        "fund",
        lambda s: (
            {"category": None, "sector_weights": {}, "top_holdings": [], "asset_classes": {}}
            if s == "VOO"
            else real(s)
        ),
    )
    a = Analytics(provider, [Holding("VOO", 10, None), Holding("NVDA", 5, None)], {}).load()
    assert a.sector_of["VOO"] == "Diversified fund"
    assert "Unclassified" in a.exposure()["sector_exposure_pct"]
