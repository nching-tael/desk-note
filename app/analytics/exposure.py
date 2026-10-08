"""Funds: how each one is classified for attribution, and look-through to
what the portfolio really owns underneath its ETFs."""

from __future__ import annotations

from collections import defaultdict

from ..data import sector_etf
from .periods import pct, usd

# A fund this concentrated in one sector is treated as a sector fund.
SECTOR_FUND_THRESHOLD = 0.5


def classify_fund(fund):
    """(sector label, sector ETF to use as its factor or None)."""
    if fund["asset_classes"].get("bonds", 0) > 0.5:
        return "Bonds", None
    sectors = fund["sector_weights"]
    if sectors:
        sector, weight = max(sectors.items(), key=lambda item: item[1])
        if weight >= SECTOR_FUND_THRESHOLD:
            return sector, sector_etf(sector)
        return "Diversified fund", None
    # no stock sectors at all: gold, bitcoin and the like
    return non_stock_label(fund), None


def non_stock_label(fund):
    return fund.get("category") or "Unclassified"


def look_through(a):
    """Exposure to each underlying stock (direct plus through funds) and to
    each sector, using every fund's published top holdings and sector weights."""
    a.load()
    total = float(a.values.iloc[-1][a.symbols].sum())
    stocks = {}
    sectors = defaultdict(float)
    funds = []

    def stock(symbol, name):
        return stocks.setdefault(symbol, {"name": name, "direct": 0.0, "via": defaultdict(float)})

    for symbol in a.symbols:
        value = float(a.values[symbol].iloc[-1])
        fund = a.funds.get(symbol)
        if fund is None:
            stock(symbol, a.name(symbol))["direct"] += value
            sectors[a.sector_of[symbol]] += value
            continue

        covered = 0.0
        for holding in fund["top_holdings"]:
            stock(holding["symbol"], holding["name"])["via"][symbol] += value * holding["weight"]
            covered += holding["weight"]

        add_fund_sectors(sectors, fund, value)
        funds.append(
            {
                "symbol": symbol,
                "name": a.name(symbol),
                "category": fund.get("category"),
                "treated_as": a.sector_of[symbol],
                "value_dollars": usd(value),
                "weight_pct": pct(value / total),
                "top_holdings_cover_pct": pct(covered),
            }
        )

    exposures = []
    for symbol, s in stocks.items():
        through_funds = sum(s["via"].values())
        exposures.append(
            {
                "symbol": symbol,
                "name": s["name"],
                "total_dollars": usd(s["direct"] + through_funds),
                "pct": pct((s["direct"] + through_funds) / total),
                "direct_dollars": usd(s["direct"]),
                "via_funds": {fund: usd(v) for fund, v in sorted(s["via"].items(), key=lambda kv: -kv[1])},
            }
        )
    exposures.sort(key=lambda e: -e["total_dollars"])
    overlaps = [e for e in exposures if len(e["via_funds"]) + (e["direct_dollars"] > 0) >= 2]

    named = sum(e["total_dollars"] for e in exposures)
    return {
        "as_of": a.as_of,
        "total_value": usd(total),
        "top_exposures": exposures[:15],
        "overlaps": overlaps[:10],
        "sector_exposure_pct": sector_percentages(sectors, total),
        "funds": sorted(funds, key=lambda f: -f["value_dollars"]),
        "named_stocks_cover_pct": pct(named / total),
        "note": (
            "Funds are opened up using their published top holdings (usually the top 10), so the "
            "rest of each fund isn't broken down by stock. Sector exposure uses each fund's full "
            "sector weights. Foreign listings (e.g. 2330.TW for TSMC) count separately from US tickers."
        ),
    }


def add_fund_sectors(sectors, fund, value):
    classes = fund["asset_classes"]
    weights = fund["sector_weights"]
    bonds = classes.get("bonds", 0.0)
    if not weights and not bonds:
        sectors[non_stock_label(fund)] += value
        return

    stock_share = (classes.get("stocks") or (0.0 if bonds else 1.0)) if weights else 0.0
    weight_total = sum(weights.values())
    for sector, weight in weights.items():
        sectors[sector] += value * stock_share * weight / weight_total
    sectors["Bonds"] += value * bonds
    rest = value * (1 - stock_share - bonds)
    if rest > 0:
        sectors["Cash and other"] += rest


def sector_percentages(sectors, total):
    ordered = sorted(sectors.items(), key=lambda item: -item[1])
    return {name: pct(value / total) for name, value in ordered if value > 0}
