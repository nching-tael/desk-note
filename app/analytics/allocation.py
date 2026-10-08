"""Target allocation: how far each position has drifted, what it would take to
get back on target, and how to split new money without selling anything."""

from __future__ import annotations

from .periods import pct, usd


def allocation(a, new_money=0.0):
    a.load()
    values = {s: float(a.values[s].iloc[-1]) for s in a.symbols}
    total = sum(values.values())

    if not a.targets:
        return {
            "targets_set": False,
            "total_value": usd(total),
            "current_weights_pct": {
                s: pct(v / total) for s, v in sorted(values.items(), key=lambda kv: -kv[1])
            },
            "note": "No target allocation saved yet. Set one with set_targets, e.g. VOO 70% as core "
            "and the rest split between satellites.",
        }

    prices = {s: latest_price(a, s) for s in set(a.targets) | set(values)}
    targets = {s: t["target_pct"] / 100 for s, t in a.targets.items()}
    positions = []
    for symbol in list(a.targets) + [s for s in values if s not in a.targets]:
        target = a.targets.get(symbol)
        value = values.get(symbol, 0.0)
        drift = (value / total - targets.get(symbol, 0.0)) * 100
        to_target = targets.get(symbol, 0.0) * total - value
        positions.append(
            {
                "symbol": symbol,
                "role": target["role"] if target else "not in targets",
                "value": usd(value),
                "weight_pct": pct(value / total),
                "target_pct": target["target_pct"] if target else 0,
                "drift_pct_points": round(drift, 1) + 0.0,
                "band_pct_points": target["band_pct"] if target else 0,
                "outside_band": abs(drift) > (target["band_pct"] if target else 0),
                "to_rebalance_dollars": usd(to_target),
                "to_rebalance_shares": round(to_target / prices[symbol], 2) if prices.get(symbol) else None,
            }
        )

    by_role = {}
    for p in positions:
        role = by_role.setdefault(p["role"], {"weight": 0.0, "target": 0.0})
        role["weight"] += values.get(p["symbol"], 0.0) / total
        role["target"] += targets.get(p["symbol"], 0.0)

    result = {
        "targets_set": True,
        "total_value": usd(total),
        "positions": positions,
        "by_role": {
            role: {
                "weight_pct": pct(r["weight"]),
                "target_pct": pct(r["target"]),
                "drift_pct_points": round((r["weight"] - r["target"]) * 100, 1) + 0.0,
            }
            for role, r in by_role.items()
        },
        "outside_band": [p["symbol"] for p in positions if p["outside_band"]],
        "note": "to_rebalance is the trade that would put each position exactly on target "
        "(positive = buy, negative = sell). It applies the user's own targets; it is not advice.",
    }
    if new_money > 0:
        result["new_money_plan"] = plan_new_money(values, targets, prices, new_money)
    return result


def plan_new_money(values, targets, prices, amount):
    """Put new money where it closes the biggest gaps to target, never selling.
    If every gap is filled, the rest is split by target weight."""
    total_after = sum(values.values()) + amount
    gaps = {s: max(0.0, w * total_after - values.get(s, 0.0)) for s, w in targets.items()}
    gap_total = sum(gaps.values())
    if gap_total >= amount:
        spend = {s: amount * g / gap_total for s, g in gaps.items()}
    else:
        spare = amount - gap_total
        spend = {s: gaps[s] + spare * w for s, w in targets.items()}

    buys = [
        {
            "symbol": symbol,
            "dollars": usd(dollars),
            "shares": round(dollars / prices[symbol], 3) if prices.get(symbol) else None,
        }
        for symbol, dollars in sorted(spend.items(), key=lambda kv: -kv[1])
        if dollars >= 1
    ]
    after = dict(values)
    for symbol, dollars in spend.items():
        after[symbol] = after.get(symbol, 0.0) + dollars

    return {
        "amount": usd(amount),
        "buys": buys,
        "weights_after_pct": {
            s: pct(v / total_after) for s, v in sorted(after.items(), key=lambda kv: -kv[1])
        },
        "whole_shares_only": whole_share_plan(values, targets, prices, amount),
    }


def whole_share_plan(values, targets, prices, amount):
    """For brokers without fractional shares: buy one share at a time of
    whichever position is furthest below target and still affordable."""
    total_after = sum(values.values()) + amount
    holding = dict(values)
    shares = {}
    cash = amount
    while True:
        affordable = [s for s in targets if prices.get(s) and prices[s] <= cash]
        if not affordable:
            break
        symbol = max(affordable, key=lambda s: targets[s] * total_after - holding.get(s, 0.0))
        cash -= prices[symbol]
        holding[symbol] = holding.get(symbol, 0.0) + prices[symbol]
        shares[symbol] = shares.get(symbol, 0) + 1
    return {
        "buys": [{"symbol": s, "shares": n, "cost": usd(n * prices[s])} for s, n in shares.items()],
        "cash_left": usd(cash),
    }


def latest_price(a, symbol):
    prices = a.price_series(symbol)
    if prices is None or prices.isna().all():
        return None
    return float(prices.dropna().iloc[-1])
