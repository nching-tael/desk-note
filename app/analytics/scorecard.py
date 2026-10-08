"""Satellite scorecard: did each satellite beat simply putting the same money
into the core?

For every day a satellite was held, the same dollar exposure is assumed to
have earned the core's return instead. Buys and sells along the way are
handled because the comparison follows the position's actual value.
"""

from __future__ import annotations

import numpy as np

from ..data import MARKET
from .periods import TRADING_DAYS, AnalyticsError, pct, usd


def scorecard(a, period="1y"):
    w = a.window(period)
    core = [s for s, t in a.targets.items() if t["role"] == "core" and s in a.all_symbols]
    benchmark, benchmark_name = core_returns(a, core, w)

    roles = {s: t["role"] for s, t in a.targets.items()}
    if any(r == "satellite" for r in roles.values()):
        candidates = [s for s in a.held_during(w) if roles.get(s) == "satellite"]
    else:
        candidates = [s for s in a.held_during(w) if s not in core]
    if not candidates:
        raise AnalyticsError("No satellite positions were held over that period.")

    rows = []
    for symbol in candidates:
        exposure = a.values[symbol].shift(1).iloc[w.days]  # value at the previous close
        held = exposure > 0
        actual = float(a.pnl_df[symbol].iloc[w.days].sum())
        if_core = float((exposure * benchmark).sum())
        own_returns = a.returns[symbol].iloc[w.days][held]
        volatility = annual_volatility(own_returns)
        rows.append(
            {
                "symbol": symbol,
                "gain_dollars": usd(actual),
                "same_money_in_core_dollars": usd(if_core),
                "added_vs_core_dollars": usd(actual - if_core),
                "return_pct": pct((1 + own_returns).prod() - 1),
                "core_return_same_days_pct": pct((1 + benchmark[held]).prod() - 1),
                "volatility_annual_pct": volatility,
                "days_held": int(held.sum()),
            }
        )
    rows.sort(key=lambda r: -r["added_vs_core_dollars"])

    added = sum(r["added_vs_core_dollars"] for r in rows)
    return {
        "period": w.period,
        "period_label": w.label,
        "start_date": w.start.date().isoformat(),
        "end_date": w.end.date().isoformat(),
        "compared_with": benchmark_name,
        "core_return_pct": pct((1 + benchmark).prod() - 1),
        "core_volatility_annual_pct": annual_volatility(benchmark),
        "satellites": rows,
        "total_added_vs_core_dollars": added,
        "satellites_beating_core": f"{sum(r['added_vs_core_dollars'] > 0 for r in rows)} of {len(rows)}",
        "note": "added_vs_core = the satellite's actual gain minus what the same dollars would have made in "
        "the core over the same days. Positive means the bet paid off versus just owning more core.",
    }


def annual_volatility(daily_returns):
    if len(daily_returns) <= 10:
        return None
    return pct(daily_returns.std() * np.sqrt(TRADING_DAYS))


def core_returns(a, core, w):
    """Daily returns of the core over the window (value-weighted if there's
    more than one core holding), or the S&P 500 if no core is defined."""
    spy = a.returns[MARKET].iloc[w.days]
    if not core:
        return spy, "S&P 500 (SPY), since no core holding is set"
    exposure = a.values[core].shift(1).iloc[w.days]
    gains = a.pnl_df[core].iloc[w.days].sum(axis=1)
    invested = exposure.sum(axis=1)
    daily = (gains / invested).where(invested > 0, spy)  # days before the core was bought: use SPY
    return daily, f"your core ({', '.join(core)})"
