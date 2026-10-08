"""Targets, drift, new-money plans and the satellite scorecard, checked by hand."""

import json

import pytest
from helpers import DummyProvider, frame

from app.analytics import Analytics, AnalyticsHolder
from app.portfolio import Holding, Trade
from app.store import Store, StoreError
from app.tools import run_tool

TARGETS = {
    "A": {"target_pct": 70, "band_pct": 5, "role": "core"},
    "B": {"target_pct": 30, "band_pct": 2, "role": "satellite"},
}

# A at $100 and B at $50: 80 A + 40 B = $8,000 + $2,000 = $10,000, i.e. 80/20
# against a 70/30 target.
FLAT = frame({"A": [100.0] * 3, "B": [50.0] * 3, "SPY": [400.0, 404.0, 408.0]})


def drifted(targets=TARGETS):
    holdings = [Holding("A", 80, None), Holding("B", 40, None)]
    return Analytics(DummyProvider(FLAT), holdings, {}, targets=targets).load()


def test_drift_and_rebalance_by_hand():
    result = drifted().allocation()
    a, b = result["positions"]
    assert (a["weight_pct"], a["drift_pct_points"], a["outside_band"]) == (80.0, 10.0, True)
    assert (b["weight_pct"], b["drift_pct_points"], b["outside_band"]) == (20.0, -10.0, True)
    # to be 70/30 on $10,000: sell $1,000 of A (10 shares), buy $1,000 of B (20 shares)
    assert (a["to_rebalance_dollars"], a["to_rebalance_shares"]) == (-1000, -10.0)
    assert (b["to_rebalance_dollars"], b["to_rebalance_shares"]) == (1000, 20.0)
    assert result["by_role"]["core"]["drift_pct_points"] == 10.0
    assert result["outside_band"] == ["A", "B"]


def test_new_money_goes_to_the_gap_first():
    # $1,000 more makes $11,000: A is already above its $7,700 target, B is
    # $1,300 short, so all $1,000 goes to B (20 shares).
    plan = drifted().allocation(new_money=1000)["new_money_plan"]
    assert plan["buys"] == [{"symbol": "B", "dollars": 1000, "shares": 20.0}]
    assert plan["weights_after_pct"] == {"A": 72.7, "B": 27.3}
    assert plan["whole_shares_only"] == {
        "buys": [{"symbol": "B", "shares": 20, "cost": 1000}],
        "cash_left": 0,
    }


def test_whole_share_plan_buys_what_it_can_afford():
    # $150: B ($50) is furthest below target, so it gets shares; A ($100) only
    # once B's gap is smaller than A's, which never happens here.
    plan = drifted().allocation(new_money=150)["new_money_plan"]["whole_shares_only"]
    assert plan == {"buys": [{"symbol": "B", "shares": 3, "cost": 150}], "cash_left": 0}


def test_enough_new_money_lands_exactly_on_target():
    # $5,000 more makes $15,000: A needs $2,500 to reach $10,500, B $2,500 to reach $4,500.
    plan = drifted().allocation(new_money=5000)["new_money_plan"]
    assert {b["symbol"]: b["dollars"] for b in plan["buys"]} == {"A": 2500, "B": 2500}
    assert plan["weights_after_pct"] == {"A": 70.0, "B": 30.0}


def test_untargeted_holding_is_flagged_for_sale():
    holdings = [Holding("A", 80, None), Holding("B", 40, None), Holding("C", 10, None)]
    closes = frame({**{k: list(v) for k, v in FLAT.items()}, "C": [20.0] * 3})
    result = Analytics(DummyProvider(closes), holdings, {}, targets=TARGETS).load().allocation()
    c = next(p for p in result["positions"] if p["symbol"] == "C")
    assert c["role"] == "not in targets" and c["to_rebalance_dollars"] == -200


def test_no_targets_explains_how_to_set_them():
    result = drifted(targets={}).allocation()
    assert result["targets_set"] is False and "set_targets" in result["note"]
    assert result["current_weights_pct"] == {"A": 80.0, "B": 20.0}


# Scorecard: core C goes 100 -> 110 -> 99 (+10%, -10%); satellite S goes
# 50 -> 50 -> 60 (0%, +20%). 20 shares of S = $1,000 exposure each day.
#   S actual gain:            20 x $0 + 20 x $10 = $200
#   same $1,000 in the core:  $1,000 x 10% + $1,000 x -10% = $0
#   added vs core:            $200
SCORE = frame({"C": [100.0, 110.0, 99.0], "S": [50.0, 50.0, 60.0], "SPY": [400.0, 404.0, 408.0]})
ROLES = {
    "C": {"target_pct": 80, "band_pct": 5, "role": "core"},
    "S": {"target_pct": 20, "band_pct": 2, "role": "satellite"},
}


def score(trades=(), targets=ROLES):
    holdings = [Holding("C", 10, None), Holding("S", 20, None)]
    return Analytics(DummyProvider(SCORE), holdings, {}, list(trades), targets).load().scorecard("1w")


def test_scorecard_by_hand():
    card = score()
    (s,) = card["satellites"]
    assert (s["gain_dollars"], s["same_money_in_core_dollars"], s["added_vs_core_dollars"]) == (200, 0, 200)
    assert s["return_pct"] == 20.0
    assert s["core_return_same_days_pct"] == -1.0  # 1.10 x 0.90 - 1
    assert card["compared_with"] == "your core (C)" and card["satellites_beating_core"] == "1 of 1"


def test_scorecard_follows_buys_during_the_period():
    # Buying 20 more S at day 1's close doubles the exposure for day 2:
    #   actual: 20 x $0 + 40 x $10 = $400
    #   core:   $1,000 x 10% + $2,000 x -10% = -$100
    buy = Trade(SCORE.index[1].date(), "S", "buy", 20, 50.0, id=1)
    (s,) = score([buy])["satellites"]
    assert (s["gain_dollars"], s["same_money_in_core_dollars"], s["added_vs_core_dollars"]) == (
        400,
        -100,
        500,
    )


def test_scorecard_without_a_core_uses_the_market():
    card = score(targets={})
    assert card["compared_with"].startswith("S&P 500")
    assert {s["symbol"] for s in card["satellites"]} == {"C", "S"}


# --- storing targets


@pytest.fixture()
def store():
    s = Store(":memory:")
    s.seed_from_files()
    return s


def test_set_targets_validates_and_defaults(store):
    saved = store.set_targets([{"symbol": "nvda", "target_pct": 60}, {"symbol": "AMD", "target_pct": 40}])
    assert saved["NVDA"] == {"target_pct": 60, "band_pct": 5.0, "role": "core"}
    assert saved["AMD"]["role"] == "core"  # 40% or more defaults to core
    with pytest.raises(StoreError, match="add up to 90"):
        store.set_targets([{"symbol": "NVDA", "target_pct": 90}])
    with pytest.raises(StoreError, match="over 100"):
        store.set_targets([{"symbol": "TSM", "target_pct": 10}], replace=False)
    store.set_targets(
        [
            {"symbol": "AMD", "target_pct": 30, "role": "satellite"},
            {"symbol": "TSM", "target_pct": 10, "role": "satellite"},
        ],
        replace=False,
    )
    assert set(store.targets()) == {"NVDA", "AMD", "TSM"}
    assert store.targets()["TSM"]["band_pct"] == 2.0


def test_set_targets_tool_updates_allocation(provider, store):
    holder = AnalyticsHolder(mock=True, store=store)
    holder.provider = provider
    targets = [
        {"symbol": "NVDA", "target_pct": 50, "role": "core"},
        {"symbol": "TSM", "target_pct": 50, "role": "satellite"},
    ]
    out = run_tool(holder.get(), "set_targets", {"targets": targets}, store=store)
    assert not out.is_error and out.changed
    holder.invalidate()
    allocation = json.loads(run_tool(holder.get(), "get_allocation", {"new_money": 1000}).content)
    assert allocation["targets_set"] and allocation["new_money_plan"]["amount"] == 1000
    assert run_tool(
        holder.get(), "set_targets", {"targets": [{"symbol": "NVDA", "target_pct": 20}]}, store=store
    ).is_error
