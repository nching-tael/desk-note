import json

from app.grounding import check, numbers_in_text

TOOL = json.dumps(
    {
        "headline": "Of your -$6,348 over the last 5 trading days, -$4,136 was the market.",
        "total_value": 20130,
        "satellites": [
            {"symbol": "SMH", "added_vs_core_dollars": 441},
            {"symbol": "XBI", "added_vs_core_dollars": 247},
        ],
        "change_pct": -3.6,
        "weight_pct": 71.2,
    }
)


def ungrounded(answer, sources=(TOOL,)):
    return check(answer, list(sources))["ungrounded"]


def test_numbers_quoted_from_tools_pass():
    answer = (
        "You lost **$6,348** (-3.6%) this week; -$4,136 of it was the market. "
        "SMH added $441 vs the core and XBI $247. Your portfolio is worth $20,130."
    )
    result = check(answer, [TOOL])
    assert result["ungrounded"] == [] and result["numbers_checked"] == 6


def test_honest_rounding_passes():
    assert ungrounded("It's worth about $20k, roughly $20,100, and VOO is about 71% of it.") == []


def test_arithmetic_is_caught():
    # 441 + 247 = 688 isn't in any tool result: the model added them itself
    assert ungrounded("Together SMH and XBI added $688.") == ["$688"]


def test_made_up_and_misrounded_numbers_are_caught():
    assert ungrounded("Nvidia fell 7.4% after a 15% jump.") == ["7.4%", "15%"]
    assert ungrounded("VOO is 70% of the portfolio.") == ["70%"]  # tool says 71.2


def test_dates_years_counts_and_index_names_are_ignored():
    text = "Over the last 5 trading days (Oct 6, 2026; 2026-10-06), the S&P 500 and 3 satellites."
    assert numbers_in_text(text) == []


def test_numbers_from_the_question_count_as_sources():
    question = "If the market drops 10%, how much do I lose?"
    assert ungrounded("If the market drops 10%, you'd lose $2,258.", [TOOL, question]) == ["$2,258"]
    assert (
        ungrounded(
            "If the market drops 10%, you'd lose $2,258.", [TOOL, question, json.dumps({"dollars": -2258})]
        )
        == []
    )


def test_scales_and_signs():
    values = [(v, t) for v, t, _ in numbers_in_text("down -$1.2k, up +2.5 million, 12 bn")]
    assert values == [(1200.0, 50.0), (2_500_000.0, 50_000.0), (12e9, 5e8)]
