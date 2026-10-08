from pathlib import Path

import yaml

from app.tools import REGISTRY
from evals.run import score


def result(answer, tools=("get_attribution",), ungrounded=()):
    return {
        "answer": answer,
        "trace": [{"tool": t} for t in tools],
        "grounding": {"ungrounded": list(ungrounded), "numbers_checked": 3},
    }


CASE = {"question": "Why did I lose money?", "expect_tools": ["get_attribution"]}


def test_good_answer_passes():
    assert score(CASE, result("You lost $6,348, mostly the market."))["passed"]


def test_each_check_can_fail():
    assert not score(CASE, result("x", tools=["get_overview"]))["checks"]["used_expected_tool"]
    assert not score(CASE, result("x", ungrounded=["$688"]))["checks"]["numbers_traced"]
    assert not score(CASE, result("You should sell NVDA now."))["checks"]["no_advice"]
    assert not score(CASE, result("I'd recommend trimming SMH before earnings."))["checks"]["no_advice"]
    assert not score(CASE, result("I couldn't put together an answer this time."))["checks"]["answered"]


def test_describing_options_is_not_advice():
    assert score(CASE, result("Some investors trim before earnings; others hold. It's your call."))["passed"]


def test_questions_only_expect_real_web_tools():
    cases = yaml.safe_load((Path(__file__).parent.parent / "evals" / "questions.yaml").read_text())
    assert len(cases) == 20
    web_tools = {name for name, t in REGISTRY.items() if t.group == "analysis"}
    for case in cases:
        assert set(case["expect_tools"]) <= web_tools, case["question"]
