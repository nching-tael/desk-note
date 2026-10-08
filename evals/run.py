"""Run the eval questions through the web agent with the real Claude API.

    python -m evals.run                 # all questions
    python -m evals.run --only 3        # the first 3
    python -m evals.run --model claude-opus-5-5

Each answer is scored on: it answered, it called an expected tool, every
number in it traces back to a tool result, and it doesn't tell the user to
buy or sell. Uses mock data, so runs are comparable. Costs real API usage,
roughly a few cents per question.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

HERE = Path(__file__).parent
# "you should sell", "I'd recommend trimming", "I advise buying"...
ADVICE = re.compile(
    r"\b(you should|i('d| would)? (recommend|suggest)|i advise)\b"
    r"[^.]{0,40}\b(buy|sell|trim|add|exit|dump)\w*",
    re.IGNORECASE,
)
FALLBACK_ANSWERS = ("I couldn't put together an answer", "I can't help with that request")


def score(case, result):
    tools = [step["tool"] for step in result["trace"]]
    answer = result["answer"]
    checks = {
        "answered": bool(answer.strip()) and not answer.startswith(FALLBACK_ANSWERS),
        "used_expected_tool": not case.get("expect_tools") or any(t in tools for t in case["expect_tools"]),
        "numbers_traced": not result["grounding"]["ungrounded"],
        "no_advice": not ADVICE.search(answer),
    }
    return {
        "question": case["question"],
        "passed": all(checks.values()),
        "checks": checks,
        "tools": tools,
        "ungrounded": result["grounding"]["ungrounded"],
        "numbers_checked": result["grounding"]["numbers_checked"],
        "answer": answer,
    }


def main():
    from app.agent import run_agent
    from app.analytics import AnalyticsHolder

    load_dotenv()
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("The eval calls the real Claude API: add ANTHROPIC_API_KEY to .env first.")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", type=int, help="run just the first N questions")
    parser.add_argument("--model")
    args = parser.parse_args()

    cases = yaml.safe_load((HERE / "questions.yaml").read_text())[: args.only]
    analytics = AnalyticsHolder(mock=True).get()
    results = []
    for i, case in enumerate(cases, 1):
        started = time.time()
        try:
            result = score(
                case, run_agent([{"role": "user", "content": case["question"]}], analytics, model=args.model)
            )
        except Exception as e:  # keep going; one API error shouldn't sink the run
            result = {"question": case["question"], "passed": False, "error": f"{type(e).__name__}: {e}"}
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        failed = [k for k, ok in result.get("checks", {}).items() if not ok] or [result.get("error", "")]
        print(
            f"{i:2}. {status} {time.time() - started:4.1f}s  {case['question']}"
            + ("" if result["passed"] else f"  <- {', '.join(failed)}")
        )
        if result.get("ungrounded"):
            print(f"      untraced numbers: {', '.join(result['ungrounded'])}")

    passed = sum(r["passed"] for r in results)
    print(f"\n{passed}/{len(results)} passed")
    out = HERE / "results" / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Answers saved to {out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
