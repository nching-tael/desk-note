"""Claude tool-use loop.

``run_agent`` sends the conversation plus tool schemas to Claude, executes any
tool calls against the analytics layer, feeds the results back, and repeats
until Claude answers (or the step limit is hit). It returns::

    {"answer": str, "trace": [tool call summaries], "charts": [chart specs]}
"""
from __future__ import annotations

import os
from typing import Any, Callable

from .analytics import Analytics
from .tools import TOOLS, run_tool

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_STEPS = 8
MAX_TOKENS = 16000
MAX_HISTORY = 20

# Models that accept server-side refusal fallbacks (`fallbacks: "default"`).
_FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """\
You are Desk Note, a personal portfolio analyst for a retail investor. You answer questions about \
their actual holdings using the tools provided.

How you work:
- Use tools for every fact and number. Never state a figure (price, $, %, beta, date, implied move) \
that did not come from a tool result in this conversation, and never do arithmetic yourself, not even \
simple sums or differences. If a number you want isn't in a tool result, say so or describe it \
qualitatively.
- Lead with dollars, then the percentage: "-$420 (-1.3%)".
- Separate luck from skill: distinguish what the market or sector did from what the user's specific \
stock picks did.
- Be honest about uncertainty. If the news doesn't explain a move, say that plainly; never invent a reason.
- This is information, not advice. If asked whether to buy or sell, lay out the relevant facts and \
considerations without recommending a trade, and add one short line that this isn't financial advice.

Answer format: one direct sentence that answers the question, then at most 3-4 short supporting \
points. Markdown is fine (bold, bullet lists), but no headers. Keep it brief and plain-spoken.

Playbook:
- "Why did I lose/make money": call get_attribution for the period (default 1w for "this week", 1d \
for "today"). Name the biggest drivers in dollars, separating the market from stock picks. Check \
get_news for the biggest stock-specific movers before explaining them.
- Thesis questions ("does X break my thesis?"): call get_thesis, then give a verdict: **supports**, \
**challenges** or **no change**. Cite the specific headlines, including read-across from watch-list \
companies (customers, suppliers, competitors), and say what to watch next (e.g. upcoming earnings).
- Risk questions: call get_risk_report and name the single biggest risk concretely, e.g. "41% of your \
money is in four semiconductor stocks that move together", using the tool's numbers.
- Call get_chart when a picture would help (attribution breakdowns, a stock vs the market, the \
portfolio vs the market). Charts are shown to the user under your answer; refer to them briefly.
- Run independent tool calls in parallel.
"""


def _context_note(analytics: Analytics) -> str:
    analytics.load()
    holdings = ", ".join(analytics.symbols)
    theses = ", ".join(sorted(analytics.theses)) or "none"
    source = "synthetic demo data (mock mode)" if analytics.provider.is_mock else "Yahoo Finance"
    return (f"\n\nContext: latest prices are from {analytics.as_of} (may be intraday in live mode), source: {source}. "
            f"Holdings: {holdings}. Theses on file for: {theses}.")


def _get(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _text_of(content: list[Any]) -> str:
    return "\n\n".join(_get(b, "text", "") for b in content if _get(b, "type") == "text").strip()


def clean_history(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Keep plain-text user/assistant turns, trimmed to the recent history,
    starting with a user turn and with consecutive same-role turns merged."""
    out: list[dict[str, str]] = []
    for m in messages[-MAX_HISTORY:]:
        role = m.get("role") if isinstance(m, dict) else None
        content = m.get("content") if isinstance(m, dict) else None
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        content = content.strip()[:8000]
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + content
        else:
            out.append({"role": role, "content": content})
    while out and out[0]["role"] != "user":
        out.pop(0)
    if not out or out[-1]["role"] != "user":
        raise ValueError("The conversation must end with a user message.")
    return out


def make_client():
    import anthropic

    return anthropic.Anthropic()


def run_agent(
    messages: list[dict[str, Any]],
    analytics: Analytics,
    client: Any = None,
    model: str | None = None,
    max_steps: int = MAX_STEPS,
    on_tool: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the tool-use loop. ``max_steps`` caps the number of tool-using
    rounds; if Claude still wants tools after that, it is asked to answer with
    what it has."""
    client = client or make_client()
    model = model or os.getenv("ANTHROPIC_MODEL") or DEFAULT_MODEL
    effort = os.getenv("ANTHROPIC_EFFORT", "medium")
    use_fallbacks = model in _FALLBACK_MODELS and os.getenv("DESK_NOTE_FALLBACKS", "1") != "0"

    convo: list[dict[str, Any]] = list(clean_history(messages))
    system = SYSTEM_PROMPT + _context_note(analytics)
    trace: list[dict[str, Any]] = []
    charts: list[dict[str, Any]] = []

    def create(final: bool = False):
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "tools": TOOLS,
            "messages": convo,
            "cache_control": {"type": "ephemeral"},
        }
        if final:
            kwargs["tool_choice"] = {"type": "none"}
        if effort and "haiku" not in model:
            kwargs["output_config"] = {"effort": effort}
        if use_fallbacks:
            return client.beta.messages.create(betas=[_FALLBACK_BETA], fallbacks="default", **kwargs)
        return client.messages.create(**kwargs)

    response = create()
    steps = 0
    while _get(response, "stop_reason") == "tool_use":
        if steps >= max_steps:
            convo.append({"role": "assistant", "content": _get(response, "content")})
            skipped = [{"type": "tool_result", "tool_use_id": _get(b, "id"), "is_error": True,
                        "content": "Not run: step limit reached."}
                       for b in _get(response, "content") or [] if _get(b, "type") == "tool_use"]
            convo.append({"role": "user", "content": skipped + [
                {"type": "text", "text": "Step limit reached. Answer now using only the tool results you already have."}]})
            response = create(final=True)
            break
        steps += 1
        convo.append({"role": "assistant", "content": _get(response, "content")})
        convo.append({"role": "user", "content": _tool_results(response, analytics, trace, charts, on_tool)})
        response = create()

    stop = _get(response, "stop_reason")
    answer = _text_of(_get(response, "content") or [])
    if stop == "refusal":
        answer = answer or "I can't help with that request."
    elif not answer:
        answer = "I couldn't put together an answer this time. Please try rephrasing the question."
    elif stop == "max_tokens":
        answer += "\n\n_(Answer cut off at the length limit.)_"
    return {"answer": answer, "trace": trace, "charts": charts, "model": model, "steps": steps}


def _tool_results(response: Any, analytics: Analytics, trace: list, charts: list,
                  on_tool: Callable[[dict[str, Any]], None] | None) -> list[dict[str, Any]]:
    results = []
    for block in _get(response, "content") or []:
        if _get(block, "type") != "tool_use":
            continue
        outcome = run_tool(analytics, _get(block, "name"), _get(block, "input") or {})
        entry = outcome.trace_entry()
        trace.append(entry)
        if on_tool:
            on_tool(entry)
        if outcome.chart:
            charts.append(outcome.chart)
        result = {"type": "tool_result", "tool_use_id": _get(block, "id"), "content": outcome.content}
        if outcome.is_error:
            result["is_error"] = True
        results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from dotenv import load_dotenv

    from .analytics import AnalyticsHolder

    load_dotenv()
    ap = argparse.ArgumentParser(description="Ask Desk Note a question from the command line.")
    ap.add_argument("question")
    ap.add_argument("--mock", action="store_true", help="use synthetic data (no network for prices)")
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        print("Add ANTHROPIC_API_KEY to .env to start chatting.", file=sys.stderr)
        return 2
    analytics = AnalyticsHolder(mock=args.mock).get()
    result = run_agent([{"role": "user", "content": args.question}], analytics, model=args.model,
                       on_tool=lambda e: print(f"  · {e['summary']}", file=sys.stderr))
    print("\n" + result["answer"] + "\n")
    for c in result["charts"]:
        print(f"[chart] {c['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
