"""The web app's Claude loop: send the conversation and tools, run whatever
tools Claude asks for, send the results back, and repeat until it answers.

    python -m app.agent "why did I lose money this week?" --mock
"""

from __future__ import annotations

import os

from .tools import TOOLS, run_tool

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_STEPS = 8
MAX_TOKENS = 16000
MAX_HISTORY = 20

# Models that support server-side refusal fallbacks.
FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

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


def system_prompt(analytics):
    source = "synthetic demo data (mock mode)" if analytics.provider.is_mock else "Yahoo Finance"
    theses = ", ".join(sorted(analytics.theses)) or "none"
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Context: latest prices are from {analytics.as_of} (may be intraday in live mode), "
        f"source: {source}. Holdings: {', '.join(analytics.symbols)}. Theses on file for: {theses}."
    )


def clean_history(messages):
    """Recent plain-text turns only, starting with the user, with back-to-back
    turns from the same role merged (the API requires alternating roles)."""
    history = []
    for message in messages[-MAX_HISTORY:]:
        role, content = message.get("role"), message.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        content = content.strip()[:8000]
        if history and history[-1]["role"] == role:
            history[-1]["content"] += "\n\n" + content
        else:
            history.append({"role": role, "content": content})

    while history and history[0]["role"] != "user":
        history.pop(0)
    if not history or history[-1]["role"] != "user":
        raise ValueError("The conversation must end with a user message.")
    return history


def run_agent(messages, analytics, client=None, model=None, max_steps=MAX_STEPS, on_tool=None):
    """Returns {answer, trace, charts, model, steps}. After max_steps rounds of
    tool calls, Claude has to answer with what it has."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    model = model or os.getenv("ANTHROPIC_MODEL") or DEFAULT_MODEL
    analytics.load()

    request = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": system_prompt(analytics),
        "tools": TOOLS,
        "cache_control": {"type": "ephemeral"},
    }
    effort = os.getenv("ANTHROPIC_EFFORT", "medium")
    if effort and "haiku" not in model:
        request["output_config"] = {"effort": effort}

    def ask(conversation, final=False):
        extra = {"tool_choice": {"type": "none"}} if final else {}
        if model in FALLBACK_MODELS and os.getenv("DESK_NOTE_FALLBACKS", "1") != "0":
            return client.beta.messages.create(
                **request, **extra, messages=conversation, betas=[FALLBACK_BETA], fallbacks="default"
            )
        return client.messages.create(**request, **extra, messages=conversation)

    conversation = clean_history(messages)
    trace, charts = [], []
    steps = 0
    response = ask(conversation)

    while response.stop_reason == "tool_use":
        calls = [block for block in response.content if block.type == "tool_use"]
        conversation.append({"role": "assistant", "content": response.content})

        if steps == max_steps:
            skipped = [
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "is_error": True,
                    "content": "Not run: step limit reached.",
                }
                for call in calls
            ]
            note = {
                "type": "text",
                "text": "Step limit reached. Answer now using only the tool results you have.",
            }
            conversation.append({"role": "user", "content": skipped + [note]})
            response = ask(conversation, final=True)
            break

        results = []
        for call in calls:
            outcome = run_tool(analytics, call.name, call.input)
            trace.append(outcome.trace_entry())
            if on_tool:
                on_tool(trace[-1])
            if outcome.chart:
                charts.append(outcome.chart)
            result = {"type": "tool_result", "tool_use_id": call.id, "content": outcome.content}
            if outcome.is_error:
                result["is_error"] = True
            results.append(result)
        # all results go back in one message, so Claude keeps making parallel calls
        conversation.append({"role": "user", "content": results})
        steps += 1
        response = ask(conversation)

    answer = "\n\n".join(b.text for b in response.content if b.type == "text").strip()
    if response.stop_reason == "refusal":
        answer = answer or "I can't help with that request."
    elif not answer:
        answer = "I couldn't put together an answer this time. Please try rephrasing the question."
    elif response.stop_reason == "max_tokens":
        answer += "\n\n_(Answer cut off at the length limit.)_"

    return {"answer": answer, "trace": trace, "charts": charts, "model": model, "steps": steps}


def main():
    import argparse
    import sys

    from dotenv import load_dotenv

    from .analytics import AnalyticsHolder

    load_dotenv()
    parser = argparse.ArgumentParser(description="Ask Desk Note a question from the command line.")
    parser.add_argument("question")
    parser.add_argument("--mock", action="store_true", help="use synthetic market data")
    parser.add_argument("--model")
    args = parser.parse_args()

    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        print("Add ANTHROPIC_API_KEY to .env to start chatting.", file=sys.stderr)
        return 2

    analytics = AnalyticsHolder(mock=args.mock).get()
    result = run_agent(
        [{"role": "user", "content": args.question}],
        analytics,
        model=args.model,
        on_tool=lambda step: print("  -", step["summary"], file=sys.stderr),
    )
    print("\n" + result["answer"] + "\n")
    for chart in result["charts"]:
        print("[chart]", chart["title"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
