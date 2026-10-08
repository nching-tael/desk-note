"""Agent loop tests with a fake Anthropic client issuing scripted tool calls."""

import json
from types import SimpleNamespace

import pytest

from app.agent import clean_history, run_agent


def text(t):
    return SimpleNamespace(type="text", text=t)


def tool_use(id_, name, inp):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


def response(stop_reason, *blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


class FakeClient:
    """Mimics client.messages.create / client.beta.messages.create."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # Snapshot the conversation as sent.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        if callable(self.script[0]):
            return self.script[0](len(self.calls))
        return self.script.pop(0)


def test_collects_trace_and_charts(analytics):
    client = FakeClient(
        [
            response(
                "tool_use",
                text("Let me check."),
                tool_use("t1", "get_attribution", {"period": "1w"}),
                tool_use("t2", "get_chart", {"kind": "attribution", "period": "1w"}),
            ),
            response("tool_use", tool_use("t3", "get_news", {"symbols": ["NVDA"], "days": 7})),
            response("end_turn", text("You lost money mostly because of the market.")),
        ]
    )
    out = run_agent(
        [{"role": "user", "content": "Why did I lose money this week?"}],
        analytics,
        client=client,
        model="claude-sonnet-5-5",
    )
    assert out["answer"] == "You lost money mostly because of the market."
    assert [t["tool"] for t in out["trace"]] == ["get_attribution", "get_chart", "get_news"]
    assert all(t["ok"] for t in out["trace"])
    assert len(out["charts"]) == 1 and out["charts"][0]["kind"] == "attribution"
    assert len(client.calls) == 3

    # Parallel tool results go back in a single user message, chart as a short note.
    second = client.calls[1]["messages"]
    results = second[-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert json.loads(results[1]["content"])["status"] == "Chart shown to user"
    assert "Of your" in json.loads(results[0]["content"])["headline"]

    # Request shape: tools + system + fallbacks for this model.
    first = client.calls[0]
    assert len(first["tools"]) == 11
    assert "Never state a figure" in first["system"]
    assert first["fallbacks"] == "default"
    assert "tool_choice" not in first


def test_tool_errors_are_returned_to_model_not_raised(analytics):
    client = FakeClient(
        [
            response("tool_use", tool_use("t1", "get_thesis", {"symbol": "ZZZZ"})),
            response("end_turn", text("I don't have a thesis for ZZZZ.")),
        ]
    )
    out = run_agent([{"role": "user", "content": "Thesis on ZZZZ?"}], analytics, client=client)
    assert out["trace"][0]["ok"] is False
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "error" in json.loads(result["content"])


def test_loop_stops_at_max_steps(analytics):
    def always_tools(n):
        if n <= 4:  # with max_steps=3, call 4 still wants tools; call 5 is the forced answer
            return response("tool_use", tool_use(f"t{n}", "get_overview", {}))
        return response("end_turn", text("Here's what I found so far."))

    client = FakeClient([always_tools])
    out = run_agent([{"role": "user", "content": "Loop forever"}], analytics, client=client, max_steps=3)
    assert len(client.calls) == 5
    assert len(out["trace"]) == 3
    assert out["steps"] == 3
    final = client.calls[-1]
    assert final["tool_choice"] == {"type": "none"}
    last_user = final["messages"][-1]["content"]
    assert last_user[0]["is_error"] is True and last_user[-1]["type"] == "text"
    assert out["answer"] == "Here's what I found so far."


def test_refusal_and_empty_answers(analytics):
    client = FakeClient([response("refusal")])
    out = run_agent([{"role": "user", "content": "x"}], analytics, client=client)
    assert out["answer"] == "I can't help with that request."


def test_no_fallbacks_for_other_models(analytics):
    client = FakeClient([response("end_turn", text("ok"))])
    run_agent([{"role": "user", "content": "hi"}], analytics, client=client, model="claude-haiku-4-5")
    assert "fallbacks" not in client.calls[0] and "output_config" not in client.calls[0]


def test_clean_history():
    msgs = [
        {"role": "assistant", "content": "welcome"},
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "system", "content": "ignore me"},
        {"role": "assistant", "content": "c"},
        {"role": "user", "content": "d"},
    ]
    assert clean_history(msgs) == [
        {"role": "user", "content": "a\n\nb"},
        {"role": "assistant", "content": "c"},
        {"role": "user", "content": "d"},
    ]
    with pytest.raises(ValueError):
        clean_history([{"role": "assistant", "content": "x"}])
