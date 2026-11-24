"""A reply without tool calls is the final plan only when it is usable.

Qwen's thinking sometimes runs into the 32768-token output cap: finish_reason "length", empty content, no tool call.
Upstream took that as the final plan, so the person got an empty plan (11 of 105 subtasks in batch 1102).
"""

import json

from agent.travel_planner import TravelPlannerAgent
from env.env_systems.travel_planner_env.clients.base_client import ModelResponse, ToolCall
from harness.trace import scratchpad_to_json
from tests.test_react_agent import PLAN, FakeClient, FakeExecutor


class FinishReasonClient(FakeClient):
    """Script entries are (content, tool_names, finish_reason)."""

    def chat_with_tools(self, messages, tools, **kw):
        self.calls.append(json.loads(json.dumps(messages, default=str)))
        content, tool_names, finish = self.script.pop(0)
        tool_calls = [ToolCall(id=f"call_{i}", name=n, arguments={"city": "B"})
                      for i, n in enumerate(tool_names)] or None
        raw = {"choices": [{"finish_reason": finish}]}
        return ModelResponse(content=content, tool_calls=tool_calls, raw_response=raw)


def _agent(monkeypatch, script, max_steps=5):
    monkeypatch.setattr("agent.travel_planner.ToolExecutor", FakeExecutor)
    client = FinishReasonClient(script)
    monkeypatch.setattr(TravelPlannerAgent, "_create_client", lambda self, model: client)
    return TravelPlannerAgent(model_name="fake-model", max_steps=max_steps, db_path="/nonexistent"), client


def test_truncated_and_empty_replies_are_asked_again(monkeypatch):
    agent, client = _agent(monkeypatch, [
        (None, ["RestaurantSearch"], "tool_calls"),
        ("", [], "length"),                      # thinking hit the output cap
        ("=== Eric's Pl", [], "length"),         # cut off mid-plan
        (None, [], "stop"),                      # empty
        (PLAN, [], "stop"),
    ])
    result = agent.run_single_person(query="q", name="Eric", round_idx=1)

    assert result.final_plan == PLAN
    assert result.total_steps == 5
    assert [s.discarded for s in result.scratchpad] == [None, "truncated", "truncated", "empty", None]
    # each retry sends the same messages again: nothing from a discarded reply enters the history
    assert client.calls[1] == client.calls[2] == client.calls[3] == client.calls[4]
    assert agent.get_usage_stats()["discarded_replies"] == 3
    assert [e.get("discarded") for e in scratchpad_to_json(result)] == [None, "truncated", "truncated", "empty", None]
    entry = json.loads(agent.build_memory_entry("q", result.final_plan))
    assert [s["step_idx"] for s in entry["scratchpad"]] == [0, 4]


def test_discarded_replies_count_toward_max_steps(monkeypatch):
    agent, _ = _agent(monkeypatch, [("", [], "length")] * 3, max_steps=3)
    result = agent.run_single_person(query="q", name="Eric", round_idx=1)

    assert result.final_plan == ""
    assert result.success is False
    assert result.total_steps == 3


def test_plan_with_stop_is_final_on_first_reply(monkeypatch):
    agent, client = _agent(monkeypatch, [(PLAN, [], "stop")])
    result = agent.run_single_person(query="q", name="Eric", round_idx=1)

    assert result.final_plan == PLAN
    assert len(client.calls) == 1
    assert agent.get_usage_stats()["discarded_replies"] == 0
