"""`run_travel.run_group` 在同一进程里跑：LocalEnvironmentClient 包真的 TravelPlannerEnvironment，
memory system 是个有 add / wrap_user_prompt 的对象。DynMem 的 core/run_official.py 就这么用。

零 API 开销：agent 的 client 和 ToolExecutor 都是假的，数据是 tmp_path 里的一组。
"""

import argparse
import json

import pytest

import run_travel
from agent.travel_planner import TravelPlannerAgent
from env.env_systems.travel_env import TravelPlannerEnvironment
from env.env_systems.travel_planner_env.clients.base_client import ModelResponse
from env.local_client import LocalEnvironmentClient
from harness.trace import Recorder

DAY = {"days": 1, "current_city": "from A to B", "transportation": "-", "breakfast": "-", "attraction": "-",
       "lunch": "-", "dinner": "-", "accommodation": "-"}
GROUP = {"id": 3, "base_person": {"name": "Alice", "query": "I am Alice.\nGo.", "daily_plans": [DAY]},
         "questions": ["I am Bob.\nJoin.", "I am Carol.\nJoin."], "answers": [[DAY], [DAY]]}


def plan(name):
    return f"=== {name}'s Plan ===\nDay 1:\nCurrent City: from A to B\nTransportation: -\nBreakfast: -\n" \
           "Attraction: -\nLunch: -\nDinner: -\nAccommodation: -"


class Client:
    def __init__(self):
        self.calls = []

    def chat_with_tools(self, messages, tools, **kw):
        self.calls.append(messages)
        name = messages[-1]["content"].split("Create a travel plan for ")[1].split(".")[0]
        return ModelResponse(plan(name), None, {})

    def get_usage_stats(self):
        return {}


class Memory:
    def __init__(self):
        self.chunks = []

    def add(self, chunk):
        self.chunks.append(json.loads(chunk))

    def wrap_user_prompt(self, q):
        return f"<memory_context>\n{len(self.chunks)} chunks\n</memory_context>\nUser: {q}"


@pytest.fixture
def env(tmp_path, monkeypatch):
    (tmp_path / "group_travel_planner").mkdir()
    (tmp_path / "group_travel_planner" / "data.jsonl").write_text(json.dumps(GROUP) + "\n")
    monkeypatch.setenv("MEMORYARENA_DATA", str(tmp_path))
    return LocalEnvironmentClient(TravelPlannerEnvironment(config={"judgement_mode": "hint"}))


def args(memory_system):
    return argparse.Namespace(model_name="m", memory_system=memory_system, judgement_mode="hint", max_steps=3,
                              use_step_memory=False)


def test_local_client_returns_what_the_env_server_returns(env):
    obs = env.reset(seed=3)
    assert obs["group_id"] == 3 and len(obs["questions"]) == 2
    out = env.step(plan("Bob"), ground_truth={"name": "Bob", "daily_plans": [DAY], "judgement_mode": "hint"},
                   need_judge=True)
    assert set(out) == {"observation", "reward", "done", "info"}
    assert out["reward"] is True and out["done"] is False and out["info"]["judgement"].startswith("Feedback for Bob")


def test_run_group_with_an_in_process_memory(env):
    client, memory = Client(), Memory()
    agent = TravelPlannerAgent(model_name="m", max_steps=3, client=client, executor=object())
    rec = Recorder()
    out = run_travel.run_group(args("mf_x"), agent, env, rec, 3, memory)
    assert [c.get("is_base_person") for c in memory.chunks] == [True, None, None]
    assert [c["name"] for c in memory.chunks[1:]] == ["Bob", "Carol"]
    assert "<memory_context>\n2 chunks" in client.calls[1][1]["content"]   # Carol's round sees base + Bob
    assert [p["name"] for p in out["m_sole-planning_results"]] == ["Bob", "Carol"]
    assert [e["result"]["reward"] for e in rec.events if e["event"] == "env.step"] == [True, True]


def test_run_group_without_memory_puts_the_base_plan_in_the_prompt(env):
    client = Client()
    agent = TravelPlannerAgent(model_name="m", max_steps=3, client=client, executor=object())
    run_travel.run_group(args("none"), agent, env, Recorder(), 3, None)
    assert "Alice's Confirmed Plan" in client.calls[0][1]["content"]


def test_mf_names_are_accepted_on_the_command_line():
    assert run_travel.memory_system_arg("mf_dynmem") == "mf_dynmem"
    assert run_travel.memory_system_arg("simplemem") == "simplemem"
    with pytest.raises(argparse.ArgumentTypeError):
        run_travel.memory_system_arg("mf_")
