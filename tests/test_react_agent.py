"""ReAct 路径：thought 有没有真的被带起来，以及原路径有没有被碰坏。

一条真 run 要花 API 额度，所以这里注入假 client：`chat_with_tools` 按脚本返回
事先写好的回复，并把收到的 messages 存下来，让测试能断言消息历史长什么样。
"""

import json
from pathlib import Path

import pytest

from env.env_systems.travel_env import _parse_person_plan_from_result

from agent.travel_planner import AgentStep, TravelPlannerAgent
from agent.travel_planner_react import ReActTravelPlannerAgent
from env.env_systems.travel_planner_env.clients.base_client import ModelResponse, ToolCall
from env.env_systems.travel_planner_env.prompts import (
    AGENT_SYSTEM_PROMPT,
    REACT_AGENT_SYSTEM_PROMPT,
)
from harness.trace import scratchpad_to_json

PLAN = "=== Eric's Plan ===\nDay 1:\nCurrent City: from A to B\n"


class FakeClient:
    """按脚本回复，并记下每次收到的 messages。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []          # 每次调用收到的 messages（深拷贝）

    def chat_with_tools(self, messages, tools, **kw):
        self.calls.append(json.loads(json.dumps(messages, default=str)))
        content, tool_names = self.script.pop(0) if self.script else (PLAN, [])
        tool_calls = [ToolCall(id=f"call_{i}", name=n, arguments={"city": "B"})
                      for i, n in enumerate(tool_names)] or None
        return ModelResponse(content=content, tool_calls=tool_calls, raw_response={})

    def format_tool_result(self, tool_call_id, result, name=None):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}

    def format_assistant_tool_calls(self, tool_calls):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.name,
                                         "arguments": json.dumps(tc.arguments)}}
                           for tc in tool_calls],
        }

    def get_usage_stats(self):
        return {"total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0.0}


class FakeExecutor:
    def __init__(self, *a, **kw):
        pass

    def execute(self, tool_name, args):
        return f"{tool_name} result"


@pytest.fixture
def build(monkeypatch):
    """造一个 agent，client 和 ToolExecutor 都是假的（后者真跑会去读 CSV）。"""
    def _build(cls, script, **kw):
        monkeypatch.setattr("agent.travel_planner.ToolExecutor", FakeExecutor)
        client = FakeClient(script)
        monkeypatch.setattr(cls, "_create_client", lambda self, model: client)
        agent = cls(model_name="fake-model", max_steps=5, db_path="/nonexistent", **kw)
        return agent, client
    return _build


def _assistant_turns(client):
    """所有以 assistant 身份写回历史的消息（最后一次调用看到的最全）。"""
    if not client.calls:
        return []
    return [m for m in client.calls[-1] if m.get("role") == "assistant"]


# --------------------------------------------------------------- thought 带起来了吗

def test_react_carries_thought_into_message_history(build):
    """ReAct 的核心改动：模型自己上一轮的推理，下一轮要看得见。

    上游三个 client 的 format_assistant_tool_calls 都把 content 写死成 None。
    原路径无所谓（模型本来不写字），但要求边想边做以后，链条会断在这里。
    """
    agent, client = build(
        ReActTravelPlannerAgent,
        [("Thought: 先查航班，日期和目的地都已知，缺的是航班号。", ["FlightSearch"]),
         (PLAN, [])],
    )
    result = agent.run_single_person(query="q", name="Eric", round_idx=1)

    turns = _assistant_turns(client)
    assert len(turns) == 1
    assert turns[0]["content"].startswith("Thought: 先查航班")
    assert turns[0]["tool_calls"][0]["function"]["name"] == "FlightSearch"

    assert result.scratchpad[0].thought.startswith("Thought: 先查航班")
    assert result.final_plan == PLAN.strip()      # react 切完会 strip
    assert result.success is True


def test_toolcall_path_is_unchanged(build):
    """回归闸：同一份脚本走原路径，assistant 消息里的 content 仍然是 None。

    这条测试的作用是钉住「加 ReAct 没有顺手改掉原路径」——两条路径要能对照，
    原路径的行为就必须逐字不变。
    """
    agent, client = build(
        TravelPlannerAgent,
        [("Thought: 先查航班。", ["FlightSearch"]), (PLAN, [])],
    )
    result = agent.run_single_person(query="q", name="Eric", round_idx=1)

    turns = _assistant_turns(client)
    assert len(turns) == 1
    assert turns[0]["content"] is None          # ← 原样
    assert result.final_plan == PLAN
    # thought 仍然被记进 scratchpad（原路径下模型不写字，所以实际是 None）
    assert result.scratchpad[0].thought == "Thought: 先查航班。"


def test_toolcall_path_leaves_thought_null_when_model_writes_nothing(build):
    """原路径的真实形态：模型只发 tool_calls，thought 是 null。

    这正是加 ReAct 的理由——事后归因时中间那一步是黑的。
    """
    agent, client = build(TravelPlannerAgent, [(None, ["FlightSearch"]), (PLAN, [])])
    result = agent.run_single_person(query="q", name="Eric", round_idx=1)
    assert result.scratchpad[0].thought is None
    assert scratchpad_to_json(result)[0]["thought"] is None


# --------------------------------------------------------------- 没写 Thought 的兜底

def test_nudge_reasks_once_and_keeps_history_clean(build):
    """模型只发 tool_calls 时补问一次；补问本身不留在历史里。"""
    agent, client = build(
        ReActTravelPlannerAgent,
        [(None, ["FlightSearch"]),                    # 第一次：没写 Thought
         ("Thought: 补上了。", ["FlightSearch"]),      # 补问后
         (PLAN, [])],
    )
    result = agent.run_single_person(query="q", name="Eric", round_idx=1)

    assert agent.thought_nudges == 1
    assert agent.thought_missing == 0
    assert result.scratchpad[0].thought == "Thought: 补上了。"

    # 催促那条消息只出现在补问用的副本里，没进最终历史
    nudged = [m for m in client.calls[1] if "没有写" in str(m.get("content", ""))
              or "reasoning" in str(m.get("content", "")).lower()]
    assert nudged, "补问应该带上催促消息"
    final_history = client.calls[-1]
    assert not any("Reply with the Thought" in str(m.get("content", ""))
                   for m in final_history), "催促消息不该留在历史里"


def test_nudge_gives_up_without_crashing(build):
    """补问完还是不写，就记一笔继续跑，不中断这一轮。"""
    agent, client = build(
        ReActTravelPlannerAgent,
        [(None, ["FlightSearch"]), (None, ["FlightSearch"]), (PLAN, [])],
    )
    result = agent.run_single_person(query="q", name="Eric", round_idx=1)

    assert agent.thought_nudges == 1
    assert agent.thought_missing == 1
    assert result.final_plan == PLAN.strip()    # 该跑完还是跑完
    assert result.success is True


def test_require_thought_off_skips_the_nudge(build):
    agent, client = build(
        ReActTravelPlannerAgent,
        [(None, ["FlightSearch"]), (PLAN, [])],
        require_thought=False,
    )
    agent.run_single_person(query="q", name="Eric", round_idx=1)
    assert agent.thought_nudges == 0
    assert len(client.calls) == 2               # 没有多余的一次调用


# --------------------------------------------------------------- prompt 没被换回去

def test_react_prompt_survives_set_base_person(build):
    """set_base_person 原先硬写 AGENT_SYSTEM_PROMPT，会把子类的 prompt 静默换回去。"""
    agent, _ = build(ReActTravelPlannerAgent, [(PLAN, [])])
    assert agent.system_prompt == REACT_AGENT_SYSTEM_PROMPT

    agent.set_base_person("Jennifer", "base query", "base plan")
    assert agent.base_messages[0]["content"] == REACT_AGENT_SYSTEM_PROMPT

    agent.reset()
    assert agent.base_messages[0]["content"] == REACT_AGENT_SYSTEM_PROMPT


def test_toolcall_prompt_unaffected_by_that_fix(build):
    agent, _ = build(TravelPlannerAgent, [(PLAN, [])])
    agent.set_base_person("Jennifer", "base query", "base plan")
    assert agent.base_messages[0]["content"] == AGENT_SYSTEM_PROMPT


def test_both_prompts_state_the_same_output_contract():
    """两条路径必须产出同一种终局文本。

    travel_env._parse_person_plan_from_result 和 run_travel.parse_all_plans 按
    `=== {Name}'s Plan ===` 解析；ReAct 的 prompt 是另抄了一份，这条测试钉住它
    不会单独漂走。
    """
    key = "**CRITICAL: OUTPUT FORMAT"
    tail = "**YOUR FINAL MESSAGE:**"
    assert (AGENT_SYSTEM_PROMPT[AGENT_SYSTEM_PROMPT.index(key):].strip()
            == REACT_AGENT_SYSTEM_PROMPT[
                REACT_AGENT_SYSTEM_PROMPT.index(key):
                REACT_AGENT_SYSTEM_PROMPT.index(tail)].strip())


def test_react_prompt_demands_reasoning_about_memory():
    """注入的记忆用没用上，要模型自己说——这是 trace-diagnose 判归因的依据。"""
    assert "Thought:" in REACT_AGENT_SYSTEM_PROMPT
    assert "memory context" in REACT_AGENT_SYSTEM_PROMPT
    assert "Thought:" not in AGENT_SYSTEM_PROMPT


# --------------------------------------------------------------- use_step_memory 漏记

class FakeMemory:
    def wrap_user_prompt(self, query):
        return "<memory_context>\n[Context 1]\nContent: X\n</memory_context>\nUser Prompt: " + query


@pytest.mark.parametrize("cls", [TravelPlannerAgent, ReActTravelPlannerAgent])
def test_step_memory_lands_in_the_trajectory(build, cls):
    """循环内注入的记忆合并后没人记。挂在步上，不发新事件。

    发 memory.wrap_user_prompt 事件会污染 round 计数：trace_prep 的一轮就是按那个
    事件开的，多发一次就多算一轮，错的 pass_rate 会一路流进 docs/experiments/runs/。
    """
    agent, _ = build(cls, [("t", ["FlightSearch"]), (PLAN, [])])
    result = agent.run_single_person(
        query="q", name="Eric", round_idx=1, memory_system=FakeMemory())

    assert "[Context 1]" in result.scratchpad[0].step_memory
    assert "[Context 1]" in scratchpad_to_json(result)[0]["step_memory"]


def test_step_memory_absent_keeps_the_old_shape(build):
    """没开 use_step_memory 时，落盘形状跟以前逐字一样。"""
    agent, _ = build(TravelPlannerAgent, [("t", ["FlightSearch"]), (PLAN, [])])
    result = agent.run_single_person(query="q", name="Eric", round_idx=1)
    assert set(scratchpad_to_json(result)[0]) == {
        "step_idx", "thought", "tool_calls", "tool_results", "final_output"}


# --------------------------------------------------------------- 两条路径同形

def test_react_result_has_the_same_shape_as_toolcall(build):
    """run_travel.py 对 agent 的要求两条路径都得满足，否则换范式会在循环里炸。"""
    surface = ["reset", "set_base_person", "prepare_for_person", "act", "last_result",
               "get_scratchpad_dict", "build_memory_entry", "add_judge_feedback",
               "get_usage_stats"]
    for name in surface:
        assert hasattr(ReActTravelPlannerAgent, name), name

    agent, _ = build(ReActTravelPlannerAgent, [("t", ["FlightSearch"]), (PLAN, [])])
    agent.prepare_for_person(name="Eric", round_idx=1)
    action = agent.act("q")
    assert action == PLAN.strip()
    assert isinstance(action, str)                       # env.step 按文本解析
    entry = json.loads(agent.build_memory_entry(task="q", action=action))
    assert set(entry) >= {"name", "query", "scratchpad", "final_plan"}
    assert agent.get_usage_stats()["agent_mode"] == "react"


def test_max_steps_exhaustion_reports_failure(build):
    agent, _ = build(ReActTravelPlannerAgent, [("t", ["FlightSearch"])] * 10)
    result = agent.run_single_person(query="q", name="Eric", round_idx=1)
    assert result.success is False
    assert "Max steps" in result.error_message
    assert result.total_steps == 5


# --------------------------------------------------------------- 路径分发

def test_agent_mode_defaults_to_toolcall_when_config_says_nothing():
    """现有 4 个 config 都没有 agent_mode，它们的行为必须一个字节都不变。"""
    import run_travel
    args = run_travel.build_args({"agent": {"model_name": "gpt-5-mini"},
                                  "memory": {}, "env": {}, "task_specific": {}, "output": {}})
    assert args.agent_mode == "toolcall"
    assert run_travel.AGENT_MODES[args.agent_mode] is TravelPlannerAgent


def test_agent_mode_from_config_selects_the_react_class():
    import run_travel
    args = run_travel.build_args({"agent": {"model_name": "m", "agent_mode": "react"},
                                  "memory": {}, "env": {}, "task_specific": {}, "output": {}})
    assert run_travel.AGENT_MODES[args.agent_mode] is ReActTravelPlannerAgent


def test_every_checked_in_travel_config_still_selects_toolcall():
    import json
    from pathlib import Path
    import run_travel
    repo = Path(__file__).resolve().parents[1]
    files = sorted((repo / "harness_configs" / "travel_planner").glob("*.json"))
    assert files
    for f in files:
        cfg = json.loads(f.read_text(encoding="utf-8"))
        mode = cfg.get("agent", {}).get("agent_mode", "toolcall")
        assert run_travel.AGENT_MODES[mode] is TravelPlannerAgent, f


# --------------------------------------------------------------- 终局消息的切分

FIXTURE = (Path(__file__).parent / "fixtures" / "react_final_message.txt").read_text(
    encoding="utf-8")


def _split(agent, content, name="Eric"):
    return agent._split_final_message(content, name)


def test_the_real_failure_is_reproduced_by_the_fixture():
    """这份 fixture 是从 2026-09-18 那条真 run 的 agent.jsonl 里取出来的。

    模型在推理里复述了格式要求，把 `=== Eric's Plan ===` 顺手写进了句子中间；
    env 侧 _parse_person_plan_from_result 用 re.search 取第一个匹配、切到下一个
    `===`，于是判分落在两个标记之间那段议论文上，解出 0 天 = 18 格全错。
    """
    assert FIXTURE.count("=== Eric's Plan ===") == 2
    assert len(_parse_person_plan_from_result(FIXTURE, "Eric")) == 0


def test_react_split_hands_the_env_a_clean_plan(build):
    agent, _ = build(ReActTravelPlannerAgent, [(FIXTURE, [])])
    preamble, plan = _split(agent, FIXTURE)

    assert plan.startswith("=== Eric's Plan ===")
    assert "Thought:" not in plan
    assert preamble.startswith("Thought:")
    # 真正判分的那个解析器现在拿得到三天
    assert len(_parse_person_plan_from_result(plan, "Eric")) == 3


def test_react_scans_backwards_past_a_quotation_in_the_thought(build):
    """倒着找：thought 里复述的标记在前，真计划在后。正着找会切在引用上。"""
    agent, _ = build(ReActTravelPlannerAgent, [("x", [])])
    content = ('Thought: must start with "=== Eric\'s Plan ===" exactly.\n'
               "=== Eric's Plan ===\n"
               "Day 1:\nCurrent City: A\n")
    preamble, plan = _split(agent, content)
    assert plan.startswith("=== Eric's Plan ===\nDay 1:")
    assert preamble.startswith("Thought: must start")


def test_react_skips_a_quotation_that_comes_after_the_plan(build):
    """光倒着找不够：模型可能在计划后面再补一句复述。

    判据是标记后面跟不跟 `Day N:`——真计划跟，只是提到标记的不跟。
    """
    agent, _ = build(ReActTravelPlannerAgent, [("x", [])])
    content = ("=== Eric's Plan ===\n"
               "Day 1:\nCurrent City: A\n\n"
               'Note: I started with "=== Eric\'s Plan ===" as required.')
    preamble, plan = _split(agent, content)
    assert plan.startswith("=== Eric's Plan ===\nDay 1:")
    assert preamble is None


def test_react_finds_the_plan_even_when_its_marker_is_not_at_a_line_start(build):
    """这条是"行首优先"那版会切错的：引用在行首，真计划在行中。

    行首只是排版习惯，`Day N:` 才是计划的实际结构，所以判据用后者。
    """
    agent, _ = build(ReActTravelPlannerAgent, [("x", [])])
    content = ("Thought: the format is\n"
               "=== Eric's Plan ===\n"
               "followed by the days. Here it is: === Eric's Plan ===\n"
               "Day 1:\nCurrent City: A\n")
    preamble, plan = _split(agent, content)
    assert plan.startswith("=== Eric's Plan ===\nDay 1:")
    assert "followed by the days" in preamble
    # 真正判分的解析器拿得到内容
    assert len(_parse_person_plan_from_result(plan, "Eric")) == 1


def test_react_falls_back_to_the_last_marker_when_none_looks_like_a_plan(build):
    """都不像计划（模型根本没写 Day）：退回最后一个，至少不切在更早的引用上。"""
    agent, _ = build(ReActTravelPlannerAgent, [("x", [])])
    content = ('Thought: start with "=== Eric\'s Plan ===".\n'
               "=== Eric's Plan ===\nI could not find flights.\n")
    preamble, plan = _split(agent, content)
    assert plan == "=== Eric's Plan ===\nI could not find flights."
    assert preamble.startswith("Thought: start with")


def test_react_split_when_the_model_complied(build):
    """模型真的只输出计划时，preamble 是 None，thought 不被改写。"""
    agent, _ = build(ReActTravelPlannerAgent, [("x", [])])
    preamble, plan = _split(agent, PLAN)
    assert preamble is None
    assert plan == PLAN.strip()


def test_react_split_falls_back_when_the_name_is_wrong(build):
    agent, _ = build(ReActTravelPlannerAgent, [("x", [])])
    content = "Thought: 写岔了名字。\n=== Erik's Plan ===\nDay 1:\nCurrent City: A\n"
    preamble, plan = _split(agent, content, name="Eric")
    assert plan.startswith("=== Erik's Plan ===")
    assert preamble == "Thought: 写岔了名字。"


def test_react_split_leaves_unformatted_output_alone(build):
    """一个标记都没有：不发明行为，原样交下游，失败形态和原路径一致。"""
    agent, _ = build(ReActTravelPlannerAgent, [("x", [])])
    junk = "模型没按格式输出，写了一段散文。"
    assert _split(agent, junk) == (None, junk)
    assert _split(agent, "") == (None, "")


def test_react_loop_end_to_end_yields_a_scorable_plan(build):
    """整条循环跑下来，run_travel 交给 env.step 的就是干净计划。"""
    agent, _ = build(ReActTravelPlannerAgent,
                     [("Thought: 先查。", ["FlightSearch"]), (FIXTURE, [])])
    result = agent.run_single_person(query="q", name="Eric", round_idx=1)

    assert result.final_plan.startswith("=== Eric's Plan ===")
    assert len(_parse_person_plan_from_result(result.final_plan, "Eric")) == 3
    # 推理没被丢掉，挪到了 thought 上
    assert result.scratchpad[-1].thought.startswith("Thought: AccommodationSearch")
    assert result.scratchpad[-1].final_output.startswith("=== Eric's Plan ===")
    # accumulated_plans 不再被上万字推理撑爆——它会进后面每一轮的 history
    assert "Thought:" not in agent.accumulated_plans


def test_toolcall_final_message_is_untouched(build):
    """默认钩子返回 (None, content)，原路径落盘形状逐字不变。"""
    agent, _ = build(TravelPlannerAgent, [(FIXTURE, [])])
    result = agent.run_single_person(query="q", name="Eric", round_idx=1)
    assert result.final_plan == FIXTURE          # 一个字都没动
    assert result.scratchpad[-1].thought == FIXTURE
    assert result.scratchpad[-1].final_output == FIXTURE


def test_react_prompt_no_longer_forbids_thinking_in_the_final_message():
    """不要求模型压抑刚被要求做的事——它不照办，而且最后一步的推理最该留下。"""
    assert "carries no" not in REACT_AGENT_SYSTEM_PROMPT
    assert "beginning of its own line" in REACT_AGENT_SYSTEM_PROMPT


def test_toolcall_keeps_final_output_none_when_the_model_says_nothing(build):
    """默认钩子必须逐字保住原来的取值，包括 None 和 "" 的区别。

    原代码是 `step.final_output = response.content` / `final_plan = content or ""`。
    加钩子的时候差点把 None 变成 ""，那会改掉落盘的 scratchpad。
    """
    agent, _ = build(TravelPlannerAgent, [(None, [])])
    result = agent.run_single_person(query="q", name="Eric", round_idx=1)
    assert result.scratchpad[-1].final_output is None
    assert result.final_plan == ""
