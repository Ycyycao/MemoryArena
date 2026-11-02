"""轨迹格式契约。

`harness/trace.py` 的 docstring 写着「加事件可以，改已有事件的字段集不行」——
四个下游工具（export_trajectory / export_travel_brief / export_agentdebug /
trace-prep skill）按这张表读。那句话此前没有任何东西在执行它，这个文件执行它。

审计还找出第二条同样硬的约束，之前没写在任何地方：**每轮的事件基数也不能变**。
export_travel_brief 曾经按位置 zip 三个事件列表，trace_prep 的一轮由
memory.wrap_user_prompt 开、env.step 关。一轮多发一个 agent.act，两边都会静默
错位——不报错，产出一份格式完好、内容全错的报告。所以 ReAct 的多步推理嵌进
scratchpad，不发新事件；下面 test_one_act_per_env_step 钉住这一点。
"""

import json
from pathlib import Path

import pytest

from harness.trace import Recorder, scratchpad_to_json

REPO = Path(__file__).resolve().parents[1]

TOP_LEVEL_KEYS = {"config_path", "config", "case", "num_queries", "trace"}

# 事件名 -> 允许的字段集（不含信封的 t / event）。一个事件可以有多种形态：
# agent.act 在 travel 和 math 两条 loop 下字段完全不同，这是已知的分歧，
# 见 .agents/skills/trace-prep/references/trajectory-schema.md。
EVENT_FIELDS = {
    "env.reset": [
        {"group_id", "base_person", "num_people"},            # travel
        {"query_id", "query", "num_subqueries"},              # search
    ],
    "memory.wrap_user_prompt": [{"question", "wrapped_prompt"}],
    "agent.act": [
        {"person", "query", "final_plan", "total_steps", "success",
         "error_message", "scratchpad"},                      # travel
        {"action"},                                           # math
        {"subquery_idx", "query", "answer", "tool_call_counts",
         "retrieved_docids"},                                 # search
    ],
    "env.step": [{"ground_truth", "result"}],
    "memory.add": [{"chunk"}],
    # 单轮失败。失败的轮没有产出，所以不发 agent.act / env.step——伪造一条等于
    # 往结果里写一轮不存在的判分。"加事件可以"是 trace.py 明写的。
    "round.error": [{"round_idx", "person", "error"}],
    # 由已删除的 marena.llm_trace 补丁产生，只在旧轨迹里；现在同样的信息由
    # llm_calls.<role>.jsonl 旁路记录，export_trajectory 读的时候按时间戳合进来。
    "llm.call": [{"request", "response", "duration"}],
}

SCRATCHPAD_FIELDS = {"step_idx", "thought", "tool_calls", "tool_results", "final_output"}
SCRATCHPAD_OPTIONAL = {"step_memory"}


# --------------------------------------------------------------- 信封

def test_recorder_envelope():
    rec = Recorder()
    rec.record("agent.act", person="Eric", query="q")
    (event,) = rec.events
    assert set(event) == {"t", "event", "person", "query"}
    assert event["event"] == "agent.act"
    assert isinstance(event["t"], float)


def test_recorder_count_only_matches_that_event():
    rec = Recorder()
    rec.record("agent.act")
    rec.record("env.step")
    rec.record("agent.act")
    assert rec.count("agent.act") == 2
    assert rec.count("nope") == 0


def test_recorded_field_names_match_the_table():
    """按五个事件各发一条，字段集必须落在表里。

    这是把 run_travel.py 的调用点和这张表对齐的地方：改了调用点而没改表（或者
    反过来），这条测试会红。
    """
    rec = Recorder()
    rec.record("env.reset", group_id=1, base_person={}, num_people=8)
    rec.record("memory.wrap_user_prompt", question="q", wrapped_prompt="w")
    rec.record("agent.act", person="Eric", query="q", final_plan="p", total_steps=2,
               success=True, error_message=None, scratchpad=[])
    rec.record("env.step", ground_truth={}, result={})
    rec.record("memory.add", chunk="c")
    rec.record("round.error", round_idx=3, person="Zoey", error="BadRequestError: 400")
    for event in rec.events:
        fields = set(event) - {"t", "event"}
        assert fields in EVENT_FIELDS[event["event"]], (event["event"], fields)


def test_search_events_match_the_table():
    """search 那条路的事件在 env server 里记（browsecomp_plus_env.run_full），
    跟着 `run_sequential` 的响应回到 run_search.py 落盘。字段集在这里对齐。
    """
    rec = Recorder()
    rec.record("env.reset", query_id="454", query="q", num_subqueries=9)
    rec.record("memory.wrap_user_prompt", question="sub1", wrapped_prompt="w")
    rec.record("agent.act", subquery_idx=1, query="sub1", answer="a",
               tool_call_counts={"search": 3}, retrieved_docids=["18639"])
    rec.record("env.step", ground_truth="gold", result={"answer": "a", "judged": False})
    rec.record("memory.add", chunk="c")
    for event in rec.events:
        fields = set(event) - {"t", "event"}
        assert fields in EVENT_FIELDS[event["event"]], (event["event"], fields)


def test_search_keeps_one_act_per_env_step():
    """每条 subquery 一轮，最后的合并查询也是一轮：act 和 step 必须一一对应。

    subquery 那一层没有判分，但仍然发 env.step（judged=False），不然轮次配对会塌。
    """
    rec = Recorder()
    for i in range(3):                      # 两条 subquery + 一次最终查询
        rec.record("memory.wrap_user_prompt", question="q", wrapped_prompt="w")
        rec.record("agent.act", subquery_idx=i or None, query="q", answer="a",
                   tool_call_counts={}, retrieved_docids=[])
        rec.record("env.step", ground_truth="g", result={"answer": "a", "judged": i == 2})
    assert rec.count("agent.act") == rec.count("env.step")
    assert rec.count("memory.wrap_user_prompt") == rec.count("agent.act")


# --------------------------------------------------------------- scratchpad

def test_scratchpad_shape():
    class Step:
        step_idx, thought, tool_calls, tool_results, final_output = 0, "t", None, None, "f"
        step_memory = None

    class Result:
        scratchpad = [Step()]

    (entry,) = scratchpad_to_json(Result())
    assert set(entry) == SCRATCHPAD_FIELDS


def test_scratchpad_tolerates_a_result_without_one():
    assert scratchpad_to_json(object()) == []
    assert scratchpad_to_json(None) == []


# --------------------------------------------------------------- 落盘的真轨迹

def _trajectories():
    return sorted((REPO / "results").rglob("trajectory.json"))


def test_there_are_trajectories_to_check():
    """没有真轨迹的话下面几条会空跑通过，说一声。"""
    if not _trajectories():
        pytest.skip("results/ 下没有 trajectory.json")


@pytest.mark.parametrize("path", _trajectories(), ids=lambda p: p.parent.name)
def test_on_disk_trajectory_matches_the_contract(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == TOP_LEVEL_KEYS, path

    for event in data["trace"]:
        name = event["event"]
        assert name in EVENT_FIELDS, f"{path}: 没见过的事件 {name}"
        fields = set(event) - {"t", "event"}
        assert fields in EVENT_FIELDS[name], f"{path}: {name} 字段集 {fields}"

        for step in event.get("scratchpad") or []:
            extra = set(step) - SCRATCHPAD_FIELDS - SCRATCHPAD_OPTIONAL
            assert not extra, f"{path}: scratchpad 多了 {extra}"


@pytest.mark.parametrize("path", _trajectories(), ids=lambda p: p.parent.name)
def test_one_act_per_env_step(path):
    """每轮的事件基数：一个 agent.act 配一个 env.step。

    下游按这个配对。ReAct 把多步推理放进 scratchpad 正是为了守住它。
    """
    trace = json.loads(path.read_text(encoding="utf-8"))["trace"]
    acts = sum(1 for e in trace if e["event"] == "agent.act")
    steps = sum(1 for e in trace if e["event"] == "env.step")
    assert acts == steps, f"{path}: agent.act {acts} 个 / env.step {steps} 个"


@pytest.mark.parametrize("path", _trajectories(), ids=lambda p: p.parent.name)
def test_wrap_never_outnumbers_act(path):
    """trace_prep 按 memory.wrap_user_prompt 开一轮。它多于 agent.act 就是多算轮了。

    memory=none 的 run 一条 wrap 都不发，所以只查上界。多出来的 wrap 必须有解释：
    每条 `round.error` 顶一个（那一轮失败了，没有产出），另外允许 1 个——run 崩在
    某一轮半路时，那轮的 wrap 已经发了、`agent.act` 还没有，`finally` 就落盘了。
    `travel-simplemem-case0-8p-react2` 就是这种（第 6 轮一个 400 掀掉了整条）。
    """
    trace = json.loads(path.read_text(encoding="utf-8"))["trace"]
    wraps = sum(1 for e in trace if e["event"] == "memory.wrap_user_prompt")
    acts = sum(1 for e in trace if e["event"] == "agent.act")
    errs = sum(1 for e in trace if e["event"] == "round.error")
    assert wraps <= acts + errs + 1, f"{path}: wrap {wraps} > act {acts} + error {errs} + 1"
