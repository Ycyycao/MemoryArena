"""export_travel_brief 的按轮分组。

原先是 `zip(wraps, acts, steps)`——三个各自过滤出来的列表按位置配对。两个失败
模式都是静默的：memory=none 的 run 根本不发 memory.wrap_user_prompt，zip 直接
truncate 成 0 轮；任何一类事件数量变了就错位，拿第 N 轮的计划对第 M 轮的标准
答案，产出一份格式完好、内容全错的 brief，不报任何错。
"""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    """tool_script/ 不是包，按路径加载。"""
    path = REPO / "tool_script" / "export_travel_brief.py"
    spec = importlib.util.spec_from_file_location("export_travel_brief", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


brief = _load()


def ev(name, **fields):
    return {"t": 0.0, "event": name, **fields}


def test_pairs_one_round_at_a_time():
    trace = [
        ev("env.reset"),
        ev("memory.wrap_user_prompt", question="q1"),
        ev("agent.act", person="Eric"),
        ev("env.step", ground_truth={"name": "Eric"}),
        ev("memory.add"),
        ev("memory.wrap_user_prompt", question="q2"),
        ev("agent.act", person="Mary"),
        ev("env.step", ground_truth={"name": "Mary"}),
    ]
    rounds = brief.group_rounds(trace)
    assert len(rounds) == 2
    assert rounds[0]["wrap"]["question"] == "q1"
    assert rounds[0]["act"]["person"] == "Eric"
    assert rounds[1]["step"]["ground_truth"]["name"] == "Mary"


def test_a_run_without_a_memory_system_still_has_rounds():
    """memory=none 的 baseline 一条 wrap 都不发。旧的 zip 在这里返回 0 轮。"""
    trace = [
        ev("env.reset"),
        ev("agent.act", person="Eric"),
        ev("env.step", ground_truth={"name": "Eric"}),
        ev("agent.act", person="Mary"),
        ev("env.step", ground_truth={"name": "Mary"}),
    ]
    rounds = brief.group_rounds(trace)
    assert len(rounds) == 2
    assert rounds[0].get("wrap") is None
    assert rounds[1]["act"]["person"] == "Mary"


def test_rounds_never_cross_even_if_an_event_repeats():
    """一轮里多发一个 agent.act 时，轮的边界还是 env.step，不会错位到下一轮。"""
    trace = [
        ev("memory.wrap_user_prompt", question="q1"),
        ev("agent.act", person="Eric-1"),
        ev("agent.act", person="Eric-2"),
        ev("env.step", ground_truth={"name": "Eric"}),
        ev("memory.wrap_user_prompt", question="q2"),
        ev("agent.act", person="Mary"),
        ev("env.step", ground_truth={"name": "Mary"}),
    ]
    rounds = brief.group_rounds(trace)
    assert len(rounds) == 2
    assert rounds[0]["act"]["person"] == "Eric-2"          # 终局计划在最后一步
    assert rounds[1]["wrap"]["question"] == "q2"           # 没被前一轮吃掉


def test_a_run_that_died_before_env_step_keeps_its_last_round():
    trace = [
        ev("memory.wrap_user_prompt", question="q1"),
        ev("agent.act", person="Eric"),
        ev("env.step", ground_truth={"name": "Eric"}),
        ev("memory.wrap_user_prompt", question="q2"),
        ev("agent.act", person="Mary"),
    ]
    rounds = brief.group_rounds(trace)
    assert len(rounds) == 2
    assert "step" not in rounds[1]


def test_empty_trace():
    assert brief.group_rounds([]) == []


@pytest.mark.parametrize(
    "run_dir",
    sorted(p.parent for p in (REPO / "results").rglob("trajectory.json")
           if "travel" in p.parent.name),
    ids=lambda p: p.name,
)
def test_real_runs_group_into_as_many_rounds_as_they_have_steps(run_dir):
    import json
    trace = json.loads((run_dir / "trajectory.json").read_text(encoding="utf-8"))["trace"]
    steps = sum(1 for e in trace if e["event"] == "env.step")
    assert len(brief.group_rounds(trace)) == steps
