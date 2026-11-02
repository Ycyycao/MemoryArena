"""travel 的文本解析层。

`parse_all_plans` 把 agent 输出的自由文本切成每个人的计划，`format_person_plan`
把标准答案渲染成同一种文本。仓库里每一个 travel 数字都在这两个函数下游，之前
一个测试都没有。ReAct 换了 prompt，这两个函数是"终局文本格式没漂"的闸门。
"""

import pytest

from run_travel import format_person_plan, parse_all_plans

PLAN = """=== Eric's Plan ===
Day 1:
Current City: from A to B
Transportation: Flight Number: F1, from A to B, Departure Time: 10:00, Arrival Time: 12:00

=== Mary's Plan ===
Day 1:
Current City: from C to D
Transportation: -
"""


def test_splits_by_name():
    out = parse_all_plans(PLAN, [{"name": "Eric", "query": "q1"},
                                 {"name": "Mary", "query": "q2"}])
    assert [r["name"] for r in out] == ["Eric", "Mary"]
    assert out[0]["result"].startswith("Day 1:")
    assert "Flight Number: F1" in out[0]["result"]
    assert "from C to D" in out[1]["result"]
    assert [r["person_idx"] for r in out] == [1, 2]


def test_missing_person_gets_empty_string_not_an_error():
    """agent 少写一个人时，那一格是空串——env 会判全错，但不该在这里崩。"""
    out = parse_all_plans(PLAN, [{"name": "Nobody", "query": "q"}])
    assert out[0]["result"] == ""


def test_name_defaults_when_the_query_has_none():
    out = parse_all_plans(PLAN, [{"query": "q"}])
    assert out[0]["name"] == "Person1"


def test_text_with_no_plan_marker_yields_nothing():
    out = parse_all_plans("模型没按格式输出，写了一段散文。", [{"name": "Eric", "query": "q"}])
    assert out[0]["result"] == ""


def test_react_style_preamble_before_the_plan_is_dropped():
    """ReAct 路径下模型可能在计划前面留一句话。切分按 === 标记走，前缀不进结果。"""
    out = parse_all_plans("Thought: 查完了，写计划。\n\n" + PLAN,
                          [{"name": "Eric", "query": "q"}])
    assert out[0]["result"].startswith("Day 1:")
    assert "Thought" not in out[0]["result"]


def test_round_trips_with_format_person_plan():
    """标准答案渲染出来的文本，解析器要能读回去。"""
    days = [{"days": 1, "current_city": "from A to B", "transportation": "Flight F1",
             "breakfast": "R1, B", "attraction": "A1, B", "lunch": "R2, B",
             "dinner": "R3, B", "accommodation": "H1, B"}]
    text = format_person_plan("Eric", days)
    assert text.startswith("=== Eric's Plan ===")
    out = parse_all_plans(text, [{"name": "Eric", "query": "q"}])
    assert "Current City: from A to B" in out[0]["result"]
    assert "Accommodation: H1, B" in out[0]["result"]


def test_format_accepts_day_or_days_key():
    assert "Day 2:" in format_person_plan("E", [{"day": 2}])
    assert "Day 2:" in format_person_plan("E", [{"days": 2}])


def test_format_fills_missing_slots_with_dash():
    text = format_person_plan("E", [{"days": 1}])
    assert text.count(": -") == 7      # 六个 slot + current_city
