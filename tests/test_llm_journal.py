"""LLM 调用的归属切分。

memory server 和 env server 是所有 run 共用的进程，一条调用属于哪条 run 靠
`request_id` 的 `<run-id>__` 前缀认。这套逻辑因为并行 run 串目录已经重写过一次
（从按文件偏移切改成按前缀切），但从没测过——而 docs/experiments/runs/ 里每个
成本数字都在它下游。
"""

import json

import pytest

from harness import llm_journal, runctx


@pytest.fixture
def journal(tmp_path, monkeypatch):
    d = tmp_path / "_llm"
    d.mkdir()
    monkeypatch.setenv("MARENA_LLM_JOURNAL_DIR", str(d))
    return d


def write(path, *records):
    for r in records:
        llm_journal.append_line(path, json.dumps(r, ensure_ascii=False))


# --------------------------------------------------------------- 前缀规则

def test_belongs_to_needs_the_double_underscore():
    assert runctx.belongs_to("run1__abc", "run1")
    assert not runctx.belongs_to("run1abc", "run1")
    assert not runctx.belongs_to(None, "run1")
    assert not runctx.belongs_to("", "run1")


def test_a_run_id_that_prefixes_another_does_not_steal_its_calls():
    """run1 和 run10 并行时不能互相吃记录。分隔符就是防这个的。"""
    assert not runctx.belongs_to("run10__abc", "run1")
    assert runctx.belongs_to("run10__abc", "run10")


# --------------------------------------------------------------- marks / harvest

def test_harvest_takes_only_this_run_and_only_after_the_mark(journal, tmp_path):
    agent_log = journal / "agent.jsonl"
    write(agent_log, {"request_id": "older__1", "seq": 0})   # 跑之前就有的

    marks = llm_journal.marks()
    assert marks["agent"] > 0

    write(agent_log,
          {"request_id": "myrun__a", "seq": 1},
          {"request_id": "otherrun__b", "seq": 2},          # 别人的
          {"request_id": None, "seq": 3},                   # 没打标签的
          {"request_id": "myrun__c", "seq": 4})

    out = tmp_path / "out"
    out.mkdir()
    counts = llm_journal.harvest(marks, out, "myrun")

    assert counts == {"agent": 2}
    got = [json.loads(l) for l in
           (out / "llm_calls.agent.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["seq"] for r in got] == [1, 4]


def test_untagged_records_stay_in_the_global_file(journal, tmp_path):
    """mem0 的 embedder 经网关留下的记录不归任何 run，不该被谁切走。"""
    agent_log = journal / "agent.jsonl"
    marks = llm_journal.marks()
    write(agent_log, {"request_id": None, "seq": 1})

    out = tmp_path / "out"
    out.mkdir()
    assert llm_journal.harvest(marks, out, "myrun") == {}
    assert not list(out.iterdir())
    assert agent_log.read_text(encoding="utf-8").strip()      # 原文件没被动


def test_harvest_splits_by_role(journal, tmp_path):
    marks = llm_journal.marks()
    write(journal / "agent.jsonl", {"request_id": "r__1"})
    write(journal / "judge.jsonl", {"request_id": "r__2"}, {"request_id": "r__3"})
    write(journal / "memory.jsonl", {"request_id": "other__4"})

    out = tmp_path / "out"
    out.mkdir()
    assert llm_journal.harvest(marks, out, "r") == {"agent": 1, "judge": 2}
    assert (out / "llm_calls.judge.jsonl").exists()
    assert not (out / "llm_calls.memory.jsonl").exists()


def test_a_broken_line_does_not_sink_the_harvest(journal, tmp_path):
    """日志是多进程追加的，截断的一行不该让整条 run 的成本统计消失。"""
    agent_log = journal / "agent.jsonl"
    marks = llm_journal.marks()
    llm_journal.append_line(agent_log, "{不是 json")
    write(agent_log, {"request_id": "r__1"})

    out = tmp_path / "out"
    out.mkdir()
    assert llm_journal.harvest(marks, out, "r") == {"agent": 1}


def test_marks_on_an_empty_dir_is_empty(journal):
    assert llm_journal.marks() == {}
