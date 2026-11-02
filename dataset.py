"""读本地数据副本。

上游 `run_math.py` 走 `load_dataset("ZexueHe/memoryarena", ...)`，在这台机器上
拉不下来（httpx brotli 解码失败，代理层的问题）。`data/memoryarena/` 下的副本
字段和 HF 版本一致，直接读 jsonl，顺带让 run 变成离线可复现的。
"""

import json
from dataclasses import dataclass
from typing import List, Optional

from harness import settings


@dataclass
class Case:
    """一条 case = 数据集里的一行 = 一个 memory session。

    formal reasoning 里一行是一篇 paper，`questions` 是这篇 paper 下按顺序问的
    若干道子题，memory 在子题之间累积。
    """

    id: int
    name: str
    questions: List[str]
    answers: List[str]
    backgrounds: List[str]

    def tasks(self, limit: int = 0):
        """上游 run 循环要的 (question, answer, background) 三元组列表。"""
        triples = list(zip(self.questions, self.answers, self.backgrounds))
        return triples[:limit] if limit else triples


def _name_of(row) -> str:
    """case 的标识。formal reasoning 有 paper_name，其他环境只有 id。"""
    return row.get("paper_name") or f"case_{row['id']}"


def _rows(hf_config: str):
    directory = settings.data_dir() / hf_config
    if not directory.is_dir():
        raise SystemExit(f"没有本地数据副本：{directory}")
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"{directory} 下没有 jsonl")
    for path in files:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def load_case(hf_config: str, index: Optional[int] = None, name: Optional[str] = None) -> Case:
    rows = list(_rows(hf_config))
    if name is not None:
        matches = [r for r in rows if _name_of(r) == name]
        if not matches:
            raise SystemExit(f"{hf_config} 里没有 paper_name={name}")
        row = matches[0]
    else:
        position = index or 0
        if not 0 <= position < len(rows):
            raise SystemExit(f"case 序号 {position} 越界：{hf_config} 有 {len(rows)} 条")
        row = rows[position]
    backgrounds = row["backgrounds"]
    if isinstance(backgrounds, str):
        # travel / shopping 是整个 task 共用一条 background
        backgrounds = [backgrounds] * len(row["questions"])
    return Case(
        id=row["id"],
        name=_name_of(row),
        questions=row["questions"],
        answers=row["answers"],
        backgrounds=backgrounds,
    )


def list_cases(hf_config: str):
    return [(r["id"], _name_of(r), len(r["questions"])) for r in _rows(hf_config)]
