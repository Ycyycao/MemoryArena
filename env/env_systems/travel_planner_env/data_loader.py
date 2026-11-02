"""group_travel_planner 数据的加载和行转换。

本地副本优先，HuggingFace 兜底。上游原本只走 HF，但那在部分机器上拉不下来
（`httpx.DecodingError: brotli`）；`$MEMORYARENA_DATA/group_travel_planner/` 下的
jsonl 副本字段和 HF 版一致，走同一个 `_convert_row`，结构分毫不差。
"""

import json
import os
import re
from pathlib import Path

DATASET_NAME = "ZexueHe/memoryarena"
DATASET_CONFIG = "group_travel_planner"


def _extract_name(query_str):
    m = re.match(r"I am (\w+)\.", query_str)
    return m.group(1) if m else "Person"


def _convert_row(row):
    """Convert a HuggingFace dataset row to the format expected by the codebase."""
    questions = []
    for i, query_str in enumerate(row["questions"], start=1):
        questions.append({
            "round_idx": i,
            "name": _extract_name(query_str),
            "query": query_str,
        })

    answers = []
    for i, daily_plans in enumerate(row["answers"], start=1):
        answers.append({
            "round_idx": i,
            "daily_plans": [dict(d) for d in daily_plans],
        })

    base_person = row["base_person"]
    return {
        "id": row["id"],
        "base_person": {
            "name": base_person["name"],
            "query": base_person["query"],
            "daily_plans": [dict(d) for d in base_person["daily_plans"]],
        },
        "questions": questions,
        "answers": answers,
    }


def _local_rows():
    """本地副本里的行；没有 `$MEMORYARENA_DATA` 或目录为空就返回 None。"""
    root = os.getenv("MEMORYARENA_DATA")
    if not root:
        return None
    directory = Path(root).expanduser() / DATASET_CONFIG
    if not directory.is_dir():
        return None
    rows = []
    for path in sorted(directory.glob("*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows or None


def load_travel_data(only_group_id=None):
    """本地副本优先，HF 兜底，返回内部格式的 group 列表。

    `only_group_id` 不为 None 时只留那一组。`run_travel` 的主循环会遍历返回的整个
    列表，所以这是把一次 run 缩到一条 case 的地方；env 要按 id 查全量，不传这个参数。
    """
    rows = _local_rows()
    if rows is None:
        from datasets import load_dataset
        rows = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")

    groups = [_convert_row(row) for row in rows]
    if only_group_id is None:
        return groups

    picked = [g for g in groups if g["id"] == only_group_id]
    if not picked:
        raise SystemExit(f"{DATASET_CONFIG} 里没有 id={only_group_id}")
    return picked
