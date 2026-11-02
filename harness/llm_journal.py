"""按 run 切 LLM 调用记录。

三个进程（agent / memory server / env server）都会调大模型，它们的 `OPENAI_BASE_URL`
都指向 `harness.gateway`。网关把每条请求和响应落到 `$MARENA_RESULTS/_llm/gateway.jsonl`，
这里负责把属于某条 run 的记录切出来。

## 怎么归到某一次 run 名下

不做启停控制面。网关**一直**往 `$MARENA_RESULTS/_llm/gateway.jsonl` 追加，
每条记录带一个 `request_id`——由 `harness.runctx` 在调用 openai SDK 时放进请求头，
形如 `<run-id>__...`。run 跑完，`harvest` 把带自己前缀的记录切进这条 run 的目录
（`llm_calls.<role>.jsonl`）。

之前是按文件偏移切（跑前记大小、跑完取新长出来的那段），几条 run 并行时会互相切进
对方目录，所以改成按 `request_id` 过滤；偏移只留作扫描起点，省得每次从头读。

这样三个进程互不协调，也不用给两个 server 各加一对 HTTP 路由。代价是 server 长期运行
时那几个文件会一直变长，需要偶尔清理——`marena-serve stop` 不动它们，因为里面可能还有
没切走的记录。

`role` 从 `MARENA_LLM_ROLE` 读，由 `marena-serve` 给每个进程注入。几个 agent 进程会
同时往 `agent.jsonl` 追加，所以一条记录一次 `os.write`——`O_APPEND` 下单次 write
不会和别的进程交错，Python 的缓冲写会分块，不行。
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from . import runctx

_LOCK = threading.Lock()


def _role() -> str:
    return os.getenv("MARENA_LLM_ROLE", "other")


def journal_dir() -> Path:
    base = os.getenv("MARENA_LLM_JOURNAL_DIR") or os.path.join(
        os.environ["MARENA_RESULTS"], "_llm")
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def journal_path(role: Optional[str] = None) -> Path:
    return journal_dir() / f"{role or _role()}.jsonl"


def append_line(path: Path, line: str) -> None:
    """一次 write 追加一整行，跨进程不交错。"""
    data = (line + "\n").encode("utf-8")
    with _LOCK:                              # 同进程内多线程再串一下
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)


# ------------------------------------------------------------------ 切片

def marks() -> Dict[str, int]:
    """记下每个 role 的文件现在多大。跑 run 之前调一次。"""
    out: Dict[str, int] = {}
    for path in journal_dir().glob("*.jsonl"):
        out[path.stem] = path.stat().st_size
    return out


def harvest(marks_before: Dict[str, int], out_dir: Path, run_id: str) -> Dict[str, int]:
    """把属于这条 run 的记录切进 out_dir，返回各 role 切了几条。

    从记号处往后扫（这条 run 的记录不可能在它开始之前），只留
    `request_id` 带 `<run_id>__` 前缀的行。没打标签的记录（比如 mem0 那个还在用
    openai SDK 的 embedder 经网关留下的）不归任何 run，留在全局文件里。
    """
    counts: Dict[str, int] = {}
    for path in sorted(journal_dir().glob("*.jsonl")):
        start = marks_before.get(path.stem, 0)
        kept = []
        with path.open("rb") as handle:
            handle.seek(start)
            for raw in handle:
                try:
                    rid = json.loads(raw).get("request_id")
                except ValueError:
                    continue
                if runctx.belongs_to(rid, run_id):
                    kept.append(raw)
        if not kept:
            continue
        target = out_dir / f"llm_calls.{path.stem}.jsonl"
        target.write_bytes(b"".join(kept))
        counts[path.stem] = len(kept)
    return counts
