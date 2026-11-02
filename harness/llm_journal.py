"""把每一次 LLM 调用记下来，不管打给哪家。

三个进程（agent / memory server / env server）都会调大模型。要复盘"这一步为什么这么
做"，光有接口层的事件不够，得有真正发出去的 messages 和真正收回来的响应。

之前这件事由 `harness.gateway` 做——它夹在中间，把 HTTP 流量抄一份。那有个洞：网关只在
目标是 OpenAI 协议时才会经过。一旦 litellm 把调用路由到 Gemini，流量直接打给 Google，
绕开网关，什么都记不到。

这里换个位置：挂在 litellm 自己身上。litellm 每次调用结束都会回调，那时候
provider 已经不重要了——不管打给谁，回调都会来。

两层现在是互补的，各自补对方的洞：

| | 抓得到 |
| --- | --- |
| 网关（HTTP 层） | 指向它的所有流量，**包括 mem0 那个还在用 openai SDK 的 embedder** |
| 这里（litellm 回调） | 所有走 litellm 的调用，**不管什么 provider** |

## 怎么归到某一次 run 名下

不做启停控制面。三个进程各自**一直**往 `$MARENA_RESULTS/_llm/<role>.jsonl` 追加，
每条记录带一个 `request_id`——由 `harness.runctx` 在调用 litellm 时塞进 `metadata`，
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
_INSTALLED = False


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


def _dump(obj: Any) -> Any:
    """litellm 传回来的可能是 pydantic 模型、dict，也可能是别的。都变成能 json 的。"""
    for attr in ("model_dump", "dict"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:
                pass
    if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
        return obj
    return str(obj)[:4000]


class _Journal:
    """litellm 的 CustomLogger 子类。成功和失败都记——失败那条往往才是要复盘的。"""

    def _write(self, kwargs: Dict[str, Any], response_obj: Any,
               start_time: Any, end_time: Any, ok: bool) -> None:
        # litellm 把请求参数原样放在 kwargs 里；`litellm_params` 那堆是它自己的内部
        # 记账（重试次数、fallback 之类），对复盘没用，滤掉。
        request = {k: _dump(v) for k, v in kwargs.items()
                   if k in ("model", "messages", "input", "tools", "tool_choice",
                            "temperature", "max_tokens", "max_completion_tokens",
                            "response_format")}
        response = _dump(response_obj)
        usage = response.get("usage") if isinstance(response, dict) else None
        metadata = (kwargs.get("litellm_params") or {}).get("metadata") or {}

        record = {
            "t": start_time.timestamp() if hasattr(start_time, "timestamp") else None,
            "source": "litellm",
            "role": _role(),
            "request_id": metadata.get(runctx.META_KEY),
            "ok": ok,
            "duration": (end_time - start_time).total_seconds()
            if hasattr(end_time, "__sub__") and hasattr(start_time, "__sub__") else None,
            "request": request,
            "response": response,
            "usage": usage,
        }
        append_line(journal_path(), json.dumps(record, ensure_ascii=False, default=str))

    # litellm 调用的四个钩子
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._write(kwargs, response_obj, start_time, end_time, ok=True)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._write(kwargs, response_obj, start_time, end_time, ok=False)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._write(kwargs, response_obj, start_time, end_time, ok=True)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._write(kwargs, response_obj, start_time, end_time, ok=False)


def append_line(path: Path, line: str) -> None:
    """一次 write 追加一整行，跨进程不交错。"""
    data = (line + "\n").encode("utf-8")
    with _LOCK:                              # 同进程内多线程再串一下
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)


def install() -> None:
    """把回调挂到 litellm 上。重复调用无害。"""
    global _INSTALLED
    if _INSTALLED:
        return
    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    journal = type("MarenaJournal", (_Journal, CustomLogger), {})()
    litellm.callbacks = list(getattr(litellm, "callbacks", None) or []) + [journal]
    _INSTALLED = True


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
