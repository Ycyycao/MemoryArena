"""LLM 网关：所有进程的 LLM 调用都从这里过，落一份完整轨迹。

## 为什么要它

这个仓库里 LLM 调用点有二十来处，**没有一处在我们自己的代码里**——全在上游的
`MemoryArena/` 和 pip 装的 mem0 里，用的 SDK 还不是一种。所以"统一走某个客户端库"
在调用点那层做不到。唯一能统一的位置是进程外：一个说 OpenAI 协议的网关。

一个网关同时解决两件事：

* **换 provider。** 调用方只知道 OpenAI 协议，换后端是改网关的配置，不是改二十个
  调用点。memory 侧连 backend 开关都没有（`reasoningbank.py` 直接
  `OpenAI(api_key=...)`），只是碰巧能用——openai SDK 不传 `base_url` 时会自己读
  `OPENAI_BASE_URL`。靠这个隐式行为撑着的东西，换个 SDK 就断，网关是唯一的显式杠杆。
* **存轨迹。** 请求和响应的完整 body 都在这里过一遍，落成 JSONL，`usage` 一并带上
  ——上游的 `usage_stats.json` 只统计 agent 侧。这取代了之前那两处 monkeypatch
  （`llm_trace` 在 agent 进程拦 SDK 录对话、`usage_meter` 在 memory server 进程拦
  SDK 数 token）：它们各自只覆盖一个进程，而且依赖 openai SDK 的内部模块路径。

## 角色靠路径分，不靠改代码

网关不知道"这次调用是第 5 轮 Emma 的 memory 侧抽取"。解决办法是给每个进程不同的
入口路径，`marena-serve` 本来就用 `tmux new-window -e` 给每个窗口单独传环境变量：

    memory server  OPENAI_BASE_URL=http://127.0.0.1:8100/memory/v1
    env server     OPENAI_BASE_URL=http://127.0.0.1:8100/judge/v1
    agent          OPENAI_BASE_URL=http://127.0.0.1:8100/agent/v1

于是每条记录自带 `role`，零补丁。路径前缀之后的部分原样转发给上游。

## 归到哪条 run：看请求头

记录常开，全部落 `$MARENA_RESULTS/_llm/gateway.jsonl`，和 litellm 回调那份放一起。
每条带 `request_id`，从请求头 `X-Marena-Request` 读——`harness.runctx` 包住的 litellm
调用会带上这个头，值是 `<run-id>__...`。run 跑完由 `llm_journal.harvest` 按前缀切走。

之前是一个 `record/start|stop` 控制面，一次只能录一条 run，第二条一启动就把第一条
截断了；几条 run 并行之后没法用，删了。

没带这个头的请求（mem0 那个 openai SDK 的 embedder；litellm 的 `embedding()` 签名
没有 `extra_headers`，也不加）记录里 `request_id` 为空，不归任何 run。

## 只做一件事：记轨迹

不解析、不改写请求体，原样转发，**响应也原样返回**。网关不会成为新的变量——上游怎么
答，调用方就看到什么。

**换 provider 不归它管。** 调用点自己用的就是 litellm（`travel_planner_env/clients/
openai_client.py`、`memory_systems/reasoningbank.py`、`formal_reasoning_env/
llm_backend.py`，以及 mem0 的 `provider: litellm`），换后端是改模型名。网关一度也有
个 litellm 路由模式，调用点转过来之后那就是同一件事做两遍，删掉了。

留着网关是因为 **litellm 的回调只能抓到走 litellm 的调用**，而 mem0 的 embedder 仍然
是 openai SDK（mem0 的 `embeddings/` 里没有 litellm 实现）。网关在 HTTP 那一层，抓得
到所有人。

请求头里的 `authorization` 不进日志。
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from . import runctx

# 角色定义在 runctx：子进程那侧（runctx.child_env）也要按角色改 base_url，
# 两处必须是同一份。
ROLES = runctx.ROLES
HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer",
              "proxy-authorization", "proxy-authenticate", "upgrade", "host",
              "content-length", "accept-encoding"}


class Journal:
    """常开地往 `_llm/gateway.jsonl` 追加。归属靠记录里的 `request_id`，不靠启停。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._path: Optional[Path] = None
        self._seq = 0

    def path(self) -> Path:
        if self._path is None:
            from . import llm_journal
            self._path = llm_journal.journal_path("gateway")
        return self._path

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"path": str(self.path()), "calls": self._seq}

    def write(self, record: Dict[str, Any]) -> None:
        from . import llm_journal
        with self._lock:
            self._seq += 1
            record = {"seq": self._seq, **record}
        llm_journal.append_line(self.path(), json.dumps(record, ensure_ascii=False))


JOURNAL = Journal()


def upstream_base() -> str:
    """真正的上游。故意不叫 OPENAI_BASE_URL——那个已经被指向网关自己了，
    两者同名会绕成一个环。"""
    base = os.getenv("MARENA_GATEWAY_UPSTREAM")
    if not base:
        raise SystemExit("MARENA_GATEWAY_UPSTREAM 没设置，网关不知道往哪转发")
    return base.rstrip("/")


def _usage_of(body: bytes) -> Optional[Dict[str, Any]]:
    try:
        usage = json.loads(body).get("usage")
    except (ValueError, AttributeError):
        return None
    return usage if isinstance(usage, dict) else None


def _json_or_raw(body: bytes, limit: int = 4_000_000) -> Any:
    """能解析成 JSON 就存结构，否则存字符串。超大响应截断——复盘要的是内容，
    不是把几十 MB 的向量原样再抄一遍。"""
    if len(body) > limit:
        return {"_truncated": True, "_bytes": len(body), "head": body[:2000].decode("utf-8", "replace")}
    try:
        return json.loads(body)
    except ValueError:
        return body.decode("utf-8", "replace")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:      # 默认会往 stderr 刷每条请求，太吵
        pass

    # -- 我们自己的控制面 ---------------------------------------------------

    def _reply(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _control(self, path: str) -> bool:
        if path == "/marena/record/status":
            self._reply(JOURNAL.status())
            return True
        if path == "/marena/health":
            self._reply({"status": "ok", "upstream": upstream_base()})
            return True
        return False

    # -- 转发 ---------------------------------------------------------------

    def _split_role(self, path: str) -> tuple[str, str]:
        for role in ROLES:
            if path.startswith(f"/{role}/"):
                return role, path[len(role) + 1:]
        return "other", path

    def _forward(self, method: str) -> None:
        if self._control(self.path):
            return

        role, rest = self._split_role(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        payload = self.rfile.read(length) if length else b""

        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        request_id = self.headers.get(runctx.HEADER)
        headers.pop(runctx.HEADER, None)         # 我们自己的头，不往上游带
        url = f"{upstream_base()}{rest}"
        started = time.time()

        request = urllib.request.Request(url, data=payload or None, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                status, body = response.status, response.read()
                out_headers = [(k, v) for k, v in response.headers.items()
                               if k.lower() not in HOP_BY_HOP]
        except urllib.error.HTTPError as error:      # 4xx/5xx 也要原样透给调用方
            status, body = error.code, error.read()
            out_headers = [(k, v) for k, v in (error.headers or {}).items()
                           if k.lower() not in HOP_BY_HOP]
        except Exception as error:                   # 网络层挂了，编一个 502
            status = 502
            body = json.dumps({"error": {"message": f"gateway upstream failed: {error}"}}).encode()
            out_headers = [("Content-Type", "application/json")]

        self._finish(started, role, request_id, rest, status, payload, body, out_headers)

    def _finish(self, started: float, role: str, request_id: Optional[str], rest: str,
                status: int, payload: bytes, body: bytes, out_headers: list) -> None:
        """收尾：先落轨迹，再把响应还给调用方。"""
        JOURNAL.write({
            "t": round(started, 3),
            "duration": round(time.time() - started, 3),
            "role": role,
            "request_id": request_id,
            "path": rest,
            "status": status,
            "request": _json_or_raw(payload) if payload else None,
            "response": _json_or_raw(body),
            "usage": _usage_of(body),
        })

        self.send_response(status)
        for key, value in out_headers:
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        self._forward("POST")

    def do_GET(self) -> None:
        self._forward("GET")


def serve(port: int) -> None:
    upstream_base()          # 早失败：没配上游就别起
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


def main() -> None:
    serve(int(os.getenv("MARENA_GATEWAY_PORT", "8100")))


if __name__ == "__main__":
    main()
