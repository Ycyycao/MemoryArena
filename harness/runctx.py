"""让共享的 server 进程知道"这次 LLM 调用是替哪条 run 打的"。

memory server 和 env server 是所有 run 共用的两个进程。几条 run 并行时，它们发出的
LLM 调用在时间上交错，事后靠时间窗或文件偏移分不开。能把一条调用归到某条 run 名下的
唯一线索，是触发它的那个 HTTP 请求：请求体里带着 `user_id`（memory）或 `task_id`
（env），而这两个 id 都由跑 run 的进程用 run-id 做了前缀。

所以做三件事，让这个 id 从请求体一路传到每条 LLM 记录上：

1. **请求 → contextvar。** ASGI 中间件读请求体里的 id，放进 `REQUEST_ID`。
2. **contextvar → litellm 调用参数。** 包一层 `litellm.completion` / `embedding`，
   调用时把 `REQUEST_ID` 塞进 `metadata`（回调里读得到，journal 从那里取）和请求头
   （网关读得到）。**必须在调用时打标签，不能等回调**：litellm 1.97 的同步回调是
   `executor.submit(success_handler)`，跑在线程池里，contextvar 传不过去。
3. **contextvar → 工作线程。** SimpleMem 的抽取和检索都开 `ThreadPoolExecutor`，
   LLM 调用发生在 worker 线程里。给 `submit` 打补丁，用 `copy_context().run`
   提交，任何用线程池的 memory system 都不用再管这件事。

跑 run 的进程（agent 侧）也用同一套：它本身就是每 run 一个，直接把 `REQUEST_ID`
设成 `<run-id>__agent`，于是三个角色的记录长得一样，切片规则只有一条：
`request_id.startswith(f"{run_id}__")`。
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import functools
import json
import os
from typing import Any, Callable, Optional

REQUEST_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "marena_request_id", default=None)

ROLES = ("agent", "memory", "judge", "other")   # 网关按入口路径分的角色
META_KEY = "marena_request_id"          # litellm metadata 里的键
HEADER = "X-Marena-Request"             # 发给网关的请求头
BODY_FIELDS = ("user_id", "task_id")    # 两个 server 的请求体里各有一个


def current() -> Optional[str]:
    return REQUEST_ID.get()


def belongs_to(request_id: Optional[str], run_id: str) -> bool:
    """切片规则，journal 和网关记录共用。"""
    return bool(request_id) and request_id.startswith(f"{run_id}__")


# ------------------------------------------------------------------ litellm

_TAGGED = False


def tag_litellm() -> None:
    """包住 litellm 的四个入口，调用时带上 `REQUEST_ID`。重复调用无害。

    `extra_headers` 只加在 completion 上：`litellm.embedding` 的签名没有这个参数，
    走 `**kwargs` 能不能到 provider 因后端而异，不赌。embedding 的记录靠 `metadata`
    归因就够了——journal 抓得到；网关那份记录里 embedding 会缺 request_id，见
    `gateway.py` 的说明。
    """
    global _TAGGED
    if _TAGGED:
        return
    import litellm

    def wrap(original: Callable[..., Any], with_header: bool) -> Callable[..., Any]:
        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            rid = REQUEST_ID.get()
            if rid:
                metadata = dict(kwargs.get("metadata") or {})
                metadata.setdefault(META_KEY, rid)
                kwargs["metadata"] = metadata
                if with_header:
                    headers = dict(kwargs.get("extra_headers") or {})
                    headers.setdefault(HEADER, rid)
                    kwargs["extra_headers"] = headers
            return original(*args, **kwargs)
        return wrapper

    litellm.completion = wrap(litellm.completion, with_header=True)
    litellm.acompletion = wrap(litellm.acompletion, with_header=True)
    litellm.embedding = wrap(litellm.embedding, with_header=False)
    litellm.aembedding = wrap(litellm.aembedding, with_header=False)
    _TAGGED = True


_SDK_TAGGED = False


def tag_openai_sdk() -> None:
    """让 openai SDK 建出来的客户端默认带上 `X-Marena-Request` 头。

    只给跑 run 的进程用：上游 math agent 的工具循环直接 `OpenAI()`，不经 litellm，
    它的调用只有网关抓得到，没有这个头网关就不知道归谁。头的值在**构造时**取——
    跑 run 的进程里 `REQUEST_ID` 是固定的，这样取没问题；server 进程里不是，所以
    server 不装这个。
    """
    global _SDK_TAGGED
    if _SDK_TAGGED:
        return
    try:
        import openai
    except ImportError:
        return
    original = openai.OpenAI.__init__

    @functools.wraps(original)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        rid = REQUEST_ID.get()
        if rid:
            headers = dict(kwargs.get("default_headers") or {})
            headers.setdefault(HEADER, rid)
            kwargs["default_headers"] = headers
        original(self, *args, **kwargs)

    openai.OpenAI.__init__ = __init__  # type: ignore[method-assign]
    _SDK_TAGGED = True


# ------------------------------------------------------------------ 线程池

_THREADS_PATCHED = False


def propagate_context_to_threads() -> None:
    """让 `ThreadPoolExecutor.submit` 把当前 context 带进 worker 线程。"""
    global _THREADS_PATCHED
    if _THREADS_PATCHED:
        return
    original = concurrent.futures.ThreadPoolExecutor.submit

    def submit(self: Any, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        ctx = contextvars.copy_context()
        return original(self, ctx.run, fn, *args, **kwargs)

    concurrent.futures.ThreadPoolExecutor.submit = submit  # type: ignore[method-assign]
    _THREADS_PATCHED = True


# ------------------------------------------------------------------ ASGI 中间件

class RequestIdMiddleware:
    """从 JSON 请求体里取 `user_id` / `task_id`，整个请求期间放进 `REQUEST_ID`。

    纯 ASGI 写法而不是 `BaseHTTPMiddleware`：读请求体要把 `receive` 里的消息先攒
    完再回放，`BaseHTTPMiddleware` 在这上面有过坑。请求体不是 JSON 或没有那两个字段
    时什么都不做。
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method") not in ("POST", "PUT"):
            await self.app(scope, receive, send)
            return

        messages = []
        body = b""
        while True:
            message = await receive()
            messages.append(message)
            body += message.get("body", b"")
            if not message.get("more_body"):
                break

        async def replay() -> Any:
            return messages.pop(0) if messages else {"type": "http.request", "body": b""}

        token = None
        rid = _extract(body)
        if rid:
            token = REQUEST_ID.set(rid)
        try:
            await self.app(scope, replay, send)
        finally:
            if token is not None:
                REQUEST_ID.reset(token)


def _extract(body: bytes) -> Optional[str]:
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    for field in BODY_FIELDS:
        value = payload.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def install_middleware(app: Any) -> None:
    """给 FastAPI/Starlette app 挂上中间件。要在 app 收到第一个请求之前调。"""
    app.add_middleware(RequestIdMiddleware)


# ------------------------------------------------------------------ 子进程

REQUEST_ID_ENV = "MARENA_REQUEST_ID"    # 传给子进程的归属标记


def child_env(env: dict, role: str = "agent") -> dict:
    """给要 fork 出去的子进程改一份环境变量，让它的 LLM 调用归对角色、归对 run。

    背景：search 那条路的 agent 不在跑 run 的进程里，是 env server `subprocess.run`
    出来的（`agent/search.py`）。子进程默认继承 env server 的环境，于是两件事都错：

    * **角色错。** env server 的 `OPENAI_BASE_URL` 指着网关的 `/judge` 入口，
      子进程照抄，agent 的对话和检索用的 embedding 全被记成判分流量。
    * **归属丢了。** `REQUEST_ID` 是 contextvar，跨不过进程边界；子进程发出的请求
      不带 `X-Marena-Request`，`llm_journal.harvest` 按 run-id 前缀切片时捡不到它们。

    这里把角色段换掉、把当前 `REQUEST_ID` 放进环境变量。子进程那侧调
    `adopt_from_env()` 接上。只改传进来的这份拷贝，不动 `os.environ`。
    """
    child = dict(env)
    child["MARENA_LLM_ROLE"] = role
    base_url = child.get("OPENAI_BASE_URL")
    if base_url:
        child["OPENAI_BASE_URL"] = _swap_role(base_url, role)
    rid = REQUEST_ID.get()
    if rid:
        child[REQUEST_ID_ENV] = rid
    return child


def _swap_role(base_url: str, role: str) -> str:
    """把 `http://host:port/judge` 换成 `.../agent`。

    只在最后一段确实是个已知角色时才动它——指向真 OpenAI 或别家 endpoint 的
    `OPENAI_BASE_URL`（`.../v1`）不能被改。
    """
    head, sep, tail = base_url.rstrip("/").rpartition("/")
    if sep and tail in ROLES:
        return f"{head}/{role}"
    return base_url


def adopt_from_env() -> Optional[str]:
    """子进程入口调：把 `child_env` 放进来的归属标记接回 `REQUEST_ID`。

    返回接到的值（没有就是 None），方便调用方打日志。
    """
    rid = os.getenv(REQUEST_ID_ENV)
    if rid:
        REQUEST_ID.set(rid)
    return rid
