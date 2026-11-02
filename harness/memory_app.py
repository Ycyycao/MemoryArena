"""memory server 的 ASGI 入口。

和直接跑上游 `server:app` 的区别只有观测。cwd 仍然必须是 `MemoryArena/memory/`
——上游那些平级绝对导入（`from memory_systems import ...`）靠 uvicorn 把 cwd
放进 sys.path。

LLM 调用带上发起它的那个请求的 `user_id`（`runctx`），几条 run 并行时网关的记录靠
这个分开。

    uvicorn harness.memory_app:app --host 0.0.0.0 --port 8010
"""

from . import runctx

# 都要在 import 上游 app 之前。
runctx.tag_openai_sdk()
runctx.propagate_context_to_threads()

from server import app  # noqa: E402  必须在上面几行之后

runctx.install_middleware(app)

__all__ = ["app"]
