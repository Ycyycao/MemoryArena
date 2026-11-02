"""memory server 的 ASGI 入口。

和直接跑上游 `server:app` 的区别只有观测。cwd 仍然必须是 `MemoryArena/memory/`
——上游那些平级绝对导入（`from memory_systems import ...`）靠 uvicorn 把 cwd
放进 sys.path。

LLM 调用的记录挂在 litellm 的回调上（`llm_journal`），每条记录带发起它的那个请求的
`user_id`（`runctx`），几条 run 并行时靠这个分开。

    uvicorn harness.memory_app:app --host 0.0.0.0 --port 8010
"""

from . import llm_journal, runctx

# 顺序：先让 litellm 的入口会打标签、线程池会传 context，再挂回调，
# 都要在 import 上游 server 之前。
runctx.tag_litellm()
runctx.propagate_context_to_threads()
llm_journal.install()

from server import app  # noqa: E402  必须在上面几行之后

runctx.install_middleware(app)

__all__ = ["app"]
