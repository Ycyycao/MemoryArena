"""env server 的 ASGI 入口。

和直接跑上游 `env/env_server.py` 的区别只有观测：给 judge 的 LLM 调用挂上记录，
每条记录带发起它的那个请求的 `task_id`（`runctx`），几条 run 并行时靠这个分开。

    uvicorn harness.env_app:app --host 0.0.0.0 --port 8011
"""

from . import llm_journal, runctx

# 顺序：先让 litellm 的入口会打标签、线程池会传 context，再挂回调。
runctx.tag_litellm()
runctx.propagate_context_to_threads()
llm_journal.install()

from env.env_server import app  # noqa: E402  必须在上面几行之后

runctx.install_middleware(app)

__all__ = ["app"]
