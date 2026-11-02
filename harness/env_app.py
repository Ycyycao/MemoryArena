"""env server 的 ASGI 入口。

和直接跑上游 `env/env_server.py` 的区别只有观测：judge 的 LLM 调用带上发起它的那个
请求的 `task_id`（`runctx`），几条 run 并行时网关的记录靠这个分开。

    uvicorn harness.env_app:app --host 0.0.0.0 --port 8011
"""

from . import runctx

# 都要在 import 上游 app 之前。
runctx.tag_openai_sdk()
runctx.propagate_context_to_threads()

from env.env_server import app  # noqa: E402  必须在上面几行之后

runctx.install_middleware(app)

__all__ = ["app"]
