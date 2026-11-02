"""Mem0 的本地（OSS）实现，接口和上游的 `Mem0MemorySystem` 一样。

上游 `MemoryArena/memory/memory_systems/mem0.py` 用的是 `MemoryClient`——mem0 的托管
服务，要 `MEM0_API_KEY`，抽取和检索都跑在他们的服务器上。我们没有那个 key，而且托
管版本会把"用哪个模型抽事实"这个变量藏起来，对比时说不清是方法差异还是模型差异。

`mem0ai` 同一个包里还导出了 `Memory`，是同一套 ADD/UPDATE/DELETE 逻辑的本地版：
LLM、embedding、向量库都可以指到自己的。这个类就是把它接成上游那两个方法。

**两处会影响结论的差异**，写在这里以免以后拿数字时忘了：

1. **没有 `categories`。** 托管版给每条记忆自动打分类标签，`wrap_user_prompt` 会把
   它拼进去。OSS 不产出这个字段，所以注入的文本里少了这一层。
2. **检索退化成纯语义。** `_search_vector_store` 会同时做语义和关键词（BM25）检索
   再融合，但 FAISS 后端没有 `keyword_search`，那一路返回 None，只剩语义 + 实体加
   权。托管版是混合检索。

`mem0-g`（graph 变体）没有对应实现：它要外挂 Neo4j/Memgraph，本地起不来。

顺带落一份 trace：mem0 抽了什么、检索回了什么，光看 agent 轨迹是看不见的，而这正是
"记忆在哪一步走偏"要看的东西。
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

# mem0 OSS 默认往 PostHog 发遥测。开关在它的 telemetry 模块 import 时就读死了，所以
# 必须赶在 `import mem0` 之前设——这个模块只在 `Mem0LocalMemorySystem.memory` 里懒
# import mem0，所以在这里设来得及。显式设过就不覆盖。
os.environ.setdefault("MEM0_TELEMETRY", "False")

EMBEDDING_DIMS = 1536


def _config(collection: str, store_path: Path) -> dict:
    """mem0 自己会读 `OPENAI_API_KEY` / `OPENAI_BASE_URL`，这里只写它猜不到的。

    模型不写死：`MARENA_MEM0_MODEL` 不设就用 mem0 自己的默认（2.0.18 是
    `gpt-5-mini`）。和 ReasoningBank 那边同一个道理——memory 侧用哪个模型是一次 run
    的参数，不该由代码替某次实验决定。
    """
    # provider 用 openai，**不要**换成 mem0 自带的 litellm provider。两条路径行为不同：
    #
    #   llms/openai.py   调 `_get_supported_params()`，对 reasoning model 会把
    #                    temperature / top_p / max_tokens 全部不发
    #   llms/litellm.py  自己拼 params，从不调那个函数，`max_completion_tokens`
    #                    恒等于 `max_tokens`（默认 2000）
    #
    # 换成 litellm 之后实测：chunk 一变长（prompt 12.7k），gpt-5-mini 就把 2000 的
    # 预算全烧在 reasoning 上，completion_tokens=0、finish_reason=length，抽出 0 条
    # 记忆。第 1 轮（prompt 8.6k）还能出 4 条，第 2 轮起全空。
    #
    # 更根本的理由：mem0 是**被测对象**，改它调 LLM 的方式就是改我们在测的东西，
    # 而且会和同一组用 openai provider 跑出来的 long_context / reasoningbank 失去
    # 可比性。基础设施统一走 litellm，被测对象保持原样。
    #
    # 轨迹不受影响：openai provider 认 `OPENAI_BASE_URL`，照样经过网关。
    llm: dict[str, Any] = {"provider": "openai", "config": {}}
    model = os.getenv("MARENA_MEM0_MODEL")
    if model:
        llm["config"]["model"] = model
    # gpt-5 系列拒绝 temperature != 1，而 mem0 的 `_is_reasoning_model` 只认
    # `gpt-5`/`o1-`/`o3-`，`gpt-5-mini` 落到普通分支会带上 temperature=0.1。显式
    # 声明成 reasoning model，让它把 temperature / max_tokens 都不发。
    if (model or "gpt-5-mini").lower().startswith(("gpt-5", "o1", "o3")):
        llm["config"]["is_reasoning_model"] = True

    return {
        "llm": llm,
        # embedder 只能留在 openai：mem0 的 `embeddings/` 里没有 litellm 实现
        # （`llms/` 里有）。它认 `OPENAI_BASE_URL`，所以照样走网关。
        "embedder": {
            "provider": "openai",
            "config": {
                "model": os.getenv("MARENA_MEM0_EMBEDDING", "text-embedding-3-small"),
                "embedding_dims": EMBEDDING_DIMS,
            },
        },
        "vector_store": {
            "provider": "faiss",
            "config": {
                "collection_name": collection,
                "path": str(store_path),
                "embedding_model_dims": EMBEDDING_DIMS,
                "distance_strategy": "cosine",
            },
        },
    }


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:120]


class Mem0LocalMemorySystem:
    """接口和上游 `Mem0MemorySystem` 一致：`add_chunk` + `wrap_user_prompt`。

    `user_id` 在构造时通常还不知道——memory server 的工厂是无参调用的，真正的
    user_id 要等 `initialize` 把实例塞进注册表时才有。所以 mem0 的实例是懒构造的，
    `bind_user_id` 先把 id 定下来（见 `memory_patches.patch_mem0`）。
    """

    def __init__(self, user_id: Optional[str] = None, state_dir: Optional[Path] = None,
                 enable_graph: bool = False):
        if enable_graph:
            raise NotImplementedError(
                "mem0-g 需要外挂图数据库（Neo4j/Memgraph），本地没有")
        self.user_id = user_id or str(uuid.uuid4())
        if state_dir:
            self._state_dir = Path(state_dir)
        else:
            from memory.state import state_dir as _default_state_dir
            self._state_dir = _default_state_dir("mem0")
        self._memory = None
        self._lock = threading.Lock()

    # -- 懒构造 ---------------------------------------------------------------

    def bind_user_id(self, user_id: str) -> None:
        if self._memory is not None:
            raise RuntimeError("mem0 实例已经建好了，不能再改 user_id")
        self.user_id = user_id

    @property
    def memory(self):
        with self._lock:
            if self._memory is None:
                from mem0 import Memory

                store = self._state_dir / _slug(self.user_id)
                store.mkdir(parents=True, exist_ok=True)
                self._memory = Memory.from_config(_config(_slug(self.user_id), store))
            return self._memory

    # -- trace ----------------------------------------------------------------

    def _trace(self, record: dict) -> None:
        path = self._state_dir / f"{_slug(self.user_id)}.trace.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -- 上游接口 --------------------------------------------------------------

    def add_chunk(self, chunk: str):
        """和上游一样包成 user/assistant 一问一答再交给 mem0。

        那句固定的 assistant 回复是上游写的，不是我们加的——mem0 的抽取 prompt 吃的
        是对话，给它一个单边的 user 消息会让它按"用户说了什么"来抽。保留原样，免得
        又多一个差异。
        """
        result = self.memory.add(
            [
                {"role": "user", "content": chunk},
                {"role": "assistant", "content": "Thanks for the information! I will remember this."},
            ],
            user_id=self.user_id,
        )
        self._trace({"op": "add", "chunk_chars": len(chunk), "chunk": chunk, "result": result})
        return result

    def wrap_user_prompt(self, prompt: str):
        """检索并拼进 prompt。文本格式逐字照抄上游，包括 `prompt.lower()` 这个查询。"""
        memories = self.memory.search(prompt.lower(), filters={"user_id": self.user_id})
        results = memories.get("results", []) if isinstance(memories, dict) else []

        lines = ["<memory_context>"]
        for result in results:
            text = result.get("memory")
            if not text:
                continue
            categories = result.get("categories") or []      # OSS 一般为空，见模块注释
            if categories:
                lines.append(f"{text} (categories: {', '.join(categories)})")
            else:
                lines.append(text)
        lines.append("</memory_context>")
        lines.append(f"User Prompt: {prompt}")
        wrapped = "\n".join(lines)

        self._trace({
            "op": "search",
            "query_chars": len(prompt),
            "hits": [{"memory": r.get("memory"), "score": r.get("score")} for r in results],
            "wrapped_chars": len(wrapped),
        })
        return wrapped
