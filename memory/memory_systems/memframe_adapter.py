"""DynMem 的 memframe 后端（mf_dynmem、mf_bm25、simplemem ...）当成 MemoryArena 的 memory system 用。

memframe（DynMem 仓库 methods/memframe/memframe.py）的后端接口是 reset / add / retrieve / end_subtask / close，
MemoryArena 的 memory system 接口是 add_chunk / wrap_user_prompt（加上我们 server 的 bind_user_id）。这里把后者
映射到前者，后端的代码一行不改；K（V2_MF_K，8）和检索块预算（V2_MF_BLOCK_TOKENS，16384）用 memframe 的默认值。

## 调用映射

| 官方 agent（run_travel.run_group）                      | 后端                                                        |
| ------------------------------------------------------- | ----------------------------------------------------------- |
| `/memory/initialize` → `bind_user_id(user_id)`            | `backend.ctx = MemContext(...)`，`reset(user_id)`             |
| `add(chunk)`：第 j 次（base person 是第 1 次，第 r 个加入的人是第 r+1 次） | memframe 子任务 k = j：chunk 拆成步，每步一次 `add(text, meta)`，然后 `end_subtask(record)` |
| `wrap_user_prompt(query)`（每轮开头；use_step_memory 时每个工具步之后也调） | `retrieve(query, K)`，ctx.subtask = 已加的 chunk 数 + 1        |
| 一组结束（只有同进程的 runner 会调）                     | `close()`                                                    |

一个官方 chunk 拆成 memframe 的步（`memframe.chunk_text` 的格式 "Subtask k / Step n: Action: Name[{json}]\\nObservation: ..."）：

- base person（`is_base_person`）：一步，`BasePlan[{"name", "query"}]`，Observation 是它定好的计划。
- 一个人的一轮（有 `scratchpad`）：按 scratchpad 的顺序，有 thought 的步先记一步 `Think[{"thought"}]`
  （V2_MF_CHUNK_THINK=0 不记，同 memframe），每个工具调用一步 `<工具名>[<参数>]`，Observation 是工具原样返回；
  最后一步 `Finish[{"response": 提交的计划}]`，Observation 是 env 给的判语（judgement_mode=hint 时是要改的格，
  没有判语就是空）。
- 别的 chunk（不是 JSON，或者没有这些字段）：一步 `Record[{}]`，Observation 是原文。

end_subtask 的 record 和 memframe 的同形：subtask、instruction（这个人的 query）、answer（提交的计划）、
completion（判语）、chunks、last_observation，外加 name。

检索结果用 `memframe.render_block` 渲染：排序不动，每条整条放或整条不放，总共不超过 BLOCK_TOKENS，
后端 `respects_k` 为假时不截 K。没有结果时返回上游 memory system 的空形态 `<memory_context>\\nNone\\n</memory_context>`。
返回值是 `<block>\\nUser: <query>`，官方 agent 按 `</memory_context>` 切出前半段放进 prompt。

和 MAGE harness 上的 memframe 的差别（都是官方 agent 决定的）：一轮只在开头检索一次，query 是这个人的原话
（memframe 是每步检索，query = 子任务文本 + 最新 observation）；当前一轮的步在这一轮结束时才写进记忆
（memframe 是每步写）；ctx.step 是这一轮里第几次检索，不是 agent 的步数。

后端出错照 memframe 的规矩：计数（mf_add_errors / mf_retrieve_errors / mf_end_subtask_errors / mf_close_errors），
每组每类第一次打到 stderr，这一步当作没有记忆接着跑；V2_MF_STRICT=1 时直接抛。

## DynMem 代码从哪来

先直接 `import memframe`（DynMem 的环境里 core/install_paths.sh 已经把目录写进 dynmem.pth）；不行就读
`DYNMEM_ROOT`，用它的 core/code_paths.py 把目录加进 sys.path。不写死任何家目录路径。

memory server 里（`memory_system_name = "mf_<后端>"`）没人传 LLM，用 `_default_llm()`：OpenAI 兼容端点
`MARENA_MEMFRAME_BASE_URL`，没有就是 server 自己的 `OPENAI_BASE_URL`（网关的 /memory 入口），再没有就是 DynMem
profile 的 RG_BASE_URL；模型 `MARENA_MEMFRAME_MODEL`，没有就是 RG_LLM。embedding 是 memframe 自己读的
VLLM_EMBEDDING_URL / V2_EMB_MODEL。
"""

import json
import os
import sys
import threading
import time
import uuid
from types import SimpleNamespace

DATASET = "travelplan"   # memframe 的 meta["dataset"] 和 MemContext.dataset，与 MAGE harness 上 travel 的名字一致
EMPTY_BLOCK = "<memory_context>\nNone\n</memory_context>"

# memframe.BACKENDS 里没有、但 DynMem 的 runner 用这个名字的后端（core/run_qa.py 同样现场注册）
EXTRA_BACKENDS = {"simplemem": "mem_simplemem:SimpleMemBackend"}

_import_lock = threading.Lock()


def import_memframe():
    """DynMem 的 memframe 模块。"""
    with _import_lock:
        try:
            import memframe
            return memframe
        except ImportError:
            root = os.getenv("DYNMEM_ROOT")
            if not root:
                raise RuntimeError("import memframe failed and DYNMEM_ROOT is not set: point it at the DynMem checkout "
                                   "(or run in DynMem's environment, where core/install_paths.sh put its folders)")
            core = os.path.join(root, "core")
            if core not in sys.path:
                sys.path.append(core)
            import code_paths
            code_paths.install(root)
            import memframe
            return memframe


def resolve(memframe, backend):
    """backend: "dynmem"、"bm25"、"simplemem" 这样的后端名（不带 mf_）。"""
    if backend not in memframe.backends() and backend in EXTRA_BACKENDS:
        memframe.register_backend(backend, EXTRA_BACKENDS[backend])
    return memframe.make_backend(backend)


def _default_llm():
    import rgconfig
    from run_qa import Endpoint
    base_url = (os.getenv("MARENA_MEMFRAME_BASE_URL") or os.getenv("OPENAI_BASE_URL") or rgconfig.get("BASE_URL"))
    model = os.getenv("MARENA_MEMFRAME_MODEL") or rgconfig.get("LLM")
    return Endpoint(base_url, model, os.getenv("OPENAI_API_KEY") or "EMPTY")


def chunk_steps(chunk, think=True):
    """官方的一个 chunk -> ([(action, params, observation)], record 字段)。映射见文件头；think=False 不记 Think 步。"""
    try:
        d = json.loads(chunk)
    except (TypeError, ValueError):
        d = None
    if not isinstance(d, dict):
        return [("Record", {}, str(chunk))], {"instruction": "", "answer": "", "completion": ""}
    if d.get("is_base_person"):
        steps = [("BasePlan", {"name": d.get("name"), "query": d.get("query")}, str(d.get("final_plan") or ""))]
        return steps, {"instruction": d.get("query") or "", "answer": d.get("final_plan") or "", "completion": "",
                       "name": d.get("name")}
    if "scratchpad" not in d and "final_plan" not in d:
        return [("Record", {}, str(chunk))], {"instruction": "", "answer": "", "completion": ""}
    steps = []
    for st in d.get("scratchpad") or []:
        thought = st.get("thought")
        if thought and think:
            steps.append(("Think", {"thought": thought}, ""))
        results = {r.get("tool_call_id"): r for r in st.get("tool_results") or []}
        for tc in st.get("tool_calls") or []:
            r = results.get(tc.get("id")) or {}
            steps.append((tc.get("name") or "?", tc.get("args") or {}, str(r.get("result", ""))))
    judgement = d.get("judgement") or ""
    steps.append(("Finish", {"response": d.get("final_plan") or ""}, judgement))
    return steps, {"instruction": d.get("query") or "", "answer": d.get("final_plan") or "", "completion": judgement,
                   "name": d.get("name")}


class MemframeMemorySystem:
    """一个 user_id（一组）一个实例；server 每次 /memory/initialize 新建一个。"""

    def __init__(self, backend, llm=None, k=None, block_tokens=None, dataset=DATASET):
        mf = import_memframe()
        self.mf, self.backend_name, self.dataset = mf, backend, dataset
        self.backend = resolve(mf, backend)
        self.k = mf.K if k is None else k
        self.block_tokens = mf.BLOCK_TOKENS if block_tokens is None else block_tokens
        self._llm = llm
        self.ctx = None
        self.n_records = 0          # 已加的官方 chunk 数 = 已结束的 memframe 子任务数
        self.searches = 0           # 当前子任务里第几次检索
        self.warned = set()
        self.closed = False

    # ---- 官方接口
    def bind_user_id(self, user_id):
        llm = self._llm or _default_llm()
        self.ctx = self.mf.MemContext(self.dataset, user_id, llm)
        self.backend.ctx = self.ctx
        self.backend.reset(user_id)

    def add_chunk(self, chunk):
        self._ensure()
        k = self.n_records + 1
        steps, rec = chunk_steps(chunk, think=self.mf.CHUNK_THINK)
        texts = []
        for n, (action, params, obs) in enumerate(steps, 1):
            self.ctx.subtask, self.ctx.step = k, n
            text = self.mf.chunk_text(k, n, SimpleNamespace(name=action, params=params), obs)
            meta = {"subtask": k, "step": n, "action": action, "params": params, "observation": obs,
                    "dataset": self.dataset}
            self._safe("add", lambda: self.backend.add(text, meta))
            self.mf.bump("mf_chunks")
            texts.append(text)
        # memories: the steps handed to the backend's add(); a backend that stores something else (SimpleMem's units,
        # DynMem's objects) writes its own add records with what it stored
        self._trace(k, {"kind": "memory", "op": "add", "source": "memframe_adapter", "chunk": chunk,
                        "memories": texts})
        if callable(getattr(self.backend, "end_subtask", None)):
            record = {"subtask": k, **rec, "chunks": texts, "last_observation": steps[-1][2] if steps else ""}
            self._safe("end_subtask", lambda: self.backend.end_subtask(record))
        self.n_records, self.searches = k, 0
        return {"subtask": k, "steps": len(texts)}

    def wrap_user_prompt(self, prompt):
        self._ensure()
        k = self.n_records + 1
        self.ctx.subtask, self.ctx.step = k, self.searches
        self.searches += 1
        self._trace(k, None)   # the backend's own calls during the search land in this subtask
        results = self._safe("retrieve", lambda: list(self.backend.retrieve(prompt, self.k) or []), default=[])
        mf = self.mf
        mf.bump("mf_retrievals")
        mf.bump("mf_results", len(results))
        limit = self.k if getattr(self.backend, "respects_k", True) else None
        block, n, used = mf.render_block(results, budget=self.block_tokens, limit=limit)
        for name, v in n.items():
            mf.bump(f"mf_{name}", v)
        if block:
            mf.bump("mf_block_steps")
            st = mf.common.stats()
            st["mf_block_tokens_max"] = max(st["mf_block_tokens_max"], used)
        self._trace(k, {"kind": "memory", "op": "search", "query": prompt, "hits": [str(r) for r in results],
                        "context": block, "shown": n.get("shown", 0), "left_out": n.get("left_out", 0)})
        return f"{block or EMPTY_BLOCK}\nUser: {prompt}"

    # ---- 同进程 runner 用
    def add(self, chunk):
        """MemoryClient 的名字：run_group 在同一进程里直接拿这个对象当 client 用。"""
        return self.add_chunk(chunk)

    def close(self):
        if self.closed or self.ctx is None:
            return
        self.closed = True
        if callable(getattr(self.backend, "close", None)):
            self._safe("close", self.backend.close)

    def dump(self):
        return self.backend.dump() if callable(getattr(self.backend, "dump", None)) else None

    # ---- 内部
    def _ensure(self):
        if self.ctx is None:
            self.bind_user_id(uuid.uuid4().hex)

    def _trace(self, k, rec):
        import traj
        t = traj.current()
        if t is None:
            return
        # trajectory rounds are the travelers: storing the base person comes before the first round (subtask None, S0),
        # the r-th joining traveler (memframe subtask r + 1) is subtask r - 1 (S<r>), the round its question is scored in
        t.at(k - 2 if k > 1 else None, None)
        if rec is not None:
            t.write(rec)

    def _safe(self, what, fn, default=None):
        t0 = time.time()
        try:
            return fn()
        except Exception as e:
            self.mf.bump(f"mf_{what}_errors")
            if what not in self.warned:
                self.warned.add(what)
                msg = (f"[memframe] task {getattr(self.ctx, 'task_id', '?')}: memory {what} failed, the step goes on "
                       f"without it (counter mf_{what}_errors; V2_MF_STRICT=1 fails the task): {e!r}")
                print(msg[:600], file=sys.stderr, flush=True)
            if self.mf.STRICT:
                raise
            return default
        finally:
            self.mf.bump(f"mf_{what}_ms", 1000 * (time.time() - t0))


def factory(name):
    """memory server 的工厂：`mf_<后端>` -> 一个新实例。"""
    return MemframeMemorySystem(name[len("mf_"):])
