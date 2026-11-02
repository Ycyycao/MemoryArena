import importlib
import os
from typing import Dict, Callable, NamedTuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv



def _lazy(module: str, cls: str) -> Callable[..., object]:
    """Import a memory system when it is created: each one brings its own dependencies."""
    def create(*args, **kwargs):
        mod = importlib.import_module(f"memory_systems.{module}")
        return getattr(mod, cls)(*args, **kwargs)
    return create


LongContextMemorySystem = _lazy("long_context", "LongContextMemorySystem")
MirixMemorySystem = _lazy("mirix", "MirixMemorySystem")
Mem0MemorySystem = _lazy("mem0", "Mem0MemorySystem")
LettaMemorySystem = _lazy("letta", "LettaMemorySystem")
RAGMemorySystem = _lazy("rag", "RAGMemorySystem")
MemoRAGMemorySystem = _lazy("memorag", "MemoRAGMemorySystem")
GraphRAGMemorySystem = _lazy("langchain_graphrag", "GraphRAGMemorySystem")
AMemMemorySystem = _lazy("amem", "AMemMemorySystem")
LightMemMemorySystem = _lazy("lightmem", "LightMemMemorySystem")
ReasoningBankMemorySystem = _lazy("reasoningbank", "ReasoningBankMemorySystem")
ZepMemorySystem = _lazy("zep", "ZepMemorySystem")

load_dotenv()

# ---------- Request schemas ----------
class InitializeRequest(BaseModel):
    user_id: str
    memory_system_name: str


class AddRequest(BaseModel):
    user_id: str
    chunk: str
    memory_system_name: str


class QueryRequest(BaseModel):
    user_id: str
    question: str
    memory_system_name: str

class ActRequest(BaseModel):
    user_id: str
    prompt: str
    memory_system_name: str

# ---------- Memory/Agent implementations (unified) ----------

def _mem0_factory():
    """默认走本地（OSS）mem0；MARENA_MEM0_HOSTED=1 退回上游的托管实现。

    托管版要 MEM0_API_KEY，而且把「用哪个模型抽事实」藏在服务端，对比时说不清差异
    来自方法还是模型。
    """
    if os.getenv("MARENA_MEM0_HOSTED") == "1":
        return Mem0MemorySystem()
    from memory_systems.mem0_local import Mem0LocalMemorySystem
    return Mem0LocalMemorySystem()


def _mem0_graph_factory():
    if os.getenv("MARENA_MEM0_HOSTED") == "1":
        return Mem0MemorySystem(enable_graph=True)
    raise NotImplementedError("mem0-g 需要外挂图数据库（Neo4j/Memgraph），本地实现没有")


def _simplemem_factory():
    """SimpleMem。不在上游 baseline 列表里，是我们加的。

    对接层在 SimpleMem 自己的仓库（`simplemem/integrations/memoryarena.py`）——它做的
    两件事（把 chunk 拆回带时间戳的对话轮、跳过 AnswerGenerator 只取检索结果）属于
    SimpleMem 自己的适配面。这里延迟 import，因为 lancedb / sentence-transformers
    只有跑 SimpleMem 时才需要装。
    """
    from memory.state import state_dir
    from simplemem.integrations.memoryarena import MemoryArenaMemory

    base = state_dir("simplemem")
    # EmbeddingModel 读的是 settings.EMBEDDING_MODEL，只能从环境变量给。
    # 指到本地权重目录，避免每次跑都去 HF 拉。
    embedding = os.getenv("MARENA_SIMPLEMEM_EMBEDDING")
    if embedding:
        os.environ.setdefault("EMBEDDING_MODEL", embedding)
    return MemoryArenaMemory(
        db_path=str(base / "lancedb"),
        model=os.getenv("MARENA_SIMPLEMEM_MODEL"),
        trace_dir=str(base),
    )


MEMORY_FACTORIES: Dict[str, Callable[[], object]] = {
    "mirix": MirixMemorySystem,
    "long_context": LongContextMemorySystem,
    "mem0": _mem0_factory,
    "mem0-g": _mem0_graph_factory,
    "letta": LettaMemorySystem,
    "rag": RAGMemorySystem,
    "memorag": MemoRAGMemorySystem,
    "graphrag": GraphRAGMemorySystem,
    "amem": AMemMemorySystem,
    "lightmem": LightMemMemorySystem,
    "reasoningbank": ReasoningBankMemorySystem, # You must add user_id when initializing ReasoningBankMemorySystem
    "zep": ZepMemorySystem,
    "simplemem": _simplemem_factory,
}


# ---------- FastAPI wiring ----------
app = FastAPI(title="Memory Agent Server")


class MemorySystemEntry(NamedTuple):
    name: str
    system: object


MEMORY_SYSTEMS: Dict[str, MemorySystemEntry] = {}


def _get_memory(user_id: str, memory_system: str):
    entry = MEMORY_SYSTEMS.get(user_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="User not initialized")
    if entry.name != memory_system:
        raise HTTPException(status_code=400, detail="Mismatched memory_system for user")
    return entry.system


@app.post("/memory/initialize")
def initialize(req: InitializeRequest):
    name = req.memory_system_name
    if name in {"bm25", "text-embedding-3-small"}:
        memory_system = RAGMemorySystem(retrieval_method=name)
    # elif name == "graphrag":
    #     if not os.getenv("GRAPHRAG_LOCAL_DIR"):
    #         raise HTTPException(status_code=400, detail="GRAPHRAG_LOCAL_DIR is required to initialize GraphRAG.")
    #     memory_system = GraphRAGMemorySystem(local_dir=os.getenv("GRAPHRAG_LOCAL_DIR"))
    elif name in {"reasoningbank"}:
        memory_system = ReasoningBankMemorySystem(user_id=req.user_id)
    else:
        factory = MEMORY_FACTORIES.get(name)
        if factory is None:
            raise HTTPException(status_code=400, detail=f"Unsupported memory_system: {name}")
        memory_system = factory()
    # 工厂是无参调用的，真正的 user_id 只有到这一刻才知道。实现如果需要它
    # （mem0_local 的 store 目录、SimpleMem 的 trace 文件名），在这里交给它。
    bind = getattr(memory_system, "bind_user_id", None)
    if bind is not None:
        bind(req.user_id)
    MEMORY_SYSTEMS[req.user_id] = MemorySystemEntry(name=name, system=memory_system)
    return {"status": "ok", "user_id": req.user_id, "memory_system_name": name}


@app.post("/memory/add")
def add(req: AddRequest):
    memory_system = _get_memory(req.user_id, req.memory_system_name)
    response = memory_system.add_chunk(req.chunk)
    outputs = {"status": "ok", "user_id": req.user_id}
    outputs['response'] = response
    return outputs


@app.post("/memory/wrap_user_prompt")
def wrap_user_prompt(req: QueryRequest):
    memory_system = _get_memory(req.user_id, req.memory_system_name)
    prompt = memory_system.wrap_user_prompt(req.question)
    return {"status": "ok", "user_id": req.user_id, "prompt": prompt}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
