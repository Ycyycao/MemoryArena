"""轨迹记录。

上游的主循环只在 INFO 日志里打印 prompt 和 action，落盘的 `generated_plan_*.json`
不含 memory 注入的完整原文，也不含 agent 内部的 tool 循环。Recorder 把每一步记下来，
由主循环在各阶段直接调用（不是从外面包一层——那样看不到循环内部）。

事件写进 `trajectory.json` 的 `trace` 数组，schema 是

    {"t": <unix 秒，3 位小数>, "event": "<名字>", ...该事件自己的字段}

下游有四个工具按这个 schema 读（`tool_script/export_trajectory.py`、
`export_travel_brief.py`、`export_agentdebug.py`、`trace-prep` skill），**加事件可以，
改已有事件的字段集不行**。

还有一条同样硬的：**每轮的事件基数不能变**。下游是按基数配对读的——
`export_travel_brief` 按轮分组，`trace_prep` 的一轮由 `memory.wrap_user_prompt` 开、
`env.step` 关。一轮多发一个 `agent.act`，两边都会静默错位，产出格式完好、内容全错
的报告。所以 agent 内部的多步推理嵌进 `scratchpad`，不发新事件；而一轮真的失败了
（没有产出）就记 `round.error`，也不补发假的 `agent.act` / `env.step`。

`tests/test_trajectory_schema.py` 执行这两条，并把 `results/` 下已有的真轨迹全验一遍。
"""

import time
from typing import Any, Dict, List


class Recorder:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def record(self, event: str, **fields: Any) -> None:
        self.events.append({"t": round(time.time(), 3), "event": event, **fields})

    def count(self, event: str) -> int:
        return sum(1 for e in self.events if e["event"] == event)


def scratchpad_to_json(result: Any) -> List[Dict[str, Any]]:
    """AgentStep 是 dataclass，转成能 json 序列化的形状。"""
    steps = []
    for step in getattr(result, "scratchpad", None) or []:
        entry = {
            "step_idx": step.step_idx,
            "thought": step.thought,
            "tool_calls": step.tool_calls,
            "tool_results": step.tool_results,
            "final_output": step.final_output,
        }
        # use_step_memory 打开时循环内注入的记忆。只在有值时出现，所以关着这个开关
        # 的 run 落盘形状跟以前逐字一样。
        step_memory = getattr(step, "step_memory", None)
        if step_memory:
            entry["step_memory"] = step_memory
        steps.append(entry)
    return steps
