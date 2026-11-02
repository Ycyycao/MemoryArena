"""ReAct 范式的 travel planner。

原路径（`TravelPlannerAgent`，`agent_mode="toolcall"`）跑的是原生 function-calling：
模型直接发 `tool_calls`，不写任何文字，所以落盘轨迹里 `scratchpad[].thought` 一直是
`null`。事后要判一轮为什么错——是注入的记忆没被用上，还是规划本身错——手里只有
「注入了什么」和「最终计划」，中间那一步是黑的。

这条路径不换传输层，仍然是 function-calling，只改两件事：

1. **换 prompt**：`REACT_AGENT_SYSTEM_PROMPT` 要求每次调工具前先写 `Thought:`，
   并且在有记忆注入时明说哪条用了、哪条没用、为什么。
2. **把 thought 带进消息历史**：上游三个 client 的 `format_assistant_tool_calls`
   都把 `content` 写死成 `None`。原路径下无所谓（本来就没文字），但要求模型边想
   边做以后，它自己上一轮的推理下一轮就看不见了，ReAct 链条断在这里。

外加一条兜底：模型发了 tool_calls 却没写 Thought 时补问一次。

`run_single_person` 不覆写，`AgentStep` / `AgentResult` 原样复用，所以
`harness.trace.scratchpad_to_json`、`build_memory_entry` 和 `run_travel.py` 用到的
全部 agent 表面都原样继承，轨迹的事件基数也不变——ReAct 的多步推理嵌在
`scratchpad` 里，不发新事件（`harness/trace.py` 的 schema 契约）。
"""

import re
from typing import Dict, List, Optional, Tuple

from .travel_planner import AgentStep, TravelPlannerAgent
from env.env_systems.travel_planner_env.tool_schemas import TOOLS
from env.env_systems.travel_planner_env.prompts import (
    REACT_AGENT_SYSTEM_PROMPT,
    REACT_MISSING_THOUGHT_NUDGE,
)


class ReActTravelPlannerAgent(TravelPlannerAgent):
    """边想边调工具：每个 tool-calling 步都带一段显式推理。"""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_steps: int = 30,
        db_path: str = None,
        system_prompt: str = None,
        require_thought: bool = True,
        max_thought_retries: int = 1,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_steps=max_steps,
            db_path=db_path,
            system_prompt=system_prompt or REACT_AGENT_SYSTEM_PROMPT,
        )
        self.require_thought = require_thought
        self.max_thought_retries = max_thought_retries
        # 补问了几次、补问完仍然没写的有几步。落进 stats_results/usage_stats.json，
        # 用来判断这条路径到底有没有生效——全是 thought_missing 的话，这轮 run 的
        # 推理数据是空的，不该拿去做归因。
        self.thought_nudges = 0
        self.thought_missing = 0

    def _get_response(self, messages: List[Dict], step: AgentStep):
        response = self.client.chat_with_tools(messages, TOOLS)

        if not self.require_thought:
            return response

        retries = 0
        while (
            response.tool_calls
            and not (response.content or "").strip()
            and retries < self.max_thought_retries
        ):
            retries += 1
            self.thought_nudges += 1
            # 补问在副本上进行：成功的话历史里只多出 thought，不多出这条催促，
            # 失败的话历史跟没补问过一样。两种情况下消息序列都保持干净。
            probe = list(messages) + [
                {"role": "user", "content": REACT_MISSING_THOUGHT_NUDGE}
            ]
            response = self.client.chat_with_tools(probe, TOOLS)

        if response.tool_calls and not (response.content or "").strip():
            self.thought_missing += 1
            print(
                f"[react] step {step.step_idx}: 补问 {retries} 次后仍然只有 tool_calls、"
                f"没有 Thought"
            )

        return response

    def _format_assistant_turn(self, response) -> Dict:
        """把这一轮的 thought 跟 tool_calls 一起写回历史。

        上游把 `content` 写成 `None`，模型看不到自己上一轮想了什么。
        """
        message = self.client.format_assistant_tool_calls(response.tool_calls)
        content = (response.content or "").strip()
        if content:
            message = {**message, "content": content}
        return message

    def _split_final_message(self, content: str, name: str) -> Tuple[Optional[str], str]:
        """终局消息里推理和计划混在一起，从计划标记处切开。

        prompt 不再要求模型"最后一条不要写 Thought"——那是在要求它压抑刚被要求做的
        事，实测它不照办，而且最后一步的推理（"为什么这份计划满足我认定的约束"）
        恰恰最该留下。允许它写，我们切。

        **倒着找。** 计划按约定是消息的最后一段，而 thought 里完全可能复述这个标记
        ——实测就出现过（`... start with "=== Eric's Plan ===". No "Thought:" line`）。
        正着找会切在引用上：env 侧 `_parse_person_plan_from_result` 用 `re.search`
        取第一个匹配、切到下一个 `===`，判分就落在两个标记之间那段议论文上，18 格
        全错，且跟规划质量无关。

        光倒着找还不够稳：模型也可能在计划**后面**再补一句复述。所以倒着扫的时候
        还要看标记后面跟的是不是 `Day N:`——真计划跟，只是提到标记的不跟。这个判据
        比"在不在行首"强：行首只是排版习惯，`Day N:` 是计划的实际结构。
        """
        if not content:
            return None, content

        def cut(at: int) -> Tuple[Optional[str], str]:
            preamble = content[:at].strip()
            return (preamble or None), content[at:].strip()

        escaped = re.escape(name)
        for pattern in (rf"===\s*{escaped}'s Plan\s*===",      # 指定人
                        r"===\s*[^=\n]+?'s Plan\s*==="):       # 名字写岔了的兜底
            matches = list(re.finditer(pattern, content))
            if not matches:
                continue
            for match in reversed(matches):
                if re.match(r"\s*Day\s*\d+\s*:", content[match.end():]):
                    return cut(match.start())
            # 一个都不像计划：退回最后一个，至少不会切在更早的引用上
            return cut(matches[-1].start())

        # 一个标记都没有：模型没按格式输出。保持原样交给下游，失败形态和原路径一致，
        # 不在这里发明行为。
        return None, content

    def get_usage_stats(self) -> Dict:
        stats = super().get_usage_stats()
        stats["agent_mode"] = "react"
        stats["thought_nudges"] = self.thought_nudges
        stats["thought_missing"] = self.thought_missing
        return stats
