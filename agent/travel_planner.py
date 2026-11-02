import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .base_agent import BaseAgent

from env.env_systems.travel_planner_env.clients.base_client import BaseModelClient, ToolCall
from env.env_systems.travel_planner_env.tool_executor import ToolExecutor
from env.env_systems.travel_planner_env.tool_schemas import TOOLS
from env.env_systems.travel_planner_env.prompts import (
    AGENT_SYSTEM_PROMPT,
    AGENT_USER_PROMPT_TEMPLATE,
    HISTORY_TEMPLATE,
    BASE_PERSON_TEMPLATE,
)


@dataclass
class AgentStep:
    """Record of a single agent step"""
    step_idx: int
    thought: Optional[str]
    tool_calls: Optional[List[Dict]]
    tool_results: Optional[List[Dict]]
    final_output: Optional[str]
    raw_response: Optional[Dict] = None
    # use_step_memory 打开时，这一步循环内注入的记忆原文。合并 src/ 那轮以后，
    # 循环里的 memory_system.wrap_user_prompt() 没有任何地方记，落盘的轨迹里
    # 看不到它注入过什么。挂在步上而不是发一条 memory.wrap_user_prompt 事件：
    # trace_prep 的一轮是按那个事件开的，多发一次就多算一轮。
    step_memory: Optional[str] = None


@dataclass
class AgentResult:
    """Result from running the agent for one person"""
    name: str
    query: str
    final_plan: str
    scratchpad: List[AgentStep] = field(default_factory=list)
    total_steps: int = 0
    success: bool = True
    error_message: Optional[str] = None


class TravelPlannerAgent(BaseAgent):
    """
    Travel planning agent that uses tools to gather information
    and creates travel plans.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_steps: int = 30,
        db_path: str = None,
        system_prompt: str = None,
    ):
        super().__init__(model_name, temperature)
        self.client = self._create_client(model_name)
        self.executor = ToolExecutor(db_path=db_path)
        self.max_steps = max_steps

        self.system_prompt = system_prompt or AGENT_SYSTEM_PROMPT
        self.base_messages: List[Dict] = [{"role": "system", "content": self.system_prompt}]
        self.accumulated_plans: str = ""
        self.base_name: str = ""
        self.base_query: str = ""
        self.all_queries: List[str] = []
        self.previous_judgement: str = ""

        self._last_result: Optional[AgentResult] = None

        self._pending_name: Optional[str] = None
        self._pending_round_idx: Optional[int] = None
        self._pending_include_previous_plans: bool = True
        self._pending_memory_context: Optional[str] = None
        self._pending_memory_system = None

    def _create_client(self, model_name: str) -> BaseModelClient:
        model_lower = model_name.lower()

        if "gemini" in model_lower:
            from env.env_systems.travel_planner_env.clients.gemini_client import GeminiClient
            return GeminiClient(model_name=model_name)
        elif "claude" in model_lower or "anthropic" in model_lower:
            if os.environ.get("OPENAI_API_BASE"):
                from env.env_systems.travel_planner_env.clients.openai_client import OpenAIClient
                return OpenAIClient(model_name=model_name)
            else:
                from env.env_systems.travel_planner_env.clients.anthropic_client import AnthropicClient
                return AnthropicClient(model_name=model_name)
        else:
            from env.env_systems.travel_planner_env.clients.openai_client import OpenAIClient
            return OpenAIClient(model_name=model_name)

    def set_base_person(self, name: str, query: str, plan: str):
        self.base_name = name
        self.base_query = query
        self.all_queries = [f"{name}: {query}"]

        base_context = BASE_PERSON_TEMPLATE.format(
            base_name=name,
            base_query=query,
            base_plan=plan,
        )

        self.base_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": base_context},
        ]
        self.accumulated_plans = plan

    def add_judge_feedback(self, feedback: str):
        self.previous_judgement = feedback

    def prepare_for_person(
        self,
        name: str,
        round_idx: int,
        include_previous_plans: bool = True,
        memory_context: Optional[str] = None,
        memory_system=None,
    ):
        self._pending_name = name
        self._pending_round_idx = round_idx
        self._pending_include_previous_plans = include_previous_plans
        self._pending_memory_context = memory_context
        self._pending_memory_system = memory_system

    def act(self, prompt: str) -> str:
        name = self._pending_name or "User"
        round_idx = self._pending_round_idx or 1
        include_previous_plans = self._pending_include_previous_plans
        memory_context = self._pending_memory_context
        memory_system = self._pending_memory_system

        result = self.run_single_person(
            query=prompt,
            name=name,
            round_idx=round_idx,
            include_previous_plans=include_previous_plans,
            memory_context=memory_context,
            memory_system=memory_system,
        )
        return result.final_plan

    @property
    def last_result(self) -> Optional[AgentResult]:
        return self._last_result

    def get_scratchpad_dict(self, model_type: str = None) -> list:
        if self._last_result is None:
            return []
        return self._scratchpad_to_dict(self._last_result.scratchpad, model_type=model_type)

    # --- 给子类换范式用的两个钩子。默认实现就是原来那两行，行为不变。 ---

    def _get_response(self, messages: List[Dict], step: "AgentStep"):
        """发一轮请求。子类可以在这里加重试、改采样参数、检查回复形状。"""
        return self.client.chat_with_tools(messages, TOOLS)

    def _split_final_message(self, content: str, name: str):
        """把终局消息拆成 (推理, 计划)。

        默认不拆：原路径下模型只在最后一条写字，那条就是计划本身。返回 `None`
        表示"没有可分离的推理"，调用处据此不动 `step.thought`，所以原路径落盘
        形状逐字不变。
        """
        return None, content

    def _format_assistant_turn(self, response) -> Dict:
        """把模型这一轮的回复写回消息历史。

        默认走 client 的 `format_assistant_tool_calls`，它把 `content` 写成 None
        ——原路径下模型本来也不写文字，没有损失。范式要求模型边想边调工具时，
        这里得把 content 一起带上，否则它自己的推理下一轮就看不见了。
        """
        return self.client.format_assistant_tool_calls(response.tool_calls)

    def run_single_person(
        self,
        query: str,
        name: str,
        round_idx: int,
        include_previous_plans: bool = True,
        memory_context: str = None,
        memory_system=None,
        raw_query: bool = False,
    ) -> AgentResult:
        result = AgentResult(name=name, query=query, final_plan="")

        self.all_queries.append(f"{name}: {query}")

        messages = list(self.base_messages)

        if memory_context:
            messages.append({
                "role": "user",
                "content": f"Here is relevant context from memory that may help with this planning task:\n\n{memory_context}",
            })

        if round_idx > 1 and include_previous_plans:
            history_context = HISTORY_TEMPLATE.format(
                all_queries="\n".join(self.all_queries),
                previous_plan=self.accumulated_plans,
                judgement=self.previous_judgement or "",
            )
            messages.append({"role": "user", "content": history_context})

        if raw_query:
            user_message = query
        else:
            user_message = AGENT_USER_PROMPT_TEMPLATE.format(name=name, query=query)

        messages.append({"role": "user", "content": user_message})

        for step_idx in range(self.max_steps):
            step = AgentStep(
                step_idx=step_idx,
                thought=None,
                tool_calls=None,
                tool_results=None,
                final_output=None,
                raw_response=None,
            )

            response = self._get_response(messages, step)

            print(f"\n[DEBUG Step {step_idx}]")
            print(f"  content: {repr(response.content)[:200] if response.content else None}")
            print(f"  tool_calls: {response.tool_calls}")
            print(f"  raw_response type: {type(response.raw_response)}")

            step.thought = response.content

            if hasattr(response, 'raw_response'):
                if isinstance(response.raw_response, dict):
                    step.raw_response = response.raw_response
                else:
                    try:
                        step.raw_response = response.raw_response.to_dict() if hasattr(response.raw_response, 'to_dict') else None
                    except Exception:
                        step.raw_response = None

            if response.tool_calls:
                step.tool_calls = [
                    {"id": tc.id, "name": tc.name, "args": tc.arguments}
                    for tc in response.tool_calls
                ]

                messages.append(self._format_assistant_turn(response))

                step.tool_results = []
                for tc in response.tool_calls:
                    tool_result = self.executor.execute(tc.name, tc.arguments)

                    step.tool_results.append({
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "result": tool_result,
                    })

                    messages.append(self.client.format_tool_result(tc.id, tool_result, name=tc.name))

                if memory_system:
                    wrapped = memory_system.wrap_user_prompt(query)
                    step_memory = wrapped.split("</memory_context>")[0] + "</memory_context>"
                    step_memory = step_memory.strip()
                    step.step_memory = step_memory
                    messages.append({
                        "role": "user",
                        "content": f"[Step Memory] Updated context from memory:\n\n{step_memory}",
                    })

            else:
                preamble, plan = self._split_final_message(response.content, name)
                if preamble is not None:
                    # 推理和计划在同一条消息里。thought 留推理，final_plan 只留计划：
                    # env 侧 _parse_person_plan_from_result 用 re.search 取第一个
                    # `=== X's Plan ===`，推理里只要复述过这个标记，判分就会落在
                    # 那段议论文上，18 格全错。
                    step.thought = preamble
                step.final_output = plan
                result.scratchpad.append(step)
                result.final_plan = plan or ""
                result.total_steps = step_idx + 1
                break

            result.scratchpad.append(step)

        else:
            result.success = False
            result.error_message = f"Max steps ({self.max_steps}) reached without final output"
            result.total_steps = self.max_steps

        if result.final_plan:
            self.accumulated_plans += f"\n\n{result.final_plan}"

        self.previous_judgement = ""

        self._last_result = result
        return result

    def build_memory_entry(
        self,
        task: str,
        action: str,
        observation: Optional[Dict] = None,
        reward: Optional[float] = None,
    ) -> str:
        if self._last_result is None:
            return json.dumps({"task": task, "action": action}, ensure_ascii=False)

        scratchpad_dict = self._scratchpad_to_dict(self._last_result.scratchpad)
        chunk = {
            "name": self._last_result.name,
            "query": task,
            "scratchpad": scratchpad_dict,
            "final_plan": action,
        }
        judgement = None
        if observation and isinstance(observation, dict):
            judgement = observation.get("judgement")
        if judgement:
            chunk["judgement"] = judgement
        return json.dumps(chunk, ensure_ascii=False)

    def _scratchpad_to_dict(self, scratchpad: List[AgentStep], model_type: str = None) -> list:
        if model_type is None:
            model_type = "openai"
            if "gemini" in self.model_name.lower():
                model_type = "gemini"
            elif "claude" in self.model_name.lower() or "anthropic" in self.model_name.lower():
                model_type = "anthropic"

        result = []
        for step in scratchpad:
            step_dict = {
                'step_idx': step.step_idx,
                'thought': step.thought,
                'final_output': step.final_output,
            }
            if step.tool_calls:
                step_dict['tool_calls'] = step.tool_calls
            if step.tool_results:
                step_dict['tool_results'] = step.tool_results
            if model_type in ["gemini", "anthropic"]:
                if step.raw_response:
                    step_dict['raw_response'] = step.raw_response
            result.append(step_dict)
        return result

    def reset(self):
        self.base_messages = [{"role": "system", "content": self.system_prompt}]
        self.accumulated_plans = ""
        self.base_name = ""
        self.base_query = ""
        self.all_queries = []
        self.previous_judgement = ""
        self._last_result = None
        self._pending_name = None
        self._pending_round_idx = None
        self._pending_include_previous_plans = True
        self._pending_memory_context = None
        self._pending_memory_system = None

    def get_usage_stats(self) -> Dict:
        return self.client.get_usage_stats()
