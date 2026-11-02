"""
OpenAI Client - Implementation of BaseModelClient for OpenAI-compatible APIs

Calls go through litellm rather than the openai SDK, so the same client reaches
OpenAI, Gemini, Anthropic, Bedrock, Vertex and the rest by model name alone
(`gemini/gemini-2.5-flash`, `anthropic/claude-...`). litellm returns an
OpenAI-shaped response, so everything downstream of the call is unchanged.

The class name stays OpenAIClient because `travel_planner.py` selects it by
`backend: openai`; it is no longer OpenAI-specific.
"""

import os
import json
import random
import time
from typing import List, Dict, Optional

import litellm

from .base_client import BaseModelClient, ModelResponse, ToolCall
from ..cost_tracker import CostTracker


class OpenAIClient(BaseModelClient):
    """
    OpenAI client with Tools API support.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None, base_url: str = None):
        super().__init__(model_name)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        # OPENAI_BASE_URL is the name the openai SDK itself uses; OPENAI_API_BASE
        # is kept as a fallback because the original code read only that one.
        self.api_base = (base_url
                         or os.environ.get("OPENAI_BASE_URL")
                         or os.environ.get("OPENAI_API_BASE"))
        self.cost_tracker = CostTracker(model_name)
        self.retry_count = 0

    # 一条 run 跑几十分钟，中间任何一个请求抛异常就掀掉整条。实测中转站在几条 run
    # 并行时会返回 400 ContentPolicyViolationError，内容是一段旅行推理加一张航班表
    # ——不是真的内容策略，是它自己的限速/过载误报（错误串尾巴带 ip:port）。所以
    # 这里退避重试。
    #
    # 400 也重试：能区分"真违规"和"中转站误报"的信息我们手里没有，而真违规重试三次
    # 也只是多花三次的钱、结果一样。重试次数会打到 stderr 并计数（retry_count），
    # 跑完在 usage_stats.json 里看得到——**不要让重试把限速信号吃掉**，那个信号是
    # 判断"要不要少跑几条并行"的唯一依据。
    RETRY_ATTEMPTS = int(os.getenv("MARENA_LLM_RETRIES", "3"))
    RETRY_BASE_SECONDS = float(os.getenv("MARENA_LLM_RETRY_BASE", "4"))

    def _call(self, **kwargs):
        """One place that talks to litellm, so credentials are not repeated."""
        last = None
        for attempt in range(1, self.RETRY_ATTEMPTS + 1):
            try:
                return litellm.completion(
                    api_key=self.api_key, api_base=self.api_base, **kwargs)
            except Exception as exc:
                last = exc
                self.retry_count += 1
                if attempt == self.RETRY_ATTEMPTS:
                    break
                # 指数退避 + 抖动。限速要的是真等，几百毫秒没用；抖动是为了几条 run
                # 并行时不要一起醒来再一起撞上去。
                delay = self.RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                delay *= 0.5 + random.random()
                print(f"[retry] 第 {attempt}/{self.RETRY_ATTEMPTS} 次失败，{delay:.1f}s 后重试: "
                      f"{type(exc).__name__}: {str(exc)[:200]}")
                time.sleep(delay)
        raise last
    
    def chat_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 32768
    ) -> ModelResponse:
        """
        Send messages to OpenAI with tools enabled.
        """
        try:
            if self.model_name.startswith('gpt-5'):
                # gpt-5 系列: 用 max_completion_tokens, 不传 temperature
                response = self._call(
                    model=self.model_name,
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice="auto" if tools else None,
                    max_completion_tokens=max_tokens
                )
            else:
                # gpt-4 / Claude / 其他: 用 temperature + max_tokens
                response = self._call(
                    model=self.model_name,
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice="auto" if tools else None,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            # Track usage. A third-party OpenAI-compatible endpoint can return
            # null token counts (gpt-5* through a proxy does), and `int += None`
            # raises TypeError, so a missing count counts as 0.
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens or 0
                completion_tokens = response.usage.completion_tokens or 0
                self.total_input_tokens += prompt_tokens
                self.total_output_tokens += completion_tokens
                self.cost_tracker.add_usage(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens
                )
            
            # Parse response
            message = response.choices[0].message
            
            # Extract tool calls if any
            tool_calls = None
            if message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                    )
                    for tc in message.tool_calls
                ]
            
            return ModelResponse(
                content=message.content,
                tool_calls=tool_calls,
                raw_response=response
            )
            
        except Exception as e:
            print(f"LLM API error: {e}")
            raise
    
    def format_tool_result(self, tool_call_id: str, result: str, name: str = None) -> Dict:
        """
        Format tool result as a message for OpenAI.
        Note: OpenAI doesn't need the name parameter, but we accept it for API consistency.
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        }
    
    def format_assistant_tool_calls(self, tool_calls: List[ToolCall]) -> Dict:
        """
        Format assistant's tool calls as a message.
        This is needed to maintain proper conversation history.
        """
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                    }
                }
                for tc in tool_calls
            ]
        }
    
    def get_usage_stats(self) -> Dict:
        """Get usage stats using CostTracker"""
        cost_info = self.cost_tracker.get_cost()
        return {
            'total_input_tokens': cost_info['input_tokens'],
            'total_output_tokens': cost_info['output_tokens'],
            'total_cost': cost_info['total_cost'],
            # 重试次数要露出来：它是"中转站在限速"的唯一可见证据
            'llm_retries': self.retry_count,
        }


# For testing
if __name__ == "__main__":
    client = OpenAIClient(model_name="gpt-4o-mini")
    
    messages = [
        {"role": "system", "content": "You are a travel planning assistant."},
        {"role": "user", "content": "Search for flights from New York to Los Angeles on 2024-03-15"}
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "FlightSearch",
                "description": "Search for flights",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string"},
                        "destination": {"type": "string"},
                        "date": {"type": "string"}
                    },
                    "required": ["origin", "destination", "date"]
                }
            }
        }
    ]
    
    response = client.chat_with_tools(messages, tools)
    print(f"Content: {response.content}")
    print(f"Tool calls: {response.tool_calls}")
    print(f"Usage: {client.get_usage_stats()}")
