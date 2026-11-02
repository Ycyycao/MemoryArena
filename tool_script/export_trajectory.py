#!/usr/bin/env python3
"""把 trajectory.json 渲染成能线性读下来的 Markdown。

轨迹里有两层：

* **接口层** —— `memory.wrap_user_prompt` / `agent.act` / `env.step` / `memory.add`，
  也就是「memory 注入了什么、agent 返回了什么、env 怎么判」。
* **LLM 层** —— 每一次 chat completion 的完整 messages 和响应，来自网关落的
  `llm_calls.gateway.jsonl`（和 `trajectory.json` 同目录；改成按 run 切片之前叫
  `llm_calls.jsonl`，老 run 还是那个名字，两个都认）。网关覆盖三个进程，所以
  memory 侧的抽取和 env 侧的判分也在里面，不只是 agent。

这个脚本以 LLM 层为主线：把连续调用按 messages 前缀拼回一条对话，逐条消息铺成
step，system / user / assistant / tool 各占一步，env 判分和写回 memory 插在它们
发生的位置。读下来就是「模型一开始收到什么 → 它做了什么 → 环境回了什么」。

老轨迹（既没有 `llm_calls.jsonl`，`trace` 里也没有 `llm.call` 事件）会退回按轮
渲染的旧版式。`llm.call` 是网关之前的形式，由已删除的 `marena.llm_trace` 产生，
只覆盖 agent 进程；这里仍然认它，好让旧 run 还能渲染。

    python tool_script/export_trajectory.py results/<run-id>
    python tool_script/export_trajectory.py <dir> --max-chars 2000   # 长内容截断
    python tool_script/export_trajectory.py <dir> --fold 0           # 一律不折叠

system prompt 永远完整展示，不受 --max-chars / --fold 影响。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROLE_LABEL = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
}


# --------------------------------------------------------------------------- 工具

def load_gateway_calls(trajectory: Path) -> List[Dict[str, Any]]:
    """把网关的 `llm_calls.jsonl` 转成 exporter 内部的 `llm.call` 事件形状。

    渲染逻辑不动，只在入口做一次适配：网关存的是 OpenAI 的原始请求/响应体，
    而渲染函数期望 `tools` 是名字列表、`response` 是拍平过的。
    """
    path = trajectory.parent / "llm_calls.gateway.jsonl"
    if not path.exists():
        path = trajectory.parent / "llm_calls.jsonl"      # 按 run 切片之前的名字
    if not path.exists():
        return []

    events: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        request = record.get("request") or {}
        if not isinstance(request, dict) or "messages" not in request:
            continue                      # embeddings 之类没有对话，跳过
        body = record.get("response") or {}
        choice = ((body.get("choices") or [{}])[0]) if isinstance(body, dict) else {}
        message = choice.get("message") or {}
        events.append({
            "t": record.get("t"),
            "event": "llm.call",
            "duration": record.get("duration"),
            "role": record.get("role"),
            "request": {
                "model": request.get("model"),
                "messages": request.get("messages") or [],
                "tools": [(tool.get("function") or {}).get("name")
                          for tool in (request.get("tools") or [])],
                "temperature": request.get("temperature"),
            },
            "response": {
                "content": message.get("content"),
                "tool_calls": [{"id": c.get("id"),
                                "name": (c.get("function") or {}).get("name"),
                                "arguments": (c.get("function") or {}).get("arguments")}
                               for c in (message.get("tool_calls") or [])],
                "finish_reason": choice.get("finish_reason"),
                "usage": record.get("usage"),
            },
        })
    return events


def find_trajectory(target: Path) -> Path:
    """接受 trajectory.json 本身，或它上面任意一级目录。"""
    if target.is_file():
        return target
    matches = sorted(target.rglob("trajectory.json"))
    if not matches:
        raise SystemExit(f"{target} 下面没有 trajectory.json")
    if len(matches) > 1:
        print(f"找到 {len(matches)} 份，用第一份：{matches[0]}", file=sys.stderr)
    return matches[0]


def block(text: Any, max_chars: int, lang: str = "") -> str:
    body = "" if text is None else str(text)
    if not body.strip():
        body = "（空）"
    if max_chars and len(body) > max_chars:
        body = body[:max_chars] + f"\n\n……（截断，原文共 {len(str(text))} 字符）"
    fence = "````" if "```" in body else "```"
    return f"{fence}{lang}\n{body}\n{fence}"


def maybe_fold(text: Any, max_chars: int, fold: int, summary: str, lang: str = "") -> str:
    """短的直接铺开，长的折起来——线性阅读时不希望被一屏工具输出打断。"""
    body = block(text, max_chars, lang)
    if fold and len(str(text or "")) > fold:
        return f"<details>\n<summary>{summary}（{len(str(text))} 字符）</summary>\n\n{body}\n\n</details>"
    return body


def round_title(question: Any) -> str:
    """给这一轮取个能认出来的标题。

    math 的题面是 `### BACKGROUND: ... ### PROBLEM: ...`，直接取首行会拿到标签；
    有 PROBLEM 段就从那里取。
    """
    text = str(question or "")
    if "### PROBLEM:" in text:
        text = text.split("### PROBLEM:", 1)[1]
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:70]
    return ""


def message_key(message: Dict[str, Any]) -> Tuple:
    """判断两次调用是不是同一条对话，只看前缀是否一致。"""
    calls = tuple(c.get("id") for c in message.get("tool_calls") or [])
    return (message.get("role"), str(message.get("content"))[:300], calls)


def common_prefix(a: List[Tuple], b: List[Tuple]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# --------------------------------------------------------------------------- 渲染

class Transcript:
    """一条线性的 step 流。"""

    def __init__(self, max_chars: int, fold: int) -> None:
        self.lines: List[str] = []
        self.step = 0
        self.max_chars = max_chars
        self.fold = fold
        self.tool_names: Dict[str, Any] = {}

    def heading(self, text: str) -> None:
        self.lines += ["", f"## {text}", ""]

    def emit(self, label: str, body: str) -> None:
        self.step += 1
        self.lines += [f"### step {self.step} · {label}", "", body, ""]

    def message(self, message: Dict[str, Any], suffix: str = "") -> None:
        role = ROLE_LABEL.get(message.get("role"), str(message.get("role")))
        label = role
        if role == "tool":
            # OpenAI 的 tool 消息只带 tool_call_id，工具名要回查发起它的那次 assistant
            name = message.get("name") or self.tool_names.get(message.get("tool_call_id"))
            if name:
                label = f"tool · {name}"
        for call in message.get("tool_calls") or []:
            if call.get("id"):
                self.tool_names[call["id"]] = call.get("name")
        if suffix:
            label = f"{label} {suffix}"

        parts: List[str] = []
        if message.get("content"):
            if role == "system":
                # system prompt 是「模型被告知它是谁」，永远完整展示：
                # 它不长，而且是读懂后面每一步的前提，不受 --max-chars / --fold 影响
                parts.append(block(message["content"], 0))
            else:
                parts.append(maybe_fold(message["content"], self.max_chars, self.fold, f"展开 {role} 内容"))
        for call in message.get("tool_calls") or []:
            arguments = call.get("arguments")
            try:
                arguments = json.dumps(json.loads(arguments), ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                pass
            parts.append(f"**调用 `{call.get('name')}`**\n\n" + block(arguments, self.max_chars, "json"))
        self.emit(label, "\n\n".join(parts) if parts else "（无内容）")


def render_conversation_view(data: Dict[str, Any], max_chars: int, fold: int) -> List[str]:
    trace = data["trace"]
    transcript = Transcript(max_chars, fold)
    conversations: List[List[Tuple]] = []
    call_no = 0
    round_no = 0
    seen_tools: List[str] = []

    for event in trace:
        name = event["event"]

        if name == "memory.wrap_user_prompt":
            round_no += 1
            transcript.heading(f"第 {round_no} 轮 —— {round_title(event.get('question'))}")
            continue

        if name == "env.reset":
            transcript.heading("开局")
            transcript.emit("env.reset", block(json.dumps(
                {k: v for k, v in event.items() if k not in {"t", "event"}},
                ensure_ascii=False, indent=2), max_chars, "json"))
            continue

        if name == "llm.call":
            call_no += 1
            request = event.get("request") or {}
            messages = request.get("messages") or []
            keys = [message_key(m) for m in messages]

            best_index, best_len = None, 0
            for index, known in enumerate(conversations):
                length = common_prefix(known, keys)
                if length > best_len:
                    best_index, best_len = index, length
            if best_index is None:
                conversations.append([])
                best_index, best_len = len(conversations) - 1, 0

            branch = "" if best_index == 0 else f"（对话 {chr(ord('A') + best_index)}）"
            for message in messages[best_len:]:
                transcript.message(message, suffix=branch)

            response = event.get("response") or {}
            tools = [t for t in (request.get("tools") or []) if t]
            role = event.get("role")
            role_tag = f"{role} 侧，" if role and role != "agent" else ""
            meta = f"（LLM 调用 #{call_no}，{role_tag}{request.get('model')}，{event.get('duration')}s"
            if tools:
                # 工具清单每轮都一样，只在第一次列全，之后报个数
                if len(tools) <= 3 or not seen_tools:
                    seen_tools.extend(tools)
                    meta += f"，可用工具 {'、'.join(tools)}）"
                else:
                    meta += f"，可用工具 {len(tools)} 个）"
            else:
                meta += "）"
            assistant = {
                "role": "assistant",
                "content": response.get("content"),
                "tool_calls": response.get("tool_calls"),
            }
            transcript.message(assistant, suffix=f"{branch}{meta}")

            conversations[best_index] = keys + [message_key(assistant)]
            continue

        if name == "env.step":
            result = event.get("result") or {}
            observation = result.get("observation") or {}
            info = result.get("info") or {}
            body = [f"`reward = {result.get('reward')}`"]
            if observation.get("judge_result") is not None:
                body.append(f"judge 输出：`{observation['judge_result']!r}`")
            if info.get("judgement"):
                body.append("**判词：**\n\n" + block(info["judgement"], max_chars))
            if event.get("ground_truth") is not None:
                ground_truth = event["ground_truth"]
                text = ground_truth if isinstance(ground_truth, str) else json.dumps(
                    ground_truth, ensure_ascii=False, indent=2)
                body.append("**标准答案：**\n\n" + maybe_fold(
                    text, max_chars, fold, "展开标准答案",
                    "" if isinstance(ground_truth, str) else "json"))
            transcript.emit("env 判分", "\n\n".join(body))
            continue

        if name == "memory.add":
            transcript.emit("写回 memory", maybe_fold(
                event.get("chunk"), max_chars, fold, "展开写回的 chunk"))
            continue

        # agent.act / memory.wrap 的内容在 llm.call 的 messages 里已经完整出现，跳过

    return transcript.lines


def render_header(data: Dict[str, Any], trace: List[Dict[str, Any]]) -> List[str]:
    cfg = data.get("config", {})
    agent_cfg = cfg.get("agent", {})
    steps = [e for e in trace if e["event"] == "env.step"]
    correct = sum(1 for e in steps if (e.get("result") or {}).get("reward"))
    calls = [e for e in trace if e["event"] == "llm.call"]

    out = [f"# 轨迹：{data.get('case', '?')}", ""]
    out += ["| 项 | 值 |", "| --- | --- |"]
    out.append(f"| config | `{data.get('config_path', '?')}` |")
    out.append(f"| memory system | `{cfg.get('memory', {}).get('memory_system_name', '?')}` |")
    out.append(f"| agent | `{agent_cfg.get('model_name', '?')}` |")
    out.append(f"| 轮次 | {data.get('num_queries', len(steps))} |")
    out.append(f"| 判定 | {correct} 对 / {len(steps) - correct} 错 |")
    if calls:
        seconds = sum(e.get("duration") or 0 for e in calls)
        out.append(f"| LLM 调用 | {len(calls)} 次，累计 {seconds:.1f}s |")
    if trace:
        out.append(f"| 总耗时 | {trace[-1]['t'] - trace[0]['t']:.1f}s |")
    out.append("")
    return out


def render(data: Dict[str, Any], max_chars: int, fold: int) -> str:
    trace = data["trace"]
    out = render_header(data, trace)
    if not any(e["event"] == "llm.call" for e in trace):
        out += [
            "> 这份轨迹旁边没有 `llm_calls.jsonl`，`trace` 里也没有 `llm.call` 事件，",
            "> 只能看到接口层。重跑一次（网关会落 `llm_calls.jsonl`）就有完整对话流。",
            "",
        ]
    out += render_conversation_view(data, max_chars, fold)
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("target", help="trajectory.json 本身，或它上面任意一级目录")
    parser.add_argument("-o", "--output", default=None,
                        help="输出路径，默认写到 trajectory.json 旁边的 trajectory.md")
    parser.add_argument("--max-chars", type=int, default=0,
                        help="单条内容的字符上限，0 表示不截断；system prompt 不受影响，永远完整")
    parser.add_argument("--fold", type=int, default=3000,
                        help="超过这个长度的内容折进 <details>，0 表示一律不折；system prompt 不受影响")
    args = parser.parse_args()

    path = find_trajectory(Path(args.target).expanduser())
    data = json.loads(path.read_text(encoding="utf-8"))

    # 网关的记录和接口层事件按时间合流，渲染时就是一条线
    gateway_calls = load_gateway_calls(path)
    if gateway_calls:
        data["trace"] = sorted(data["trace"] + gateway_calls,
                               key=lambda e: e.get("t") or 0)
    markdown = render(data, args.max_chars, args.fold)

    out_path = Path(args.output).expanduser() if args.output else path.with_suffix(".md")
    out_path.write_text(markdown, encoding="utf-8")
    print(f"{path}  →  {out_path}  ({len(markdown)} 字符)")


if __name__ == "__main__":
    main()
