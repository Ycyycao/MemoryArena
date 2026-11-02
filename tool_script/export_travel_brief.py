#!/usr/bin/env python3
"""把一次 travel run 的 trajectory.json 导成一份精简的 Markdown，只留接口层。

每一轮一节：这个人的请求 → memory 注入了什么 → agent 调了哪些工具 → 输出的计划 →
逐格和标准答案对（用上游 eval 同一种相似度）。不含 LLM 侧的 messages，那个看
`export_trajectory.py`。

    python tool_script/export_travel_brief.py results/simplemem/travel-simplemem-case1-8p
    python tool_script/export_travel_brief.py <run-dir> --tool-chars 400 --no-memory
"""

import argparse
import difflib
import json
import re
from pathlib import Path

SLOTS = ["transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"]
SIM_TH = 0.7


def group_rounds(trace):
    """顺序扫 trace，按轮把事件配成组。

    原先这里是 `zip(wraps, acts, steps)`——三个各自过滤出来的列表按位置配对。
    两个问题：memory=none 的 run 压根不发 `memory.wrap_user_prompt`，`wraps` 是空
    列表，zip 直接truncate 成 0 轮，整份 brief 只剩标题；而任何一类事件数量变了，
    zip 会静默错位，拿第 N 轮的计划去对第 M 轮的标准答案，产出一份格式完好、内容
    全错的报告，不报任何错。

    改成和 trace_prep 同一个状态机：`env.step` 收一轮，`memory.wrap_user_prompt`
    可以缺，一轮里多个 `agent.act` 时保留最后一个（终局计划就在最后那步）。
    """
    rounds, cur = [], {}
    for e in trace:
        name = e["event"]
        if name == "memory.wrap_user_prompt":
            cur["wrap"] = e
        elif name == "agent.act":
            cur["act"] = e
        elif name == "env.step":
            cur["step"] = e
            rounds.append(cur)
            cur = {}
    if cur.get("act"):          # 跑崩在 env.step 之前的最后一轮，也留下来
        rounds.append(cur)
    return rounds


def similarity(a, b):
    """和上游 eval.py 一样：小写、截到较短的长度、SequenceMatcher。"""
    if not a or not b:
        return 0.0
    a, b = str(a).strip().lower(), str(b).strip().lower()
    if a == "-" and b == "-":
        return 1.0
    m = min(len(a), len(b))
    return difflib.SequenceMatcher(None, a[:m], b[:m]).ratio() if m else 0.0


def parse_plan(text):
    """'Day N:' 块 + 'Key: value' 行 → {day: {slot: value}}。"""
    days = {}
    for m in re.finditer(r"Day\s+(\d+):(.*?)(?=Day\s+\d+:|\Z)", text, re.S):
        day, body = int(m.group(1)), m.group(2)
        cells = {}
        for line in body.strip().splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            cells[key.strip().lower()] = val.strip()
        days[day] = cells
    return days


def memory_block(wrapped):
    m = re.search(r"<memory_context>(.*?)</memory_context>", wrapped, re.S)
    return m.group(1).strip() if m else ""


def fmt_args(args):
    return ", ".join(f"{k}={v!r}" for k, v in (args or {}).items())


def first_line(s, limit):
    s = str(s).strip()
    if len(s) > limit:
        s = s[:limit].rstrip() + f" …(+{len(s) - limit} 字符)"
    return s.replace("\n", " ⏎ ")


def plan_table(days, gt_days=None):
    """计划表；给了 gt 就逐格标 ✅/❌，错的把标准答案写在下面一行。"""
    out = ["| day | " + " | ".join(SLOTS) + " |", "|---|" + "---|" * len(SLOTS)]
    wrong = 0
    for day in sorted(set(days) | set(gt_days or {})):
        cells = days.get(day, {})
        gt = (gt_days or {}).get(day, {})
        row = []
        for slot in SLOTS:
            val = cells.get(slot, "")
            if gt_days is None:
                row.append(val)
                continue
            ok = similarity(gt.get(slot, "-"), val) >= SIM_TH
            wrong += not ok
            row.append(("✅ " if ok else "❌ ") + val + ("" if ok else f"<br>→ 应为 `{gt.get(slot, '-')}`"))
        out.append(f"| {day} | " + " | ".join(row) + " |")
    return "\n".join(out), wrong


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--output", default=None, help="默认写到 <run-dir>/trace.brief.md")
    ap.add_argument("--tool-chars", type=int, default=160, help="每个工具返回值最多留多少字符")
    ap.add_argument("--no-memory", action="store_true", help="不打印注入的记忆正文，只给条数和字符数")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    traj = json.loads((run_dir / "trajectory.json").read_text(encoding="utf-8"))
    trace = traj["trace"]
    cfg = traj.get("config", {})
    out = []

    reset = next(e for e in trace if e["event"] == "env.reset")
    base = reset["base_person"]
    out.append(f"# {run_dir.name} · group {reset.get('group_id')} · base {base['name']} · "
               f"{reset.get('num_people')} 人 · memory={cfg.get('memory', {}).get('memory_system_name')}")
    out.append("")
    out.append(f"## base · {base['name']}")
    out.append("")
    out.append("> " + base["query"].replace("\n", "  \n> "))
    out.append("")
    gt_base = {d["days"]: {k: d.get(k, "-") for k in SLOTS} for d in base["daily_plans"]}
    out.append(plan_table(gt_base)[0])
    out.append("")

    rounds = group_rounds(trace)
    total_wrong = 0
    scored = 0
    for i, rnd in enumerate(rounds, 1):
        w, a, s = rnd.get("wrap"), rnd.get("act"), rnd.get("step")
        if a is None or s is None:
            continue
        gt = s["ground_truth"]
        gt_days = {d["days"]: {k: d.get(k, "-") for k in SLOTS} for d in gt["daily_plans"]}
        # memory=none 的 run 没有注入这一步
        mem = memory_block(w["wrapped_prompt"]) if w else ""
        n_items = (len(re.findall(r"^\[Context \d+\]", mem, re.M))
                   or (mem.count("\n\n") + 1 if mem else 0))
        scored += 1

        out.append(f"## 第 {i} 轮 · {a.get('person') or gt['name']}")
        out.append("")
        out.append("### 请求")
        out.append("")
        question = (w or a).get("question") or a.get("query") or ""
        out.append("> " + question.strip().replace("\n", "  \n> "))
        out.append("")
        out.append(f"### 注入的记忆 · {n_items} 条 · {len(mem):,} 字符"
                   + ("" if w else "（这条 run 没有 memory system）"))
        out.append("")
        if mem and not args.no_memory:
            out.append("```text")
            out.append(mem)
            out.append("```")
            out.append("")
        out.append(f"### 工具调用 · {sum(len(st.get('tool_calls') or []) for st in a.get('scratchpad') or [])} 次")
        out.append("")
        for st in a.get("scratchpad") or []:
            thought = st.get("thought")
            if thought and not st.get("final_output") and not thought.lstrip().startswith("==="):
                out.append(f"- 💭 {first_line(thought, 300)}")   # 最后一步的 thought 就是计划正文，不重复
            results = {r.get("tool_call_id"): r for r in st.get("tool_results") or []}
            for tc in st.get("tool_calls") or []:
                res = results.get(tc.get("id"), {}).get("result", "")
                out.append(f"- `{tc['name']}({fmt_args(tc.get('args'))})` → {first_line(res, args.tool_chars)}")
        out.append("")
        out.append(f"### 输出的计划 · {a.get('total_steps')} 步")
        out.append("")
        table, wrong = plan_table(parse_plan(a["final_plan"]), gt_days)
        total_wrong += wrong
        out.append(table)
        out.append("")
        fb = (s["result"].get("info") or {}).get("judgement")
        verdict = "全对" if wrong == 0 else f"错 {wrong} 格"
        out.append(f"**判定：{verdict}**" + (f"　env 反馈：`{first_line(fb, 200)}`" if fb else ""))
        out.append("")

    out.append(f"---\n合计错 {total_wrong} 格 / {scored * 18} 格")
    target = Path(args.output) if args.output else run_dir / "trace.brief.md"
    target.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{target}  ({len(out)} 行)")


if __name__ == "__main__":
    main()
