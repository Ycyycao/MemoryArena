"""
Travel Planner - Memory-Augmented Agent Benchmark
Follows the same pattern as example_math.py:
    env_client.reset → memory.wrap → agent.act → env_client.step → agent.build_memory_entry → memory.add

Requires:
    - Environment server running: python env/env_server.py
    - Memory server running:      python memory/server.py  (if using a memory system)
"""

import os
import re
import json
import time
import uuid
import argparse
from pathlib import Path

from tqdm import tqdm

from agent.travel_planner import TravelPlannerAgent
from agent.travel_planner_react import ReActTravelPlannerAgent

# 范式。默认 toolcall = 上游原样的原生 function-calling；react 换 prompt 要求模型
# 每步先写 Thought，并把那段推理带进消息历史（见 agent/travel_planner_react.py）。
# 两条路径产出的 AgentResult 形状相同，轨迹事件基数也相同。
AGENT_MODES = {
    "toolcall": TravelPlannerAgent,
    "react": ReActTravelPlannerAgent,
}

# 连续多少轮失败就中止这一组。1 太敏感（偶发 400 是常态），大了只是烧钱。
MAX_CONSECUTIVE_ROUND_FAILURES = int(os.getenv("MARENA_MAX_ROUND_FAILURES", "2"))
from env.env_client import EnvironmentClient
from env.env_systems.travel_planner_env.data_loader import load_travel_data
from env.env_systems.travel_planner_env.combination import combine
from env.env_systems.travel_planner_env.eval import evaluate
from harness import config as config_mod
from harness import llm_journal, runctx
from harness.trace import Recorder, scratchpad_to_json


def get_memory_system(memory_system_name: str, user_id: str, server_url: str = "http://0.0.0.0:8000"):
    if memory_system_name == "none":
        return None
    from memory.client import MemoryClient
    return MemoryClient(user_id=user_id, memory_system_name=memory_system_name, base_url=server_url)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------



def parse_all_plans(result_text, queries):
    all_results = []
    pattern = r'===\s*([^=]+?)\'s Plan\s*==='
    parts = re.split(pattern, result_text)

    name_plan_pairs = {}
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            name = parts[i].strip()
            plan = parts[i + 1].strip()
            name_plan_pairs[name] = plan

    for idx, round_item in enumerate(queries, start=1):
        name = round_item.get('name', f'Person{idx}')
        query = round_item['query']
        result = name_plan_pairs.get(name, "")
        all_results.append({
            'person_idx': idx,
            'name': name,
            'query': query,
            'result': result,
        })

    return all_results


def format_person_plan(name, daily_plans):
    lines = [f"=== {name}'s Plan ==="]
    for day in daily_plans:
        day_idx = day.get('days') or day.get('day')
        lines.append(f"Day {day_idx}:")
        lines.append(f"Current City: {day.get('current_city', '-')}")
        lines.append(f"Transportation: {day.get('transportation', '-')}")
        lines.append(f"Breakfast: {day.get('breakfast', '-')}")
        lines.append(f"Attraction: {day.get('attraction', '-')}")
        lines.append(f"Lunch: {day.get('lunch', '-')}")
        lines.append(f"Dinner: {day.get('dinner', '-')}")
        lines.append(f"Accommodation: {day.get('accommodation', '-')}")
        lines.append("")
    return "\n".join(lines)


def build_args(cfg):
    """把 config dict 摊成跟 CLI flags 一样的 namespace。

    `cfg` 是 `harness.config.load()` 解析过占位符的结果——原先这里自己 json.load，
    拿到的是没解析的 `<OPENAI_BASE_URL>` 这种字面量。
    """
    agent = cfg.get("agent", {})
    memory = cfg.get("memory", {})
    env = cfg.get("env", {})
    task_specific = cfg.get("task_specific", {})
    output = cfg.get("output", {})

    # Set API key / base_url as environment variables
    backend = agent.get("backend", "openai")
    api_key = agent.get("api_key", "")
    if backend == "gemini":
        os.environ.setdefault("GOOGLE_API_KEY", api_key)
        os.environ.setdefault("GEMINI_API_KEY", api_key)
    else:
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        base_url = agent.get("base_url", "")
        if base_url:
            os.environ.setdefault("OPENAI_API_BASE", base_url)

    ns = argparse.Namespace(
        model_name=agent.get("model_name", "gpt-4.1-mini"),
        memory_system=memory.get("memory_system_name", "none"),
        use_step_memory=memory.get("use_step_memory", False),
        server_url=memory.get("server_url", "http://0.0.0.0:8000"),
        env_server_url=env.get("env_server_url", "http://0.0.0.0:8001"),
        judgement_mode=env.get("env_config", {}).get("judgement_mode", "none"),
        max_steps=task_specific.get("max_steps", 30),
        agent_mode=agent.get("agent_mode", "toolcall"),
        output_dir=output.get("output_dir", "./"),
        log_dir=output.get("log_dir", ""),
        global_csv=output.get("global_csv", ""),
    )
    return ns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=None,
                        help="config 路径；相对路径按 $MARENA_CONFIGS 解析")
    parser.add_argument("--model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--judgement_mode", type=str, default="hint",
                        choices=["answer", "hint", "none"])
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--memory_system", type=str, default="none",
                        choices=["none", "long_context", "mirix", "letta", "mem0", "mem0-g",
                                 "rag", "bm25", "text-embedding-3-small", "cognee", "memorag",
                                 "graphrag", "lightmem", "amem", "reasoningbank", "zep",
                                 "simplemem"])
    parser.add_argument("--server_url", type=str, default="http://0.0.0.0:8000")
    parser.add_argument("--env_server_url", type=str, default="http://0.0.0.0:8001")
    # 跑单条 case 用的
    parser.add_argument("--case-index", type=int, default=None,
                        help="第几组（按数据集顺序）；不给就跑全部 270 组")
    parser.add_argument("--case-id", type=int, default=None,
                        help="按 group id 选，优先于 --case-index")
    parser.add_argument("--max-people", type=int, default=0,
                        help="只跑这组前 N 个加入的人，0 表示全部（一组 5~8 人）")
    parser.add_argument("--run-id", default=None,
                        help="输出落到 $MARENA_RESULTS/<memory-system>/<run-id>/，默认取 config 文件名")
    parser.add_argument("--all", action="store_true",
                        help="跑全部 270 组。不给 --case-index / --case-id 时必须显式给这个")
    parser.add_argument("--list", action="store_true", help="只列出前 20 组，不跑")
    parser.add_argument("--agent-mode", default=None, choices=sorted(AGENT_MODES),
                        help="agent 范式；不给就用 config 里的 agent.agent_mode，"
                             "config 也没有就是 toolcall")
    raw_args = parser.parse_args()

    if raw_args.list:
        for group in load_travel_data()[:20]:
            base = group["base_person"]
            print(f"id={group['id']:>4}  base={base['name']:<10} 加入 {len(group['questions'])} 人")
        return

    cfg = None
    config_path = None
    run_id = raw_args.run_id
    if raw_args.config:
        config_path = config_mod.resolve_path(raw_args.config)
        cfg = config_mod.load(config_path)
        run_id = run_id or config_path.stem
        output_dir = config_mod.run_dir(cfg, run_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg["output"]["output_dir"] = str(output_dir)
        if raw_args.max_steps != parser.get_default("max_steps"):
            cfg["task_specific"]["max_steps"] = raw_args.max_steps
        # 范式先落回 cfg 再摊成 args：trajectory.json 存的 "config" 就是这个 cfg，
        # 从命令行给的 --agent-mode 不写回去的话，事后看轨迹认不出这条 run 跑的是
        # 哪条路径。config 里也没写就显式补 toolcall，别让它只靠默认值隐含。
        if raw_args.agent_mode:
            cfg.setdefault("agent", {})["agent_mode"] = raw_args.agent_mode
        cfg.setdefault("agent", {}).setdefault("agent_mode", "toolcall")
        args = build_args(cfg)
        # 解析过占位符的 config 落一份，方便事后知道这条 run 到底用了什么
        (output_dir / "resolved-config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        args = raw_args
        args.agent_mode = raw_args.agent_mode or "toolcall"
        run_id = run_id or "cli"

    os.makedirs(args.output_dir, exist_ok=True)

    # 选 case。全量要显式说：270 组 × 5~8 人 × 每人一轮 ReAct，漏给一个 flag
    # 就能烧掉几百刀，所以这里不给"什么都不传就跑全部"留默认值。
    only_group_id = raw_args.case_id
    if only_group_id is None and raw_args.case_index is not None:
        only_group_id = load_travel_data()[raw_args.case_index]["id"]
    if only_group_id is None and not raw_args.all:
        parser.error(
            "要跑哪些组？--case-index N / --case-id N 跑一组，--all 跑全部 270 组。"
            "--list 先看看有哪些组。"
        )

    # LLM 调用记录在网关（HTTP 层），每条带 request_id；跑完按 run-id 前缀切进这条 run
    # 的目录。agent 进程自己的调用打 `<run-id>__agent`，server 那边的靠请求体里带 run-id
    # 前缀的 user_id / task_id，见 harness.runctx。
    os.environ.setdefault("MARENA_LLM_ROLE", "agent")
    runctx.REQUEST_ID.set(f"{run_id}__agent")
    runctx.tag_openai_sdk()
    runctx.propagate_context_to_threads()
    journal_marks = llm_journal.marks()

    recorder = Recorder()
    try:
        _run_all(args, recorder, run_id, only_group_id, raw_args.max_people)
    finally:
        output_dir = Path(args.output_dir)
        harvested = llm_journal.harvest(journal_marks, output_dir, run_id)
        if harvested:
            detail = "、".join(f"{role} {n}" for role, n in sorted(harvested.items()))
            print(f"llm_calls.<role>.jsonl  {detail}")
        trajectory = output_dir / "trajectory.json"
        trajectory.write_text(json.dumps({
            "config_path": str(config_path) if config_path else "",
            "config": cfg if cfg is not None else vars(args),
            "case": f"group_{only_group_id}" if only_group_id is not None else "all",
            "num_queries": recorder.count("agent.act"),
            "trace": recorder.events,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\ntrajectory.json  {trajectory}")


def _run_all(args, recorder, run_id, only_group_id, max_people):
    """跑 config 指定的 group（`only_group_id` 不为 None 时只跑那一组）。"""
    log_dir = getattr(args, 'log_dir', '') or ''
    global_csv = getattr(args, 'global_csv', '') or ''
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("SOLE PLANNING - Agent Mode")
    print("=" * 60)
    print(f"Model:         {args.model_name}")
    print(f"Data source:   $MEMORYARENA_DATA 本地副本，没有则回落 HuggingFace")
    print(f"Output dir:    {args.output_dir}")
    print(f"Judgement:     {args.judgement_mode}")
    print(f"Memory system: {args.memory_system}")
    print(f"Memory URL:    {args.server_url}")
    print(f"Env URL:       {args.env_server_url}")
    print(f"Max steps:     {args.max_steps}")
    print("=" * 60 + "\n")

    query_data_list = load_travel_data(only_group_id=only_group_id)
    print(f"Loaded {len(query_data_list)} data items")
    if query_data_list:
        print('First item keys:', list(query_data_list[0].keys()))

    # db_path 不给的话上游默认指向 MemoryArena 内部；指到 data/travelplanner/，
    # 数据不落进代码树。
    db_path = os.getenv("TRAVELPLANNER_DB")
    if not db_path:
        raise SystemExit(
            "环境变量 TRAVELPLANNER_DB 没有设置。它要指向含 flights/ restaurants/ 等六个"
            "子目录的数据库根目录，见 docs/conventions/memoryarena-runbook.md。"
        )
    agent_mode = getattr(args, "agent_mode", "toolcall")
    agent = AGENT_MODES[agent_mode](
        model_name=args.model_name,
        max_steps=args.max_steps,
        db_path=db_path,
    )
    print(f"agent_mode={agent_mode}  ({type(agent).__name__})")

    # task_id 加 run-id 前缀，env server 里 judge 的 LLM 记录才知道自己属于哪条 run
    # （见 harness.runctx）。
    task_id = f"{run_id}__{uuid.uuid4()}"
    env_client = EnvironmentClient(
        task_id=task_id,
        env_name="travel_planner",
        base_url=args.env_server_url,
        env_config={
            "judgement_mode": args.judgement_mode,
        },
    )

    start_time = time.time()

    for query_data in tqdm(query_data_list):
        data_idx = query_data['id']
        output_file = os.path.join(args.output_dir, f'generated_plan_{data_idx}.json')
        if os.path.exists(output_file):
            print(f"[Data {data_idx}] Output file already exists, skipping: {output_file}")
            continue

        # --- Reset env for this group ---
        obs = env_client.reset(seed=data_idx)
        if max_people:
            obs["questions"] = obs.get("questions", [])[:max_people]
            obs["answers"] = obs.get("answers", [])[:max_people]
        recorder.record(
            "env.reset",
            group_id=obs.get("group_id"),
            base_person=obs.get("base_person"),
            num_people=len(obs.get("questions", [])),
        )

        # --- Initialize memory system for this group ---
        # user_id 加 run-id 前缀：上游的 id 只含 data_idx/model/memory，跨 run 是同一个，
        # 有状态的实现（ReasoningBank 落 jsonl、mem0 落向量库）会把上一次 run 的记忆读
        # 进来，违反「每个 task 从空记忆开始」。
        memory_user_id = f"{run_id}__data_{data_idx}_{args.model_name}_{args.memory_system}"
        memory_system = get_memory_system(args.memory_system, memory_user_id, args.server_url)

        if memory_system:
            print(f"\n[Memory] Initialized {args.memory_system} for user_id: {memory_user_id}")

        multi_queries = obs["questions"]
        ground_truth_list = obs.get("answers", [])

        gt_by_round = {}
        for gt in ground_truth_list:
            gt_by_round[gt['round_idx']] = gt

        all_round_results = []
        all_results = []
        all_scratchpads = []
        total_rounds = len(multi_queries)
        accumulated_plan_text = ""
        round_judgements = {}

        agent.reset()

        # --- Handle base_person ---
        base_person = obs.get("base_person")
        if base_person:
            base_name = base_person['name']
            base_query = base_person['query']
            base_plan = format_person_plan(base_name, base_person['daily_plans'])
            accumulated_plan_text = base_plan
            print(f"[Data {data_idx}] Base person set: {base_name}")

            if memory_system:
                base_chunk = json.dumps({
                    "name": base_name,
                    "query": base_query,
                    "is_base_person": True,
                    "final_plan": base_plan,
                }, ensure_ascii=False)
                memory_system.add(base_chunk)
                recorder.record("memory.add", chunk=base_chunk)

                if args.memory_system in ['mem0', 'mem0-g']:
                    print("[Memory] Waiting 10s for Mem0 indexing...")
                    time.sleep(50)
                print(f"[Memory] Added base person to memory ({len(base_chunk)} chars)")
                print(f"[Memory] Base person query: {base_query[:100]}...")
                print(f"[Memory] Base chunk content:\n{base_chunk[:1000]}{'...' if len(base_chunk) > 1000 else ''}")
            else:
                agent.set_base_person(base_name, base_query, base_plan)

        subquery_times = {}
        consecutive_round_failures = 0

        # --- Loop over persons ---
        for round_item in multi_queries:
            subquery_start = time.time()

            single_query = round_item['query']
            round_idx = round_item['round_idx']
            name = round_item.get('name', f'User{round_idx}')

            print(f"\n{'=' * 50}")
            print(f"[Data {data_idx}, Round {round_idx}, Name: {name}] Starting Agent...")
            print(f"{'=' * 50}")
            print(f"Query: {single_query[:200]}...")

            # 一条 run 跑几十分钟，中间一个请求抛异常就把整条掀了——react2 那次就是
            # 第 6 轮一个 400 掀掉了前 5 轮之外的全部。单轮失败记一条 round.error
            # 接着跑下一个人。
            #
            # 不发假的 agent.act / env.step：那两个事件在下游是配对读的（见
            # harness/trace.py 和 tests/test_trajectory_schema.py），伪造一条等于
            # 往结果里写一轮不存在的判分。失败的轮就是没有产出，如实记成新事件。
            try:
                # 1. Memory wrap
                memory_context = None
                if memory_system:
                    wrapped_prompt = memory_system.wrap_user_prompt(single_query)
                    recorder.record("memory.wrap_user_prompt",
                                    question=single_query, wrapped_prompt=wrapped_prompt)
                    memory_context = wrapped_prompt.split("</memory_context>")[0] + "</memory_context>"
                    memory_context = memory_context.strip()

                    print(f"\n[Memory] === Retrieved Context for {name} ===")
                    print(f"[Memory] Context length: {len(memory_context)} chars")
                    print(f"[Memory] Content:\n{memory_context[:2000]}{'...' if len(memory_context) > 2000 else ''}")
                    print(f"[Memory] === End Retrieved Context ===")

                # 2. Agent act (full ReAct loop inside)
                use_step_memory = getattr(args, 'use_step_memory', False)
                agent.prepare_for_person(
                    name=name,
                    round_idx=round_idx,
                    include_previous_plans=(memory_system is None),
                    memory_context=memory_context,
                    memory_system=memory_system if use_step_memory else None,
                )
                action = agent.act(single_query)
                agent_result = agent.last_result
                recorder.record(
                    "agent.act",
                    person=name,
                    query=single_query,
                    final_plan=action,
                    total_steps=getattr(agent_result, "total_steps", None),
                    success=getattr(agent_result, "success", None),
                    error_message=getattr(agent_result, "error_message", None),
                    scratchpad=scratchpad_to_json(agent_result),
                )

                accumulated_plan_text += f"\n\n{action}" if accumulated_plan_text else action

                print(f"\n{'=' * 50}")
                print(f"[Data {data_idx}, Round {round_idx}, Name: {name}] RESULT:")
                print(f"{'=' * 50}\n")
                print(f"{action[:1000]}..." if len(action) > 1000 else action)
                print(f"\nSteps: {agent_result.total_steps}, Success: {agent_result.success}")

                # 3. Env step (evaluate via env_server)
                gt = gt_by_round.get(round_idx)
                ground_truth = None
                if gt and 'daily_plans' in gt:
                    ground_truth = {
                        "name": name,
                        "daily_plans": gt['daily_plans'],
                        "judgement_mode": args.judgement_mode,
                    }
                    result = env_client.step(
                        action, ground_truth=ground_truth, need_judge=True,
                    )
                else:
                    result = env_client.step(action)

                recorder.record("env.step", ground_truth=ground_truth, result=result)

                observation = result["observation"]
                reward = result["reward"]
                info = result.get("info", {})

                round_result = {
                    'round': round_idx,
                    'name': name,
                    'query': single_query,
                    'result': action,
                }
                all_round_results.append(round_result)

                scratchpad_dict = agent.get_scratchpad_dict()
                all_scratchpads.append({
                    'round': round_idx,
                    'name': name,
                    'scratchpad': scratchpad_dict,
                    'total_steps': agent_result.total_steps,
                    'success': agent_result.success,
                    'error_message': agent_result.error_message,
                })

                parsed_this_round = parse_all_plans(accumulated_plan_text, multi_queries[:round_idx])
                all_results.append(parsed_this_round)

                judgement = info.get("judgement")
                if judgement:
                    round_judgements[round_idx] = judgement

                # 4. Build memory entry and add to memory
                if memory_system:
                    obs_with_judgement = {**(observation or {}), "judgement": judgement}
                    memory_entry = agent.build_memory_entry(
                        task=single_query,
                        action=action,
                        observation=obs_with_judgement,
                        reward=reward,
                    )
                    memory_system.add(memory_entry)
                    recorder.record("memory.add", chunk=memory_entry)

                    if args.memory_system in ['mem0', 'mem0-g']:
                        print(f"[Memory] Waiting for Mem0 indexing...")
                        time.sleep(60)
                    print(f"\n[Memory] === Added Trace for {name} ===")
                    print(f"[Memory] Chunk size: {len(memory_entry)} chars")
                    print(f"[Memory] Scratchpad steps: {len(scratchpad_dict)}")
                    print(f"[Memory] Has judgement: {judgement is not None}")
                    if judgement:
                        judge_preview = judgement[:500] + "..." if len(judgement) > 500 else judgement
                        print(f"[Memory] Judgement:\n{judge_preview}")
                    print(f"[Memory] Chunk content:\n{memory_entry[:2000]}{'...' if len(memory_entry) > 2000 else ''}")
                    print(f"[Memory] === End Trace ===\n")

            except Exception as exc:
                consecutive_round_failures += 1
                detail = f"{type(exc).__name__}: {exc}"
                recorder.record(
                    "round.error",
                    round_idx=round_idx,
                    person=name,
                    error=detail[:2000],
                )
                print(f"\n[round.error] 第 {round_idx} 轮（{name}）失败，跳过：{detail[:300]}")
                if consecutive_round_failures >= MAX_CONSECUTIVE_ROUND_FAILURES:
                    # 连着挂说明不是偶发，是 provider 或 server 塌了。继续跑只是
                    # 烧钱，把异常放出去让 finally 落盘已经跑完的部分。
                    print(f"[round.error] 连续 {consecutive_round_failures} 轮失败，中止这一组")
                    raise
                continue
            consecutive_round_failures = 0

            # Non-memory: feed cumulative judge feedback to agent
            if round_idx < total_rounds and memory_system is None:
                all_feedback = [
                    round_judgements[r]
                    for r in sorted(round_judgements.keys())
                    if r <= round_idx
                ]
                if all_feedback:
                    combined_judge = "\n\n".join(all_feedback)
                    agent.add_judge_feedback(combined_judge)
                    print(f"[Added judge ({args.judgement_mode}) for all travelers up to round {round_idx}]")

            subquery_time = time.time() - subquery_start
            subquery_times[round_idx] = subquery_time
            print(f"\n[Timing] Data {data_idx}, Round {round_idx}, {name}: {subquery_time:.2f}s")

        # --- Save output ---

        final_parsed_results = all_results[-1] if all_results else []

        for person in final_parsed_results:
            person_idx = person.get('person_idx')
            if person_idx is not None and person_idx in subquery_times:
                person['subquery_time'] = round(subquery_times[person_idx], 2)

        result_to_save = {
            'metadata': {
                'model_name': args.model_name,
                'judgement_mode': args.judgement_mode,
                'memory_system': args.memory_system,
                'mode': 'agent',
                'max_steps': args.max_steps,
            },
            f'{args.model_name}_sole-planning_results': final_parsed_results,
            'all_results': all_results,
            'scratchpads': all_scratchpads,
        }

        print(f"\n{'=' * 60}")
        print(f"[Data {data_idx}] GROUP SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total persons: {total_rounds}")
        print(f"Memory system: {args.memory_system}")
        print(f"Saving to: {output_file}")
        print(f"{'=' * 60}\n")

        with open(output_file, 'w') as f:
            json.dump(result_to_save, f, indent=4, ensure_ascii=False)

    # --- Cleanup ---
    env_client.close()

    # --- Usage stats ---
    duration_seconds = time.time() - start_time
    duration_minutes = duration_seconds / 60

    usage = agent.get_usage_stats()
    usage['duration_seconds'] = round(duration_seconds, 2)
    usage['duration_minutes'] = round(duration_minutes, 2)

    print(f"\n=== Usage Stats ===")
    print(f"Total input tokens: {usage['total_input_tokens']}")
    print(f"Total output tokens: {usage['total_output_tokens']}")
    print(f"Total cost: ${usage['total_cost']:.4f}")
    print(f"Duration: {duration_minutes:.2f} minutes ({duration_seconds:.2f} seconds)")

    stats_dir = os.path.join(args.output_dir, "stats_results")
    os.makedirs(stats_dir, exist_ok=True)
    usage_file = os.path.join(stats_dir, "usage_stats.json")
    with open(usage_file, 'w') as f:
        json.dump(usage, f, indent=2)
    print(f"Usage stats saved to: {usage_file}")

    # --- Postprocess: combination + eval ---
    submission_dir = os.path.join(args.output_dir, "submission")

    print("\n--- Running combination ---")
    submission_file = combine(
        model_name=args.model_name,
        output_dir=args.output_dir,
        submission_file_dir=submission_dir,
        mode="sole_planning",
    )

    print("\n--- Running evaluation ---")
    evaluate(
        submission_path=submission_file,
        model_name=args.model_name,
        memory_system=args.memory_system,
        global_csv=global_csv,
    )

    print("Done!")


if __name__ == "__main__":
    main()
