"""
Complete integration example showing Memory + Environment working together.

This script demonstrates the exact workflow from your pseudo-code:
1. Create task_id
2. Initialize memory client
3. Initialize environment client
4. Run task loop with memory-wrapped prompts
5. Store experiences in memory
"""

import argparse
import logging
import uuid
from typing import List, Dict, Any
from env.env_client import EnvironmentClient
import time
import os
import sys
import json
from memory.client import MemoryClient
from agent import MathAgent
from dataclasses import asdict
from pathlib import Path

import dataset as dataset_mod
from harness import config as config_mod
from harness import llm_journal, runctx
from harness.trace import Recorder

logger = logging.getLogger(__name__)


def _save_logs_jsonl(logs: List[Dict[str, Any]], json_output_file: str) -> None:
    """Save logs as JSONL with backward-compatible fields."""
    print(f"Saving Logs.... len={len(logs)}")
    os.makedirs(os.path.dirname(json_output_file), exist_ok=True)
    with open(json_output_file, "w", encoding="utf-8") as jsonf:
        for log in logs:
            log_dict = log.copy()
            if 'judge_result' in log_dict and hasattr(log_dict['judge_result'], '__dict__'):
                log_dict['judge_result'] = asdict(log_dict['judge_result'])
            jsonf.write(json.dumps(log_dict, ensure_ascii=False) + "\n")


def _get_json_output_dir(cfg: Dict[str, Any]) -> str:
    """Return the method-specific JSON output directory for this run."""
    memory_name = cfg["memory"]["memory_system_name"]
    if memory_name != "long_context":
        return os.path.join(cfg["output"]["json_output_dir"], memory_name)
    return os.path.join(
        cfg["output"]["json_output_dir"],
        f"{memory_name}_{cfg['agent']['model_name']}",
    )


def _get_paper_result_file(cfg: Dict[str, Any], paper_key: str) -> str:
    """Return the result file path for a paper."""
    json_path = _get_json_output_dir(cfg)
    if paper_key:
        return os.path.join(json_path, paper_key, "result.jsonl")
    return os.path.join(json_path, "result.jsonl")


def _is_paper_processed(
    cfg: Dict[str, Any],
    paper_key: str,
    expected_num_queries: int,
) -> bool:
    """Check whether a paper already has a complete result file."""
    result_file = _get_paper_result_file(cfg, paper_key)
    if not os.path.isfile(result_file):
        return False

    if expected_num_queries <= 0:
        return os.path.getsize(result_file) > 0

    with open(result_file, "r", encoding="utf-8") as f:
        completed_queries = sum(1 for line in f if line.strip())

    return completed_queries >= expected_num_queries


def run_task_with_memory_and_env(
    cfg: Dict[str, Any] ,
    tasks: List[str],
    paper_key: str,
    recorder,
    run_id: str = "",
):
    """
    Run a complete task loop with memory and environment integration.
    
    This follows your exact pseudo-code pattern:
    + task_id = uuid.uuid4()
    + client = MemoryClient(task_id, memory_system_name='mirix')
     
     obs = get_initial_obs(env)
     for i in range(task_num):
    +    prompt = client.wrap_user_prompt(task_i)
    +    action = agent(prompt)
         obs = env(action)
    +    client.add(f"action: {action}\\nobs: {obs}")
    """
    logger.info("\n%s", "=" * 80)
    logger.info("RUNNING TASK: %s", cfg["env"]["env_name"].upper())
    logger.info("%s", "=" * 80)

    if _is_paper_processed(cfg, paper_key, len(tasks)):
        logger.info(
            "Paper %s already has a complete result file. Skipping.",
            paper_key,
        )
        return []
    
    # 0. Create unique task ID。加 run-id 前缀：memory server 和 env server 里的 LLM
    # 记录靠它知道自己属于哪条 run（见 harness.runctx），几条 run 并行时才分得开。
    # 有状态的 memory system 也顺带按 run 隔离了。
    task_id = str(uuid.uuid4())
    if run_id:
        task_id = f"{run_id}__{task_id}"
    
    #1. Initialize task agent
    agent=MathAgent(
        model_name=cfg["agent"]["model_name"],temperature=cfg["agent"]["temperature"],max_tokens=cfg["agent"]["max_tokens"],
        backend=cfg["agent"]["backend"],
        base_url=cfg["agent"]["base_url"] if cfg["agent"]["backend"]=="openai" else None)
    
    # 2. Initialize memory client
    memory_client = MemoryClient(
        task_id, 
        memory_system_name=cfg["memory"]["memory_system_name"], 
        base_url=cfg["memory"]["base_url"],
        timeout=cfg["memory"]["timeout"],
       )
    
    # 3. Initialize environment client
    env_client = EnvironmentClient(
        task_id=task_id,
        env_name=cfg["env"]["env_name"],
        base_url=cfg["env"]["base_url"],
        timeout=cfg["env"]["timeout"],
        env_config=cfg["env"]["env_config"] or {"max_steps": 10},
    )
    json_path = _get_json_output_dir(cfg)
    if not os.path.exists(json_path):
        os.makedirs(json_path, exist_ok=True)
        logger.info("Created directory for JSON output: %s", json_path)
    
    # 4. Get initial observation
    obs = env_client.reset()
    
    # 5. Add initial observation to memory
    logs: List[Dict[str, Any]] = []
    
    if obs is not None:
        memory_client.add(f"Initial result: Empty\n")
        recorder.record("memory.add", chunk="Initial result: Empty\n")
    
    # 6. Run task loop
    
    step_count = 0
    # 
    for subtask_idx, (subtask, ground_truth, background) in enumerate(tasks):
        t0=time.time()
        query=agent.build_prompt(task=subtask, background=background)
        # Wrap task with memory context
        prompt = memory_client.wrap_user_prompt(query)
        recorder.record("memory.wrap_user_prompt", question=query, wrapped_prompt=prompt)
        # Agent generates action based on memory-wrapped prompt
        # this is an simple example, use your agent and agent step. 
        action = agent.act(prompt)
        recorder.record("agent.act", action=action)
        
        logger.info("\n--- Agent -> Env ---")
        logger.info("Memory-wrapped prompt: %s", prompt)
        logger.info("Agent action: %s", action)
         
        # Execute action in environment
        # we judge the result every subtask in order to calculate ps@k score, therefore need_judge=True every step,
        # but we do not include such reward signal into memory entry for next step. 
        # You can also include such signal into memory entry by setting cfg["memory"]["judge_result_in_memory"]=True
        if subtask_idx==len(tasks)-1:
            result = env_client.step(action, ground_truth=ground_truth, need_judge=True)
        else:
            result = env_client.step(action, ground_truth=ground_truth,need_judge=True)
        recorder.record("env.step", ground_truth=ground_truth, result=result)

        step_count += 1
        logger.info("--- Env -> Agent ---")
        logger.info("Env response: %s", result["observation"]["final"])
        if result.get("reward") is not None:
            logger.info("Judge reward: %s", result.get("reward"))
        
       
        
        # add judge result into memory entry if needed
        if cfg["memory"]["judge_result_in_memory"] and result.get("judge_result") is not None:
            memory_entry = agent.build_memory_entry(
                task=subtask,
                observation=result.get("observation"),
                action=action,
                reward=result.get("reward")
                )
        else:
            # Add action-observation pair to memory
            memory_entry = agent.build_memory_entry(
                task=subtask,
                observation=result.get("observation"),
                action=action,
                reward=None, # result.get("reward")
                )
        memory_client.add(memory_entry)
        recorder.record("memory.add", chunk=memory_entry)
        judge_result = {
                    "feedback": result["observation"]["judge_result"] if "judge_result" in result["observation"] else None,
                    "is_correct": result["reward"] if "reward" in result else None,
                    
                }
        logs.append({
            "query_id": subtask_idx,
            "query": subtask,
            "memory_context": result['observation']['memory_context'] if 'memory_context' in result['observation'] else None,
            "response": result["observation"]["final"],
            "is_correct": result["reward"] if "reward" in result else None,
            "time": time.time() - t0,
            "ground_truth": ground_truth,
            "judge_result":judge_result,
            "observation": result.get("observation"),
        })
    
    
    
    
    # 8. Cleanup
     
    env_client.close()
    json_output_file = _get_paper_result_file(cfg, paper_key)
    _save_logs_jsonl(logs, json_output_file)
    logger.info("Saved %s logs to: %s", len(logs), json_output_file)

    return logs


def main() -> None:
    """跑一条 formal-reasoning case，并把整条轨迹落盘。

    上游的 `run_all()` 一次遍历整个 split（math 是 40 条），没有 limit 开关；那个函数
    合并后没人调了，已经删掉。这里复用
    同一个 `run_task_with_memory_and_env`——loop 语义完全一致——只把任务集合缩到一条
    case，外加轨迹落盘。

        uv run marena-case -c formal_reasoning/math-longcontext-gpt5mini.json \
            --case-index 0 --max-queries 2
    """
    parser = argparse.ArgumentParser(
        description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-c", "--config", required=True,
                        help="config 路径；相对路径按 $MARENA_CONFIGS 解析")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--case-name", default=None, help="按 paper_name 选，优先于 --case-index")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="只跑这条 case 的前 N 道子题，0 表示全部")
    parser.add_argument("--run-id", default=None,
                        help="输出落到 $MARENA_RESULTS/<memory-system>/<run-id>/，默认取 config 文件名")
    parser.add_argument("--force", action="store_true",
                        help="删掉已有 result.jsonl 重跑；默认跳过已完成的 case")
    parser.add_argument("--list", action="store_true", help="只列出这个数据集里的 case，不跑")
    args = parser.parse_args()

    config_path = config_mod.resolve_path(args.config)
    run_id = args.run_id or config_path.stem
    cfg = config_mod.load(config_path, run_id=run_id)
    hf_config = config_mod.hf_config(cfg)

    if args.list:
        for case_id, name, count in dataset_mod.list_cases(hf_config):
            print(f"{case_id:>3}  {name:<24} {count} queries")
        return

    case = dataset_mod.load_case(hf_config, index=args.case_index, name=args.case_name)
    tasks = case.tasks(limit=args.max_queries)

    # 用上游自己的路径规则定位输出文件，保证和它写的位置一致
    result_file = Path(_get_paper_result_file(cfg, case.name))
    result_file.parent.mkdir(parents=True, exist_ok=True)
    if args.force and result_file.exists():
        result_file.unlink()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(result_file.parent / "run.log", encoding="utf-8")],
    )
    logger.info("case=%s queries=%s memory=%s model=%s",
                case.name, len(tasks), cfg["memory"]["memory_system_name"], cfg["agent"]["model_name"])
    logger.info("输出目录 %s", result_file.parent)

    # LLM 记录：和 marena-travel-case 同一套。agent 进程自己的调用打 `<run-id>__agent`，
    # 两个 server 那边的靠带前缀的 user_id / task_id；跑完按前缀切进这条 run 的目录。
    os.environ.setdefault("MARENA_LLM_ROLE", "agent")
    runctx.REQUEST_ID.set(f"{run_id}__agent")
    runctx.tag_litellm()
    runctx.tag_openai_sdk()          # math agent 的工具循环直接用 openai SDK
    runctx.propagate_context_to_threads()
    llm_journal.install()
    journal_marks = llm_journal.marks()

    recorder = Recorder()
    logs = None
    try:
        logs = run_task_with_memory_and_env(
            cfg=cfg, tasks=tasks, paper_key=case.name, recorder=recorder, run_id=run_id)
    finally:
        harvested = llm_journal.harvest(journal_marks, result_file.parent, run_id)
        if harvested:
            detail = "、".join(f"{role} {n}" for role, n in sorted(harvested.items()))
            print(f"llm_calls.<role>.jsonl  {detail}")
        trajectory_file = result_file.parent / "trajectory.json"
        trajectory_file.write_text(json.dumps({
            "config_path": str(config_path),
            "config": cfg,
            "case": case.name,
            "num_queries": len(tasks),
            "trace": recorder.events,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"case {case.name}   {len(tasks)} queries")
    for log in logs or []:
        verdict = "correct" if log.get("is_correct") else "wrong"
        print(f"  q{log['query_id']}  {verdict:<8} {log['time']:6.1f}s  {str(log['response'])[:56]!r}")
    print()
    print(f"result.jsonl     {result_file}")
    print(f"trajectory.json  {trajectory_file}")
    print(f"run.log          {result_file.parent / 'run.log'}")


if __name__ == "__main__":
    main()
