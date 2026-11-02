"""
Run BrowseComp-Plus search via the environment server.

This script calls the env server (env/env_server.py, port 8001) and memory server (memory/server.py, port 8000),
runs the search agent for each query, and returns the results.

Usage (from project root, env server already running via `python env/env_server.py`):

    python run_search.py --config configs/web_search_configs/search_task.json

All settings (including query_ids) are loaded from the config JSON file.
"""

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path

# Project root = directory containing run_search.py (no "MemoryArena" package name)
_SCRIPT_DIR = Path(__file__).resolve().parent
# Load .env so OPENAI_API_KEY and OPENAI_BASE_URL are set (optional: python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv(_SCRIPT_DIR / ".env", override=True)
except ImportError:
    pass
_REPO_ROOT = _SCRIPT_DIR.parent
_MEMORY = _SCRIPT_DIR / "memory"  # memory/client.py for MemoryClient
_ENV_SYSTEMS = _SCRIPT_DIR / "env" / "env_systems"

def _setup_paths():
    # So "agent" and "env" resolve to agent/ and env/ (no MemoryArena package)
    if str(_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPT_DIR))
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    if str(_ENV_SYSTEMS) not in sys.path:
        sys.path.insert(0, str(_ENV_SYSTEMS))
    # So "from client import MemoryClient" finds memory/client.py
    if _MEMORY.exists() and str(_MEMORY) not in sys.path:
        sys.path.insert(0, str(_MEMORY))

_setup_paths()

from agent.search import load_correct_answers

from harness import config as config_mod, llm_journal, runctx
from harness.trace import Recorder


def load_config(config_path: Path) -> dict:
    """Load JSON config file and return as dict."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


class ConfigArgs:
    """Simple namespace to hold config values."""
    pass


def config_to_args(config: dict) -> ConfigArgs:
    """Convert config dict to args-like namespace."""
    args = ConfigArgs()
    
    # Agent settings
    agent_cfg = config.get("agent", {})
    args.model_name = agent_cfg.get("model_name", "gpt-5-mini")
    args.embedding_model = agent_cfg.get("embedding_model", "text-embedding-3-small")
    args.max_iterations = agent_cfg.get("max_iterations", 35)
    args.provider = agent_cfg.get("provider")

    # Memory settings
    mem_cfg = config.get("memory", {})
    args.memory_system = mem_cfg.get("memory_system_name", "bm25")
    args.memory_url = mem_cfg.get("memory_url", "http://0.0.0.0:8000")
    args.no_memory = mem_cfg.get("no_memory", False)
    args.step_memory = mem_cfg.get("use_step_memory", False)

    # Env settings
    env_cfg = config.get("env", {})
    args.env_server_url = env_cfg.get("env_server_url", "http://0.0.0.0:8001")
    args.script_path = env_cfg.get("script_path")
    args.index_path = env_cfg.get("index_path", "env/env_systems/web_search_env/embeddings/shard*.index")
    args.corpus_path = env_cfg.get("corpus_path", "web_search_env/data/corpus.jsonl")
    args.mcp_url = env_cfg.get("mcp_url")
    args.timeout = env_cfg.get("timeout", 3000)
    args.mcp_name = env_cfg.get("mcp_name", "retrieval-mcp-server")

    # Task-specific settings
    task_cfg = config.get("task_specific", {})
    args.query_ids = task_cfg.get("query_ids", [])
    args.data_dir = Path(task_cfg.get("data_dir", "env/env_systems/web_search_env/data"))
    args.qrel_evidence = task_cfg.get("qrel_evidence", "topics-qrels/qrel_evidence.txt")
    args.judge_model = task_cfg.get("judge_model", "gpt-4.1")

    # Output settings
    out_cfg = config.get("output", {})
    args.output_dir = Path(out_cfg.get("output_dir", "out/search_run"))

    return args

# Environment server client (optional)
try:
    from env.env_client import EnvironmentClient
except ImportError:
    EnvironmentClient = None


def no_replay_session():
    """给 env client 用的 session：连接失败可以重试，服务端报错不能重放。

    `EnvironmentClient` 的默认 session 对 POST 的 429/5xx 自动重试 3 次。travel 的
    一步是一次判分，重放顶多多判一次；这里不一样——`run_sequential` 一个请求就是
    整条 pipeline（逐条 subquery 跑 agent、写记忆，最后判分）。500 回来时 server
    那边已经跑了不知道多少，重放等于把整条 run 从头再跑一遍：钱翻倍，同一个
    task_id 的记忆被重复写入。search-longcontext-case1040 第一次跑就这样被重放了
    两次。

    所以只留 connect 重试（请求没发出去，重试是安全的），read / status 一律不重试，
    让 500 直接抛给调用方。
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry = Retry(total=3, connect=3, read=0, status=0, other=0,
                  backoff_factor=0.2, raise_on_status=False)
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def load_ground_truth(jsonl_path: Path) -> dict:
    gt = {}
    if not jsonl_path.exists():
        return gt
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            qid = str(obj.get("query_id", ""))
            gt[qid] = {
                "query": obj.get("query", ""),
                "answer": obj.get("answer", ""),
                "evidence": obj.get("evidence_docs", obj.get("evidence", [])),
                "gold_docs": obj.get("gold_docs", []),
            }
    return gt


def load_qrel(qrel_path: Path) -> dict:
    """Load qrel (query id -> list of relevant docids) if available."""
    qrel = {}
    if not qrel_path or not qrel_path.exists():
        return qrel
    with qrel_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                qid, docid, rel = parts[0], parts[2], parts[-1]
                if rel != "0":
                    qrel.setdefault(qid, []).append(docid)
    return qrel


def load_qrel_data(qrel_path: Path) -> dict:
    """Load qrel evidence data for recall calculation: query_id -> list[doc_id]."""
    from collections import defaultdict

    qrel_data = defaultdict(list)
    if not qrel_path or not qrel_path.exists():
        return dict(qrel_data)

    with qrel_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                query_id = parts[0]
                doc_id = parts[2]
                qrel_data[query_id].append(doc_id)
    return dict(qrel_data)


def _summarize_single_result(out: dict, qrel_data: dict | None = None) -> dict:
    """
    Build an evaluation summary for a single result dict produced by the env server.

    The summary mirrors the human-readable lines printed at the end of main().
    """
    total = 1
    skipped = 0

    judgement = out.get("judgement") or {}
    if not judgement or "correct" not in judgement:
        skipped = 1
        evaluated = 0
        accuracy = None
        calibration_error = None
    else:
        evaluated = 1
        correct = bool(judgement.get("correct"))
        confidence = judgement.get("confidence")
        acc_val = 1.0 if correct else 0.0
        accuracy = acc_val
        if isinstance(confidence, (int, float)):
            calibration_error = abs(confidence / 100.0 - acc_val)
        else:
            calibration_error = None

    # Recall: compute from retrieved_docids vs qrel for this query_id.
    # If there is no qrel entry or no relevant docs, recall is None.
    query_id = str(out.get("query_id") or "")
    avg_recall = None
    if qrel_data and query_id and query_id in qrel_data:
        relevant_docids = {str(d) for d in qrel_data[query_id]}
        if relevant_docids:
            retrieved = {str(d) for d in (out.get("retrieved_docids") or [])}
            if not retrieved:
                avg_recall = 0.0
            else:
                avg_recall = len(retrieved & relevant_docids) / len(relevant_docids)

    # Average tool calls per tool (for a single evaluation, just echo counts).
    tool_counts = out.get("tool_call_counts") or {}
    avg_tool_calls: dict[str, float] = {}
    if tool_counts and evaluated > 0:
        for tool, count in tool_counts.items():
            try:
                c = float(count)
            except Exception:
                continue
            avg_tool_calls[str(tool)] = c / evaluated

    summary = {
        "processed_evaluations": total,
        "skipped_evaluations": skipped,
        "evaluated_responses": total - skipped,
        "accuracy": accuracy,  # fraction in [0,1] or None
        "recall": avg_recall,  # fraction in [0,1] or None
        "average_tool_calls": avg_tool_calls,  # {tool_name: avg_calls}
        "calibration_error": calibration_error,  # fraction in [0,1] or None
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run BrowseComp-Plus search (agent + browsecomp_plus env)."
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        required=True,
        help="config 路径；相对路径按 $MARENA_CONFIGS 解析",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="输出落到 $MARENA_RESULTS/<memory-system>/<run-id>/，默认取 config 文件名",
    )
    parser.add_argument(
        "--query-id",
        dest="cli_query_ids",
        action="append",
        default=None,
        help="只跑这些 query id，可给多次；不给就用 config 里的 task_specific.query_ids。"
             "注意这里是 BrowseComp 的 query_id（如 454），不是 HF 数据集里的行号，"
             "两者的对应见 docs/references/",
    )
    cli_args = parser.parse_args()

    # config 走 harness.config：相对路径按 $MARENA_CONFIGS 解析，<占位符> 换成环境变量
    config_path = config_mod.resolve_path(cli_args.config)
    config = config_mod.load(config_path)
    run_id = cli_args.run_id or config_path.stem
    args = config_to_args(config)
    args.query_id = cli_args.cli_query_ids or args.query_ids

    if not args.query_id:
        print("Error: No query_ids specified in config.")
        sys.exit(1)

    print(f"Loaded config from {config_path}")

    # 结果只落 $MARENA_RESULTS/<memory-system>/<run-id>/。config 里写的 output_dir
    # 只是默认值，run-id 一给就以 run_dir 为准——两处规则和 travel / math 完全一致。
    data_dir = args.data_dir if args.data_dir.is_absolute() else (_SCRIPT_DIR / args.data_dir)
    output_dir = config_mod.run_dir(config, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir
    config.setdefault("output", {})["output_dir"] = str(output_dir)
    (output_dir / "resolved-config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # 观测：和 run_travel 同一套。agent 那侧的调用其实发生在 env server fork 出来的
    # 子进程里（agent/search.py），归属靠 runctx.child_env 往下传，见那里的说明。
    os.environ.setdefault("MARENA_LLM_ROLE", "agent")
    runctx.REQUEST_ID.set(f"{run_id}__agent")
    runctx.tag_openai_sdk()
    runctx.propagate_context_to_threads()
    journal_marks = llm_journal.marks()
    recorder = Recorder()

    ground_truth_path = data_dir / "browsecomp_plus_decrypted.jsonl"
    if not ground_truth_path.exists():
        print(f"Warning: ground truth not found at {ground_truth_path}")

    gt = load_ground_truth(ground_truth_path)
    query_ids = [str(q) for q in args.query_id]

    # Filter query IDs by availability in ground truth
    missing_ids = [qid for qid in query_ids if qid not in gt]
    for qid in missing_ids:
        print(f"Query ID {qid} not in ground truth. Skipping.")

    valid_ids = [qid for qid in query_ids if qid in gt]
    if not valid_ids:
        print(f"No valid query IDs found in ground truth. Available: {list(gt.keys())[:20]}...")
        sys.exit(1)

    script_path = args.script_path
    if script_path is None:
        script_path = str(
            _ENV_SYSTEMS / "web_search_env" / "search_agent" / "openai_client.py"
        )
    # Load qrel evidence for recall calculation; falls back to empty if not present.
    qrel_evidence_path = Path(args.qrel_evidence) if args.qrel_evidence else None
    qrel_data = load_qrel_data(qrel_evidence_path) if qrel_evidence_path else {}

    if EnvironmentClient is None:
        sys.exit(
            "EnvironmentClient not available. Install env dependency and ensure env/env_client.py is importable."
        )

    per_query_summaries = []
    per_query_results = {}

    try:
        for qid in valid_ids:
            original_query = gt[qid]["query"]
            # Subqueries: use decomposed answers if available, else single query
            subqueries = [original_query]
            correct_answers = []
            try:
                correct_data = load_correct_answers(qid, data_dir)
                subqueries = correct_data.get("subqueries") or [original_query]
                correct_answers = [
                    a.get("correct_answer", "") for a in correct_data.get("correct_answers", [])
                ]
            except FileNotFoundError:
                pass

            # task_id 必须带 run-id 前缀：env server 的中间件从请求体里取它当 REQUEST_ID，
            # 这条 run 在 server 侧引发的 LLM 调用全靠这个前缀归位（harness.runctx）。
            task_id = f"{run_id}__{uuid.uuid4()}"
            recorder.record("env.reset", query_id=str(qid), query=original_query,
                            num_subqueries=len(subqueries))
            env_config = {
                "task_id": task_id,
                "query_id": qid,
                "timeout": args.timeout,
                "original_query": original_query,
                "subqueries": subqueries,
                "correct_answers": correct_answers,
                "output_dir": str(output_dir),
                "memory_url": args.memory_url,
                "memory_system_name": args.memory_system,
                "no_memory": args.no_memory,
                "step_memory": args.step_memory,
                "script_path": script_path,
                "index_path": args.index_path,
                "corpus_path": args.corpus_path,
                "agent_model": args.model_name,
                "model_name": args.embedding_model,
                "judge_model": args.judge_model,
                "ground_truth_path": str(ground_truth_path),
                "provider": args.provider,
                "mcp_url": args.mcp_url,
                "mcp_name": args.mcp_name,
                "qrel_data": qrel_data,
                "max_iterations": args.max_iterations,
            }
            print(f"Using environment server at {args.env_server_url} (env_name=browsecomp-plus, query_id={qid}).")
            env_client = EnvironmentClient(
                task_id=task_id,
                timeout=args.timeout,
                env_name="browsecomp-plus",
                base_url=args.env_server_url,
                env_config=env_config,
                session=no_replay_session(),
            )
            env_client.reset()
            step_response = env_client.step({"command": "run_sequential"})
            result = step_response.get("observation", {})
            env_client.close()

            out = {
                "metadata": result.get("metadata", {}),
                "query_id": str(qid),
                "tool_call_counts": result.get("tool_call_counts"),
                "usage": result.get("usage", {}),
                "status": result.get("status", "completed"),
                "retrieved_docids": result.get("retrieved_docids", []),
                "result": result.get("trace", []),
                "api_calls": result.get("api_calls", []),
                "final_query": result.get("final_query"),
                "predicted_answer": result.get("predicted_answer"),
                "correct_answer": result.get("correct_answer"),
                "judgement": result.get("judgement"),
                "recall": result.get("recall"),
            }
            # env server 在 run_full 里记的事件（每条 subquery 一轮）跟着响应回来，
            # 接到这条 run 的轨迹上。老字段一个没少，只是多读一个 events。
            recorder.events.extend(result.get("events") or [])

            summary = _summarize_single_result(out, qrel_data=qrel_data)
            per_query_summaries.append(summary)
            per_query_results[qid] = {"summary": summary, "result": out}
    finally:
        # 不管跑完还是中途炸了，轨迹和 LLM 记录都要落盘——半条 run 的轨迹也是证据。
        harvested = llm_journal.harvest(journal_marks, output_dir, run_id)
        if harvested:
            detail = "、".join(f"{role} {n}" for role, n in sorted(harvested.items()))
            print(f"llm_calls.<role>.jsonl  {detail}")
        trajectory = output_dir / "trajectory.json"
        trajectory.write_text(json.dumps({
            "config_path": str(config_path),
            "config": config,
            "case": ",".join(str(q) for q in valid_ids),
            "num_queries": recorder.count("agent.act"),
            "trace": recorder.events,
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\ntrajectory.json  {trajectory}")


    # Aggregate summaries across all valid query IDs
    total_processed = sum(s["processed_evaluations"] for s in per_query_summaries)
    total_skipped = sum(s["skipped_evaluations"] for s in per_query_summaries)
    total_evaluated = sum(s["evaluated_responses"] for s in per_query_summaries)

    # Overall accuracy: count queries whose judgement is correct; denominator is
    # the total number of valid input query IDs (even if a particular query had
    # no judgement and is effectively incorrect).
    correct_queries = 0
    for s in per_query_summaries:
        acc = s.get("accuracy")
        # For our per-query summaries, accuracy is either 1.0, 0.0, or None.
        if acc is not None and acc >= 0.5:
            correct_queries += 1
    overall_accuracy = None
    if valid_ids:
        overall_accuracy = correct_queries / len(valid_ids)

    # Overall recall: average over queries where recall is not None
    recall_sum = 0.0
    recall_count = 0
    for s in per_query_summaries:
        r = s.get("recall")
        if r is not None:
            recall_sum += r
            recall_count += 1
    overall_recall = None
    if recall_count > 0:
        overall_recall = recall_sum / recall_count

    # Average tool calls per tool across all evaluated responses
    total_tool_calls = {}
    for s in per_query_summaries:
        n = s.get("evaluated_responses", 0)
        tools = s.get("average_tool_calls") or {}
        for tool, avg_calls in tools.items():
            total_tool_calls[tool] = total_tool_calls.get(tool, 0.0) + avg_calls * n
    overall_avg_tool_calls = {}
    if total_evaluated > 0:
        for tool, total_calls in total_tool_calls.items():
            overall_avg_tool_calls[tool] = total_calls / total_evaluated

    # Overall calibration error: average over evaluations where it is defined
    calib_sum = 0.0
    calib_n = 0
    for s in per_query_summaries:
        n = s.get("evaluated_responses", 0)
        ce = s.get("calibration_error")
        if ce is not None and n:
            calib_sum += ce * n
            calib_n += n
    overall_calib = None
    if calib_n > 0:
        overall_calib = calib_sum / calib_n

    overall_summary = {
        "processed_evaluations": total_processed,
        "skipped_evaluations": total_skipped,
        "evaluated_responses": total_evaluated,
        "accuracy": overall_accuracy,
        "recall": overall_recall,
        "average_tool_calls": overall_avg_tool_calls,
        "calibration_error": overall_calib,
    }

    # Print aggregated human-readable summary
    print(f"Processed {total_processed} evaluations ({total_skipped} skipped)")
    print(f"Evaluated {total_evaluated} responses:")

    if overall_accuracy is not None:
        print(f"Accuracy: {overall_accuracy * 100:.2f}%")
    else:
        print("Accuracy: N/A")

    if overall_recall is not None:
        print(f"Recall: {overall_recall * 100:.2f}%")
    else:
        print("Recall: N/A")

    if overall_avg_tool_calls:
        formatted_tools = {
            k: float(f"{v:.2f}") for k, v in overall_avg_tool_calls.items()
        }
        print(f"Average Tool Calls: {formatted_tools}")
    else:
        print("Average Tool Calls: {}")

    if overall_calib is not None:
        print(f"Calibration Error: {overall_calib * 100:.2f}%")
    else:
        print("Calibration Error: N/A")

    # Write a single JSON file summarizing all query IDs, plus per-query details
    if len(valid_ids) == 1:
        out_file = output_dir / f"query_{valid_ids[0]}_result.json"
    else:
        out_file = output_dir / "results_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "query_ids": valid_ids,
                "summary": overall_summary,
                "per_query": per_query_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    print(f"Result summary written to {out_file}")


if __name__ == "__main__":
    main()
