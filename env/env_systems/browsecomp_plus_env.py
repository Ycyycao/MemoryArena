from typing import Any, Dict, Optional, Tuple
from .base_env import BaseEnvironment
import json
from pathlib import Path
from typing import List

def make_judge_client():
    """env server 里判分用的 openai 客户端，带上这条 run 的归属头。

    server 进程不装 `runctx.tag_openai_sdk()`（它在构造时取 REQUEST_ID，server 里
    每个请求的值不同）。所以在这里、请求的上下文里构造时显式带上：中间件已经把
    请求体里带 run-id 前缀的 task_id 放进了 REQUEST_ID。不带的话判分那一次调用在
    网关记录里 request_id 为空，harvest 切不进这条 run（case1040 首次跑通时就是）。
    """
    import os

    import openai

    headers = None
    try:
        from harness import runctx
        rid = runctx.current()
        if rid:
            headers = {runctx.HEADER: rid}
    except ImportError:
        pass
    base_url = os.environ.get("OPENAI_BASE_URL")
    return openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
                         base_url=base_url if base_url else None,
                         default_headers=headers)


class BrowseCompPlusEnvironment(BaseEnvironment):
    """BrowseComp-Plus environment for web browsing tasks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.history = []
        self.browser_state = {}
        self._current_observation = None
        self.memory_client = self.config.get("memory_client")
        self.agent = self.config.get("agent")
        self.query_id = self.config.get("query_id")
        self.base_dir = self.config.get("base_dir")
        self.output_dir = self.config.get("output_dir")
        self.judge_model = self.config.get("judge_model", "gpt-4.1")
        self.agent_model = self.config.get("agent_model", "gpt-5-mini")
        # Default to the new web_search_env layout under env/env_systems/web_search_env
        _env_systems = Path(__file__).resolve().parent  # this file is in env_systems
        _default_script = "web_search_env/search_agent/openai_client.py"
        _default_index = "web_search_env/embeddings/shard*.index"
        self.script_path = self.config.get("script_path", _default_script)
        self.index_path = self.config.get("index_path", _default_index)
        # Resolve relative paths so subprocess can find index/corpus regardless of cwd
        _root = Path(__file__).resolve().parents[2]  # project root
        def _resolve(p: str) -> str:
            if Path(p).is_absolute():
                return p
            # Paths like env/env_systems/... are relative to project root
            if p.startswith("env/") or p.startswith("env\\"):
                return str(_root / p)
            return str(_env_systems / p)
        if not Path(self.script_path).is_absolute():
            self.script_path = _resolve(self.script_path)
        if not Path(self.index_path).is_absolute():
            self.index_path = _resolve(self.index_path)
        self.model_name = self.config.get("model_name", "text-embedding-ada-002")
        self.mcp_url = self.config.get("mcp_url")
        self.mcp_name = self.config.get("mcp_name", "retrieval-mcp-server")
        self.gpu = self.config.get("gpu")
        self.provider = self.config.get("provider")
        self.qrel_data = self.config.get("qrel_data", {})
        self.ground_truth = self.config.get("ground_truth")
        self.client = self.config.get("client")
        self.max_iterations = self.config.get("max_iterations", 30)
        self.step_memory = self.config.get("step_memory", False)
        # If True, store evaluation metadata (judgement + correct answer when incorrect) in memory
        self.store_eval_in_memory: bool = bool(self.config.get("store_eval_in_memory", False))
        # When running on env server: create memory_client from config if not provided
        if self.memory_client is None and self.config.get("task_id") and self.config.get("memory_url"):
            import sys
            import logging
            _root = Path(__file__).resolve().parents[2]  # env_systems -> env -> project root
            _memory_dir = _root / "memory"
            _log = logging.getLogger(__name__)
            if _memory_dir.exists():
                try:
                    if str(_memory_dir) not in sys.path:
                        sys.path.insert(0, str(_memory_dir))
                    from client import MemoryClient
                    self.memory_client = MemoryClient(
                        user_id=str(self.config["task_id"]),
                        memory_system_name=self.config.get("memory_system_name", "bm25"),
                        base_url=self.config["memory_url"].rstrip("/"),
                    )
                    _log.info(
                        "Memory client created from memory/ (system=%s, base_url=%s).",
                        self.config.get("memory_system_name"),
                        self.config.get("memory_url"),
                    )
                except Exception as e:
                    _log.warning(
                        "Failed to create memory client from memory/: %s. Memory will be disabled for this run.",
                        e,
                        exc_info=True,
                    )
                    self.memory_client = None
            else:
                _log.warning("memory/ directory not found at %s. Memory will be disabled.", _memory_dir)
                self.memory_client = None

    @staticmethod
    def evaluate_answer_with_judge(client, question: str, predicted_answer: Optional[str], correct_answer: str, model: str = "gpt-4.1", max_output_tokens: int = 8000) -> Dict:
        import sys
        _web_search_env = Path(__file__).resolve().parent / "web_search_env"
        if str(_web_search_env) not in sys.path:
            sys.path.insert(0, str(_web_search_env))
        from evaluate_with_openai import create_judge_prompt, call_openai_judge, parse_judge_response
        predicted_answer = predicted_answer or ""
        prompt = create_judge_prompt(question, predicted_answer, correct_answer)
        response = call_openai_judge(
            client=client,
            prompt=prompt,
            model=model,
            max_output_tokens=max_output_tokens,
            reasoning_effort=None,
            system_prompt=None
        )
        judge_response_text = (
            response.output_text
            if hasattr(response, "output_text")
            else ""
        )
        parsed = parse_judge_response(judge_response_text)
        return {
            "correct": parsed.get("correct", False),
            "confidence": parsed.get("confidence", 0.0),
            "reasoning": parsed.get("reasoning", ""),
            "extracted_final_answer": parsed.get("extracted_final_answer", ""),
            "parse_error": parsed.get("parse_error", False)
        }

    @staticmethod
    def load_ground_truth(jsonl_path: Path) -> Dict[str, Dict]:
        """Load ground truth with query, answer, evidence, and gold documents."""
        gt: Dict[str, Dict] = {}
        if not jsonl_path.exists():
            return gt
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                gt[str(obj["query_id"])] = {
                    "query": obj.get("query", ""),
                    "answer": obj.get("answer", ""),
                    "evidence": obj.get("evidence_docs", obj.get("evidence", [])),
                    "gold_docs": obj.get("gold_docs", []),
                }
        return gt

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Return initial observation (query_id, subqueries count)."""
        subqueries = self.config.get("subqueries") or []
        self._current_observation = {
            "query_id": self.query_id,
            "subqueries_count": len(subqueries),
        }
        return self._current_observation

    def run_full(self) -> Dict[str, Any]:
        """
        Run full browsecomp flow on server: subqueries then final query.
        Uses config (task_id, subqueries, correct_answers, output_dir, script_path, ...).
        """
        import os
        import sys
        # Add project root so agent.search is importable (no package name required)
        _root = Path(__file__).resolve().parents[2]  # env_systems -> env -> project root
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from agent.search import run_query_with_agent_and_memory
        from harness.trace import Recorder

        # 轨迹在这里记，不在 run_search.py 里：subquery 的循环整个跑在 env server
        # 进程内（一次 `run_sequential` 就是全流程），跑 run 的那个进程只看得到最后
        # 一次合并查询的结果。事件跟着响应回去，由 run_search.py 落 trajectory.json。
        recorder = Recorder()

        subqueries = self.config.get("subqueries") or []
        correct_answers = self.config.get("correct_answers") or []
        output_dir = self.config.get("output_dir")
        if output_dir is not None and not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        _env_systems = Path(__file__).resolve().parent
        ground_truth_path = self.config.get("ground_truth_path")
        if ground_truth_path is None:
            ground_truth_path = _env_systems / "web_search_env/data/browsecomp_plus_decrypted.jsonl"
        else:
            if not isinstance(ground_truth_path, Path):
                ground_truth_path = Path(ground_truth_path)
            if not ground_truth_path.is_absolute():
                ground_truth_path = _env_systems / ground_truth_path

        # Create judge client on server
        try:
            judge_client = make_judge_client()
        except Exception:
            judge_client = None

        # Pad correct_answers
        while len(correct_answers) < len(subqueries):
            correct_answers.append("")
        original_query = self.config.get("original_query") or (subqueries[0] if subqueries else "")

        _corpus = self.config.get("corpus_path", "web_search_env/data/corpus.jsonl")
        if not Path(_corpus).is_absolute():
            _corpus = str(_env_systems / _corpus)

        # Run each subquery, add to memory
        for i, (subquery, correct_answer) in enumerate(zip(subqueries, correct_answers)):
            sq_dir = (output_dir / f"query_{self.query_id}" / "subqueries" / f"subquery_{i + 1}") if output_dir else None
            if sq_dir:
                sq_dir.mkdir(parents=True, exist_ok=True)
            result = run_query_with_agent_and_memory(
                query=subquery,
                memory_client=self.memory_client,
                output_dir=sq_dir,
                script_path=self.script_path,
                index_path=self.index_path,
                corpus_path=_corpus,
                model=self.agent_model,
                model_name=self.model_name,
                mcp_url=self.mcp_url,
                provider=self.provider,
                mcp_name=self.mcp_name,
                gpu=self.gpu,
                max_iterations=self.max_iterations,
                step_memory=self.step_memory,
            )
            answer = result.get("answer", "")
            # 记忆注入发生在 `run_query_with_agent_and_memory` 内部，拿不到调用前的
            # 时刻，所以三个事件都在这一轮结束后按顺序补记。顺序和 travel / math 一致：
            # wrap 开一轮，env.step 关一轮。
            recorder.record("memory.wrap_user_prompt",
                            question=subquery,
                            wrapped_prompt=result.get("query_with_memory", subquery))
            recorder.record("agent.act",
                            subquery_idx=i + 1,
                            query=subquery,
                            answer=answer,
                            tool_call_counts=result.get("tool_call_counts"),
                            retrieved_docids=result.get("retrieved_docids", []))
            # subquery 这一层没有判分（只有最后的合并查询过 judge），但每条 subquery
            # 的标准答案是有的，一起记下来，事后能看出是哪一轮开始跑偏。
            recorder.record("env.step",
                            ground_truth=correct_answer,
                            result={"answer": answer, "judged": False})
            if self.memory_client:
                trace = result.get("trace", [])
                trace_summary = ""
                if trace:
                    try:
                        trace_summary = json.dumps(trace, ensure_ascii=False)
                    except Exception:
                        trace_summary = str(trace)
                memory_entry = f"Subquery {i + 1}: {subquery}\n\nPredicted Answer: {answer}"
                if trace_summary:
                    memory_entry += f"\n\nTrace: {trace_summary}"
                self.memory_client.add(memory_entry)
                recorder.record("memory.add", chunk=memory_entry)

        # Final combined query (use resolved _corpus so path is correct on server)
        final = BrowseCompPlusEnvironment.run_final_query(
            query_id=self.query_id,
            original_query=original_query,
            subqueries=subqueries,
            output_dir=output_dir,
            memory_client=self.memory_client,
            script_path=self.script_path,
            index_path=self.index_path,
            corpus_path=_corpus,
            agent_model=self.agent_model,
            model_name=self.model_name,
            mcp_url=self.mcp_url,
            provider=self.provider,
            mcp_name=self.mcp_name,
            gpu=self.gpu,
            qrel_data=self.qrel_data,
            client=judge_client,
            judge_model=self.judge_model,
            ground_truth_path=ground_truth_path,
            step_memory=self.step_memory,
            store_eval_in_memory=self.store_eval_in_memory,
            recorder=recorder,
        )
        # `events` 是新字段，老字段一个没动——run_search.py 之外的读法不受影响。
        final["events"] = recorder.events
        return final

    # --- Final Query Logic as a Standalone Function ---
    @staticmethod
    def run_final_query(
        query_id,
        original_query,
        subqueries,
        output_dir,
        memory_client,
        script_path,
        index_path,
        corpus_path,
        agent_model,
        model_name,
        mcp_url,
        provider,
        mcp_name,
        gpu,
        qrel_data,
        client,
        judge_model,
        ground_truth_path: Optional[Path] = None,
        step_memory: bool = False,
        store_eval_in_memory: bool = False,
        recorder=None,
    ):
        """
        Run the final query with full memory context, judge, and return result dict.
        """
        import sys
        _root = Path(__file__).resolve().parents[2]
        _env_systems = Path(__file__).resolve().parent  # env/env_systems
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from agent.search import run_query_with_agent_and_memory
        # Construct final query from ALL subqueries combined
        final_query = original_query
        final_correct_answer = ""
        if ground_truth_path is None:
            ground_truth_path = _env_systems / "web_search_env/data/browsecomp_plus_decrypted.jsonl"
        else:
            ground_truth_path = Path(ground_truth_path)
            if not ground_truth_path.is_absolute():
                ground_truth_path = _env_systems / ground_truth_path
        print(f"Loading ground truth from {ground_truth_path}")
        ground_truth_data = BrowseCompPlusEnvironment.load_ground_truth(ground_truth_path)
        if ground_truth_data and query_id in ground_truth_data:
            final_correct_answer = ground_truth_data[query_id].get("answer", "")
        print(f"\n--- Final Combined Query ---")
        print(f"Combined Question (from all {len(subqueries)} subqueries): {final_query[:300]}...")
        # Create output directory for final query
        final_output_dir = output_dir / f"query_{query_id}" / "final_query" if output_dir else None
        # Run agent for final query with full memory
        print("Running agent for final query with full memory...")
        final_result_data = run_query_with_agent_and_memory(
            query=final_query,
            memory_client=memory_client,
            output_dir=final_output_dir,
            script_path=script_path,
            index_path=index_path,
            corpus_path=corpus_path,
            model=agent_model,
            model_name=model_name,
            mcp_url=mcp_url,
            provider=provider,
            mcp_name=mcp_name,
            gpu=gpu,
            step_memory=step_memory,
        )
        final_predicted_answer = final_result_data["answer"]
        final_trace = final_result_data["trace"]
        final_tool_calls = final_result_data["tool_call_counts"]
        final_retrieved_docids = final_result_data["retrieved_docids"]
        final_memory_context = final_result_data.get("memory_context", "")
        final_query_with_memory = final_result_data.get("query_with_memory", final_query)
        full_result = final_result_data.get("full_result", {})
        final_api_calls = full_result.get("api_calls", [])
        final_metadata = full_result.get("metadata", {})
        final_usage = full_result.get("usage", {})
        final_status = full_result.get("status", "completed")
        # Ensure predicted answer is a string
        final_predicted_answer = final_predicted_answer or ""
        # Save memory context to a separate file
        if output_dir:
            memory_context_file = output_dir / f"query_{query_id}" / "final_query_memory_context.json"
            memory_context_file.parent.mkdir(parents=True, exist_ok=True)
            memory_context_data = {
                "query_id": query_id,
                "final_query": final_query,
                "memory_context": final_memory_context,
                "query_with_memory": final_query_with_memory,
                "memory_context_length": len(final_memory_context),
                "total_query_length": len(final_query_with_memory)
            }
            with open(memory_context_file, 'w', encoding='utf-8') as f:
                json.dump(memory_context_data, f, indent=2, ensure_ascii=False)
            print(f"[DEBUG] Saved memory context to: {memory_context_file}")
        if recorder is not None:
            # 最后这一轮：带全部记忆的合并查询，和上面每条 subquery 同一套事件。
            recorder.record("memory.wrap_user_prompt",
                            question=final_query,
                            wrapped_prompt=final_query_with_memory)
            recorder.record("agent.act",
                            subquery_idx=None,
                            query=final_query,
                            answer=final_predicted_answer,
                            tool_call_counts=final_tool_calls,
                            retrieved_docids=final_retrieved_docids)

        # Calculate final recall
        final_recall = None
        if qrel_data and query_id in qrel_data:
            relevant_docids = qrel_data[query_id]
            if len(relevant_docids) > 0:
                retrieved_set = set(final_retrieved_docids)
                relevant_set = set(relevant_docids)
                final_recall = len(retrieved_set & relevant_set) / len(relevant_set)
        print(f"Final Predicted Answer: {final_predicted_answer[:200]}...")
        print(f"Final Tool calls: {final_tool_calls}")
        if final_recall is not None:
            print(f"Final Recall: {final_recall*100:.2f}%")
        # Evaluate final answer
        final_evaluation = BrowseCompPlusEnvironment.evaluate_answer_with_judge(
            client=client,
            question=final_query,
            predicted_answer=final_predicted_answer,
            correct_answer=final_correct_answer,
            model=judge_model
        )
        is_final_correct = final_evaluation["correct"]
        print(f"Final Evaluation: {'CORRECT' if is_final_correct else 'INCORRECT'}")
        if recorder is not None:
            recorder.record("env.step",
                            ground_truth=final_correct_answer,
                            result={"answer": final_predicted_answer,
                                    "judged": True,
                                    "judgement": final_evaluation,
                                    "recall": final_recall})
        return {
            "final_query": final_query,
            "query": final_query,
            "trace": final_trace,
            "predicted_answer": final_predicted_answer,
            "correct_answer": final_correct_answer,
            "judgement": final_evaluation,
            "evaluation": final_evaluation,
            "tool_call_counts": final_tool_calls,
            "retrieved_docids": final_retrieved_docids,
            "recall": final_recall,
            "memory_context": final_memory_context,
            "api_calls": final_api_calls,
            "metadata": final_metadata,
            "usage": final_usage,
            "status": final_status,
        }
    def step(self, action: Any, ground_truth: Any = None, need_judge: bool = False, **kwargs: Any) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
        """
        Execute one step. If action is {"command": "run_sequential"}, run full flow and return result.
        Otherwise run agent for single query. Returns (observation, reward, info) for env server compatibility.
        """
        # Server-driven full run: one step runs the entire sequential + final flow
        if isinstance(action, dict) and action.get("command") == "run_sequential":
            result = self.run_full()
            return (result, None, {"done": True})

        observation = {}
        reward = None
        info = {}
        if self.agent:
            agent_output = self.agent.run_query_with_memory(action)
            observation["agent_output"] = agent_output
            self.history.append({"action": action, "agent_output": agent_output})
        if need_judge and self.agent and self.client:
            judge_result = self.evaluate_answer_with_judge(
                self.client,
                action,
                observation.get("agent_output", ""),
                ground_truth,
                model=self.judge_model
            )
            reward = judge_result.get("confidence")
            info["judge_result"] = judge_result
        return (observation, reward, info)

    def step_sequential(self, subqueries: List[str], correct_answers: List[str], output_dir: Optional[Path] = None) -> Dict:
        """
        Sequentially process a list of subqueries, run agent, judge, update memory, aggregate results.
        Returns a dict with all subquery results and final result.
        """
        results = {
            "subquery_results": [],
            "final_result": None
        }
        for i, (subquery, correct_answer) in enumerate(zip(subqueries, correct_answers)):
            obs, reward, done, info = self.step(
                action=subquery,
                ground_truth=correct_answer,
                need_judge=True
            )
            results["subquery_results"].append({
                "subquery_index": i + 1,
                "subquery": subquery,
                "predicted_answer": obs.get("agent_output", ""),
                "correct_answer": correct_answer,
                "judgement": info.get("judge_result", {}),
                "reward": reward
            })
        return results


    def get_observation(self) -> Dict[str, Any]:
            return self._current_observation or self.reset()

    def close(self):
            self.history = []
            self.browser_state = {}
            self._current_observation = None
