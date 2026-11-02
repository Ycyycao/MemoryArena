"""progressive_search（BrowseComp-Plus）这条路的接线。

这条路和 travel / math 有一处结构性的不同：**agent 不在跑 run 的进程里**。
一次 `run_sequential` 请求过去，subquery 的整个循环都在 env server 进程里跑，
agent 本身是 env server `subprocess.run` 出来的第三个进程。于是有三件事只能靠
显式接线保证，这个文件钉住它们：

1. 子进程用哪个解释器、哪个网关入口、归哪条 run（`runctx.child_env`）；
2. 结果落哪儿、task_id 带不带 run-id 前缀（硬规则二 + 归属切片）；
3. env server 记的事件怎么变成 trajectory.json。

全部零 API 开销：env client 是假的，落盘用 tmp_path。
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import run_search
from harness import runctx

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- 子进程环境

def test_child_env_swaps_the_gateway_role():
    """env server 的 OPENAI_BASE_URL 指着 /judge，agent 子进程必须换成 /agent。

    不换的话，agent 的对话和检索用的 embedding 全被记成判分流量。
    """
    got = runctx.child_env({"OPENAI_BASE_URL": "http://127.0.0.1:18780/judge"})
    assert got["OPENAI_BASE_URL"] == "http://127.0.0.1:18780/agent"
    assert got["MARENA_LLM_ROLE"] == "agent"


def test_child_env_leaves_a_non_gateway_base_url_alone():
    """指向真 OpenAI 的 base_url 结尾是 /v1，不是角色段，不能动它。"""
    for url in ("https://api.openai.com/v1", "https://api.shubiaobiao.cn/v1"):
        assert runctx.child_env({"OPENAI_BASE_URL": url})["OPENAI_BASE_URL"] == url


def test_child_env_does_not_touch_the_parent_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:18780/judge")
    import os
    runctx.child_env(os.environ.copy())
    assert os.environ["OPENAI_BASE_URL"] == "http://127.0.0.1:18780/judge"


def test_request_id_crosses_the_process_boundary(monkeypatch):
    """REQUEST_ID 是 contextvar，跨不过 fork。父进程放进环境变量，子进程接回来。"""
    token = runctx.REQUEST_ID.set("search-run__abc")
    try:
        child = runctx.child_env({})
    finally:
        runctx.REQUEST_ID.reset(token)
    assert child[runctx.REQUEST_ID_ENV] == "search-run__abc"

    # 子进程那侧：环境变量 -> REQUEST_ID
    monkeypatch.setenv(runctx.REQUEST_ID_ENV, child[runctx.REQUEST_ID_ENV])
    token = runctx.REQUEST_ID.set(None)
    try:
        assert runctx.adopt_from_env() == "search-run__abc"
        assert runctx.REQUEST_ID.get() == "search-run__abc"
    finally:
        runctx.REQUEST_ID.reset(token)


def test_adopt_from_env_without_the_variable_is_a_no_op(monkeypatch):
    monkeypatch.delenv(runctx.REQUEST_ID_ENV, raising=False)
    assert runctx.adopt_from_env() is None


def test_agent_subprocess_uses_this_interpreter_not_whatever_python_is_on_path():
    """上游写的是 `["python", script]`。env server 是用 .venv 的解释器起的，但它的
    tmux 窗口里 `python` 很可能是别的（这台机器上是 conda base），子进程会拿错环境。
    """
    source = (REPO / "agent" / "search.py").read_text(encoding="utf-8")
    assert '"python"' not in source
    assert "sys.executable" in source


def test_agent_subprocess_runs_in_env_systems():
    """index/corpus 的相对默认值是按 env_systems 解析的，cwd 必须钉在那儿。"""
    import agent.search as mod
    assert mod._ENV_SYSTEMS.name == "env_systems"
    assert mod._ENV_SYSTEMS.is_dir()


# --------------------------------------------------------------- searcher 惰性 import

def test_searcher_package_does_not_pull_tevatron():
    """`from searcher.searchers import SearcherType` 曾经把 faiss_searcher 一起拉进来，
    而它顶层 import tevatron（要从 git 装）。我们只用 OpenAISearcher。

    子进程里验，免得同一个 pytest 进程里别的用例已经把模块 import 进 sys.modules。
    """
    env_dir = REPO / "env" / "env_systems" / "web_search_env"
    code = (
        "import sys;"
        "from searcher.searchers import SearcherType;"
        "print(SearcherType.get_choices());"
        "print('faiss_searcher' in ' '.join(sys.modules));"
        "print('tevatron' in sys.modules);"
        "print(SearcherType.get_searcher_class('openai').__name__)"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=env_dir,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    choices, faiss_imported, tevatron_imported, cls = out.stdout.strip().split("\n")
    assert "openai" in choices and "faiss" in choices     # 选项一个没少
    assert faiss_imported == "False"
    assert tevatron_imported == "False"
    assert cls == "OpenAISearcher"


def test_custom_searcher_does_not_print_the_api_key():
    """上游在 custom_searcher.py 顶层 `print("ENV OPENAI_API_KEY:", repr(...))`。

    那一行会把 key 打进 env server 的日志和 results/_servers/*.log。读环境变量本身
    没问题，打印出来才是问题，所以只查 print。
    """
    source = (REPO / "env" / "env_systems" / "web_search_env"
              / "searcher" / "searchers" / "custom_searcher.py").read_text(encoding="utf-8")
    for line in source.splitlines():
        assert not (line.lstrip().startswith("print(") and "API_KEY" in line), line


# --------------------------------------------------------------- 一整条假 run

class _FakeEnvClient:
    """假 env server。记下 task_id 和 env_config，回一份形状完整的结果。"""

    created = []

    def __init__(self, task_id, timeout, env_name, base_url, env_config, session=None):
        self.task_id = task_id
        self.env_config = env_config
        self.env_name = env_name
        self.session = session
        _FakeEnvClient.created.append(self)

    def reset(self):
        return {}

    def step(self, action):
        assert action == {"command": "run_sequential"}
        return {"observation": {
            "predicted_answer": "Queen Arwa University",
            "correct_answer": "Queen Arwa University",
            "judgement": {"correct": True, "confidence": 90},
            "recall": 0.5,
            "retrieved_docids": ["18639"],
            "tool_call_counts": {"search": 3},
            "trace": [],
            "api_calls": [],
            "usage": {},
            "status": "completed",
            "final_query": "q",
            "metadata": {},
            # env server 在 run_full 里记的事件，跟着响应回来
            "events": [
                {"t": 1.0, "event": "memory.wrap_user_prompt",
                 "question": "sub1", "wrapped_prompt": "w"},
                {"t": 2.0, "event": "agent.act", "subquery_idx": 1, "query": "sub1",
                 "answer": "a", "tool_call_counts": {"search": 3},
                 "retrieved_docids": ["18639"]},
                {"t": 3.0, "event": "env.step", "ground_truth": "gold",
                 "result": {"answer": "a", "judged": False}},
                {"t": 4.0, "event": "memory.add", "chunk": "c"},
            ],
        }}

    def close(self):
        pass


@pytest.fixture
def search_run(env_dirs, monkeypatch, tmp_path):
    """一条能跑通的假 run：config、ground truth、subquery 文件都在 tmp_path 下。"""
    _FakeEnvClient.created = []
    data_dir = tmp_path / "browsecomp" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "browsecomp_plus_decrypted.jsonl").write_text(
        json.dumps({"query_id": 454, "query": "原始问题", "answer": "Queen Arwa University",
                    "evidence_docs": [], "gold_docs": []}) + "\n", encoding="utf-8")
    # question 的最后一项是合并后的最终查询，前面的才是 subquery（agent/search.py
    # 的 load_correct_answers 就是这么切的）。这里三问 -> 两条 subquery + 一次最终查询。
    (data_dir / "browsecomp_all_jsons.jsonl").write_text(
        json.dumps({"id": 454,
                    "question": ["sub1", "sub2", "最终查询"],
                    "answer": ["a1", "a2", "最终答案"]}) + "\n",
        encoding="utf-8")
    qrels = tmp_path / "browsecomp" / "topics-qrels"
    qrels.mkdir()
    (qrels / "qrel_evidence.txt").write_text("454 Q0 18639 1\n454 Q0 99999 1\n", encoding="utf-8")

    monkeypatch.setenv("MARENA_BROWSECOMP_DATA", str(tmp_path / "browsecomp"))
    monkeypatch.setenv("MARENA_MEMORY_URL", "http://127.0.0.1:18781")
    monkeypatch.setenv("MARENA_ENV_URL", "http://127.0.0.1:18782")

    cfg_dir = env_dirs["MARENA_CONFIGS"] / "web_search"
    cfg_dir.mkdir()
    (cfg_dir / "search-fake.json").write_text(json.dumps({
        "task_name": "search",
        "agent": {"model_name": "gpt-5-mini", "embedding_model": "text-embedding-3-small",
                  "max_iterations": 35, "provider": "openai"},
        "memory": {"memory_system_name": "long_context", "memory_url": "<MARENA_MEMORY_URL>",
                   "use_step_memory": False, "no_memory": False},
        "env": {"env_server_url": "<MARENA_ENV_URL>",
                "index_path": "<MARENA_BROWSECOMP_DATA>/embeddings/shard*.index",
                "corpus_path": "<MARENA_BROWSECOMP_DATA>/data/corpus.jsonl"},
        "task_specific": {"query_ids": ["454"],
                          "data_dir": "<MARENA_BROWSECOMP_DATA>/data",
                          "qrel_evidence": "<MARENA_BROWSECOMP_DATA>/topics-qrels/qrel_evidence.txt",
                          "judge_model": "gpt-4.1"},
        "output": {"output_dir": "<MARENA_RESULTS>/long_context/search-fake"},
    }), encoding="utf-8")

    monkeypatch.setattr(run_search, "EnvironmentClient", _FakeEnvClient)
    monkeypatch.setattr(sys, "argv",
                        ["run_search", "-c", "web_search/search-fake.json",
                         "--run-id", "search-longcontext-case454"])
    run_search.main()
    return env_dirs["MARENA_RESULTS"] / "long_context" / "search-longcontext-case454"


def test_results_land_under_memory_system_and_run_id(search_run):
    """硬规则二。config 里的 output_dir 只是默认值，run-id 一给就以 run_dir 为准。"""
    assert search_run.is_dir()
    assert (search_run / "trajectory.json").exists()
    assert (search_run / "resolved-config.json").exists()


def test_resolved_config_has_no_placeholders_left(search_run):
    cfg = json.loads((search_run / "resolved-config.json").read_text(encoding="utf-8"))
    assert "<" not in cfg["env"]["index_path"]
    assert cfg["env"]["env_server_url"] == "http://127.0.0.1:18782"


def test_task_id_carries_the_run_id_prefix():
    """env server 的中间件拿 task_id 当 REQUEST_ID，这条 run 在 server 侧的 LLM
    调用全靠这个前缀归位。没有前缀的话 llm_journal 切片时捡不到。
    """
    (client,) = _FakeEnvClient.created
    assert client.task_id.startswith("search-longcontext-case454__")
    assert client.env_name == "browsecomp-plus"


def test_subqueries_come_from_browsecomp_all_jsons():
    """subquery 列表不含最后那一问——那是最终查询，env 侧单独跑、单独判分。"""
    (client,) = _FakeEnvClient.created
    assert client.env_config["subqueries"] == ["sub1", "sub2"]
    assert client.env_config["correct_answers"] == ["a1", "a2"]
    assert client.env_config["query_id"] == "454"
    # original_query 来自 ground truth（解密后的那份），不是 all_jsons 的最后一问
    assert client.env_config["original_query"] == "原始问题"


def test_trajectory_keeps_the_env_side_events(search_run):
    traj = json.loads((search_run / "trajectory.json").read_text(encoding="utf-8"))
    assert set(traj) == {"config_path", "config", "case", "num_queries", "trace"}
    names = [e["event"] for e in traj["trace"]]
    # 客户端这侧开一条 env.reset，其余四条来自 env server
    assert names == ["env.reset", "memory.wrap_user_prompt", "agent.act",
                     "env.step", "memory.add"]
    assert traj["num_queries"] == 1
    assert traj["case"] == "454"


def test_query_id_can_be_overridden_on_the_cli(env_dirs, monkeypatch, search_run):
    """config 里写死一个 query_id，冒烟时想换一个不必新建 config。"""
    _FakeEnvClient.created = []
    monkeypatch.setattr(run_search, "EnvironmentClient", _FakeEnvClient)
    monkeypatch.setattr(sys, "argv",
                        ["run_search", "-c", "web_search/search-fake.json",
                         "--run-id", "search-cli-override", "--query-id", "999"])
    with pytest.raises(SystemExit):        # 999 不在 ground truth 里
        run_search.main()
    assert _FakeEnvClient.created == []


# --------------------------------------------------------------- case1040 首跑暴露的三处

AGENT_SCRIPT = (REPO / "env" / "env_systems" / "web_search_env"
                / "search_agent" / "openai_client.py")


def test_agent_script_defines_main_once_and_adopts_the_request_id_first():
    """接 request-id 的那段曾被插在一个多出来的 `def main():` 里，紧跟着的真 main
    把它覆盖掉，于是从没执行过——首跑时 agent 的 3 次调用在网关里 request_id 全是
    None，harvest 一条都没切到。`--help` 能跑不说明任何问题，这里查结构。
    """
    import ast
    tree = ast.parse(AGENT_SCRIPT.read_text(encoding="utf-8"))
    mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    assert len(mains) == 1, f"openai_client.py 顶层有 {len(mains)} 个 main"

    src = ast.get_source_segment(AGENT_SCRIPT.read_text(encoding="utf-8"), mains[0])
    adopt = src.find("runctx.adopt_from_env()")
    tag = src.find("runctx.tag_openai_sdk()")
    first_client = src.find("openai.OpenAI(")
    assert adopt != -1 and tag != -1
    # 头是在 OpenAI 构造时取的，打补丁必须早于第一个客户端
    assert adopt < tag < first_client


def test_usage_without_completion_tokens_does_not_crash():
    """中转站回的 usage 可以没有 completion_tokens（case1040 首跑：prompt 261、
    total 261、字段缺失）。SDK 给 None，`int += None` 在第一轮就把 agent 打死。

    子进程里跑：这个脚本顶层 import `prompts` / `utils` 这类通名模块，放在 pytest
    进程里会和别处的同名模块串。
    """
    code = """
import types, json
import openai_client as oc

msg = types.SimpleNamespace(content="final answer", tool_calls=None)
usage = types.SimpleNamespace(prompt_tokens=261, completion_tokens=None, total_tokens=261)
resp = types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)

class Completions:
    def create(self, **kw):
        return resp
client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions()))

out = oc.run_conversation_with_tools(
    client, {"model": "m", "messages": [{"role": "user", "content": "q"}],
             "max_completion_tokens": 100}, tool_handler=None, max_iterations=2)
print(json.dumps(out[2]))
"""
    out = subprocess.run([sys.executable, "-c", code], cwd=AGENT_SCRIPT.parent,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    usage = json.loads(out.stdout.strip().splitlines()[-1])
    assert usage["input_tokens"] == 261
    assert usage["output_tokens"] == 0
    assert usage["total_tokens"] == 261


def _count_posts(session, status):
    """起一个本地 HTTP server，对每个 POST 回 `status`，返回它被打了几次。"""
    import http.server
    import threading

    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            hits.append(1)
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(status)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        session.post(f"http://127.0.0.1:{server.server_port}/env/step", json={}, timeout=5)
    finally:
        server.shutdown()
    return len(hits)


def test_run_sequential_is_not_replayed_on_a_server_error():
    """`run_sequential` 一次请求 = 整条 pipeline，500 回来时 server 那边已经跑了
    不知道多少。重放 = 从头再跑一遍，钱翻倍、同一个 task_id 的记忆重复写入。
    """
    assert _count_posts(run_search.no_replay_session(), 500) == 1


def test_the_default_env_session_does_replay_which_is_why_search_overrides_it():
    """对照：上游默认 session 对 POST 的 500 会重放。travel 的一步只是一次判分，
    重放代价小，所以只在 search 这里换掉，默认值不动。
    """
    from env.env_client import _build_session
    assert _count_posts(_build_session(), 500) > 1


def test_run_search_hands_the_no_replay_session_to_the_env_client(search_run):
    (client,) = _FakeEnvClient.created
    adapter = client.session.get_adapter("http://127.0.0.1:18782")
    retry = adapter.max_retries
    assert retry.status == 0 and retry.read == 0
    assert retry.connect == 3          # 请求没发出去时重试是安全的


def test_judge_client_carries_the_run_attribution(monkeypatch):
    """env server 里的判分调用是直接 openai.OpenAI() 发的，server 不装 SDK 补丁。
    case1040 首次跑通时，唯独这一次调用在网关记录里 request_id 为空。
    """
    from env.env_systems.browsecomp_plus_env import make_judge_client
    monkeypatch.setenv("OPENAI_API_KEY", "not-used")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:18780/judge")
    token = runctx.REQUEST_ID.set("search-longcontext-case1040__task")
    try:
        client = make_judge_client()
    finally:
        runctx.REQUEST_ID.reset(token)
    assert client.default_headers.get(runctx.HEADER) == "search-longcontext-case1040__task"


def test_judge_client_without_a_run_has_no_header(monkeypatch):
    from env.env_systems.browsecomp_plus_env import make_judge_client
    monkeypatch.setenv("OPENAI_API_KEY", "not-used")
    token = runctx.REQUEST_ID.set(None)
    try:
        client = make_judge_client()
    finally:
        runctx.REQUEST_ID.reset(token)
    assert runctx.HEADER not in client.default_headers
