"""config 解析和结果目录规则。

`run_dir()` 就是 CLAUDE.md 硬规则二（「结果只落
`results/<memory-system>/<run-id>/`」）的实现；占位符解析决定 config 能不能进版本库。
"""

import json

import pytest

from harness import config as config_mod
from harness import settings


# --------------------------------------------------------------- 占位符

def test_resolves_placeholders_from_env(monkeypatch):
    monkeypatch.setenv("SOME_URL", "http://127.0.0.1:18780/agent/v1")
    got = config_mod._resolve({"agent": {"base_url": "<SOME_URL>", "model": "gpt-5-mini"}})
    assert got == {"agent": {"base_url": "http://127.0.0.1:18780/agent/v1",
                             "model": "gpt-5-mini"}}


def test_resolves_inside_lists(monkeypatch):
    monkeypatch.setenv("X", "v")
    assert config_mod._resolve(["<X>", {"k": "<X>"}, 1]) == ["v", {"k": "v"}, 1]


def test_missing_env_var_is_a_hard_stop(monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    with pytest.raises(SystemExit):
        config_mod._resolve({"a": "<DEFINITELY_NOT_SET>"})


def test_empty_env_var_counts_as_missing(monkeypatch):
    """空串当没设置——否则 base_url 会变成空串，报错跑到 SDK 里才出现。"""
    monkeypatch.setenv("EMPTY_ONE", "")
    with pytest.raises(SystemExit):
        config_mod._resolve({"a": "<EMPTY_ONE>"})


def test_non_placeholder_strings_pass_through():
    assert config_mod._resolve("plain") == "plain"
    assert config_mod._resolve("http://x/<y>/z") == "http://x/<y>/z"


def test_only_upper_snake_names_count_as_placeholders():
    """判定收紧成「大写开头、只有大写数字下划线」。

    原来的规则是 startswith("<") and endswith(">")，任何被尖括号整个包住的字面值
    都会被当成占位符、报一句读不懂的错。search 的 config 要在一个字符串里拼路径
    （`<MARENA_BROWSECOMP_DATA>/corpus.jsonl`），换成正则之后顺手把这条收紧了，
    prompt 片段里的 `<thinking>` 这种不会再被误伤。
    """
    assert config_mod._resolve("<not an env var name>") == "<not an env var name>"
    assert config_mod._resolve("<thinking>") == "<thinking>"


def test_placeholder_embedded_in_a_longer_string(monkeypatch):
    """一个目录变量拼出底下的多个文件，换机器只改 .env 一行。"""
    monkeypatch.setenv("MARENA_BROWSECOMP_DATA", "/data/browsecomp")
    got = config_mod._resolve({
        "index_path": "<MARENA_BROWSECOMP_DATA>/embeddings/shard*.index",
        "corpus_path": "<MARENA_BROWSECOMP_DATA>/data/corpus.jsonl",
    })
    assert got == {"index_path": "/data/browsecomp/embeddings/shard*.index",
                   "corpus_path": "/data/browsecomp/data/corpus.jsonl"}


def test_two_placeholders_in_one_string(monkeypatch):
    monkeypatch.setenv("A_DIR", "/a")
    monkeypatch.setenv("B_NAME", "b.json")
    assert config_mod._resolve("<A_DIR>/x/<B_NAME>") == "/a/x/b.json"


def test_embedded_placeholder_missing_is_still_a_hard_stop(monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    with pytest.raises(SystemExit):
        config_mod._resolve("/prefix/<DEFINITELY_NOT_SET>/suffix")


# --------------------------------------------------------------- 路径规则

def test_run_dir_is_results_slash_memory_system_slash_run_id(env_dirs):
    cfg = {"memory": {"memory_system_name": "simplemem"}}
    got = config_mod.run_dir(cfg, "travel-simplemem-case0-8p")
    assert got == env_dirs["MARENA_RESULTS"] / "simplemem" / "travel-simplemem-case0-8p"


def test_resolve_path_absolute_passes_through(env_dirs, tmp_path):
    p = tmp_path / "abs.json"
    assert config_mod.resolve_path(p) == p


def test_resolve_path_relative_goes_under_configs_dir(env_dirs):
    got = config_mod.resolve_path("travel_planner/x.json")
    assert got == env_dirs["MARENA_CONFIGS"] / "travel_planner" / "x.json"


def test_settings_refuses_an_unset_dir(monkeypatch):
    monkeypatch.delenv("MARENA_RESULTS", raising=False)
    with pytest.raises(SystemExit):
        settings.results_dir()


def test_settings_refuses_a_path_that_is_not_a_dir(tmp_path, monkeypatch):
    f = tmp_path / "afile"
    f.write_text("x")
    monkeypatch.setenv("MARENA_RESULTS", str(f))
    with pytest.raises(SystemExit):
        settings.results_dir()


# --------------------------------------------------------------- load()

def test_load_resolves_and_rewrites_output_dir(env_dirs, monkeypatch):
    monkeypatch.setenv("MARENA_ENV_URL", "http://127.0.0.1:18782")
    cfg_file = env_dirs["MARENA_CONFIGS"] / "c.json"
    cfg_file.write_text(json.dumps({
        "memory": {"memory_system_name": "simplemem"},
        "env": {"base_url": "<MARENA_ENV_URL>"},
        "output": {"json_output_dir": "./whatever"},
    }), encoding="utf-8")

    cfg = config_mod.load(cfg_file, run_id="math-simplemem-case0")
    assert cfg["env"]["base_url"] == "http://127.0.0.1:18782"
    assert cfg["output"]["json_output_dir"] == str(
        env_dirs["MARENA_RESULTS"] / "simplemem" / "math-simplemem-case0")


def test_load_without_run_id_leaves_output_alone(env_dirs):
    cfg_file = env_dirs["MARENA_CONFIGS"] / "c.json"
    cfg_file.write_text(json.dumps({
        "memory": {"memory_system_name": "mem0"},
        "output": {"json_output_dir": "/keep/me"},
    }), encoding="utf-8")
    assert config_mod.load(cfg_file)["output"]["json_output_dir"] == "/keep/me"


# --------------------------------------------------------------- 真 config

def test_every_checked_in_config_still_parses():
    """占位符名字改了、config 少了一段，这里会红。不连网、不花钱。"""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    files = sorted((repo / "harness_configs").rglob("*.json"))
    assert files, "configs/ 下没有 config"
    for f in files:
        cfg = json.loads(f.read_text(encoding="utf-8"))
        assert "memory" in cfg and "memory_system_name" in cfg["memory"], f
        assert "agent" in cfg and "model_name" in cfg["agent"], f
        # agent_mode 可以不写（默认 toolcall），写了就必须是认得的值
        mode = cfg["agent"].get("agent_mode", "toolcall")
        assert mode in {"toolcall", "react"}, (f, mode)


def test_the_servers_get_the_local_data_paths():
    from harness import serve
    assert {"MEMORYARENA_DATA", "TRAVELPLANNER_DB"} <= set(serve.PASSTHROUGH)
    assert {"SIMPLEMEM_TEMPLATE_THINKING", "SIMPLEMEM_MAX_TOKENS"} <= set(serve.PASSTHROUGH)
