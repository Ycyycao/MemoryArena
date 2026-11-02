"""测试的公共夹具。

测试全部零 API 开销：碰 LLM 的地方注入假 client，碰磁盘的地方用 tmp_path。
仓库里原有的 `memory/test_memory.py` 和 `env/test_env_system.py` 是要连 server 的
冒烟脚本（而且前者 import 的 `MemActBench` 随 submodule 一起没了），不在这里收集，
见根 pyproject.toml 的 `norecursedirs`。

`MemoryArena/` 本身在 sys.path 上（editable 安装写了 .pth，见 MemoryArena/pyproject.toml
的 dev-mode-dirs），所以 `import harness` / `import run_travel` / `import agent` 直接可用。
"""

import pytest


@pytest.fixture
def env_dirs(tmp_path, monkeypatch):
    """把 harness.settings 读的三个目录指到 tmp_path。

    settings 在 import 时就 load_dotenv() 了，不隔离的话测试会读写真实的
    results/ 和 configs/。
    """
    dirs = {}
    for var, name in (("MEMORYARENA_DATA", "data"),
                      ("MARENA_RESULTS", "results"),
                      ("MARENA_CONFIGS", "configs")):
        path = tmp_path / name
        path.mkdir()
        monkeypatch.setenv(var, str(path))
        dirs[var] = path
    return dirs
