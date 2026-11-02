"""有状态 memory system 的落盘位置。

几个实现默认写 `./xxx_data/`，那是 memory server 进程的 cwd（`MemoryArena/memory/`），
会被 `git clean` 抹掉，而且跨 run 混在一起。统一挪到 `$MARENA_MEMORY_STATE`，
不设就退回 `$MARENA_RESULTS/_memory`。
"""

import os
from pathlib import Path


def state_dir(name: str = "") -> Path:
    """`<root>/_memory/<name>/`，目录会被建出来。"""
    base = os.getenv("MARENA_MEMORY_STATE")
    if not base:
        results = os.getenv("MARENA_RESULTS")
        if not results:
            raise SystemExit(
                "环境变量 MARENA_RESULTS 或 MARENA_MEMORY_STATE 至少要设一个，"
                "否则有状态的 memory system 不知道该往哪写。"
            )
        base = os.path.join(results, "_memory")
    path = Path(base).expanduser()
    if name:
        path = path / name
    path.mkdir(parents=True, exist_ok=True)
    return path
