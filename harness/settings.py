"""路径配置。全部来自环境变量（.env），代码里不拼路径。"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _required_dir(name: str) -> Path:
    value = os.getenv(name)
    if not value:
        raise SystemExit(
            f"环境变量 {name} 没有设置。把它写进仓库根目录的 .env，"
            f"或者用 `uv run --env-file .env` 跑。"
        )
    path = Path(value).expanduser()
    if not path.is_dir():
        raise SystemExit(f"{name}={path} 不是一个存在的目录")
    return path


def data_dir() -> Path:
    """HF 数据副本的根目录，下面按 hf_config 分子目录。"""
    return _required_dir("MEMORYARENA_DATA")


def results_dir() -> Path:
    """所有 run 输出的根目录，下面按 run-id 分子目录。"""
    return _required_dir("MARENA_RESULTS")


def configs_dir() -> Path:
    """我们自己的 config 根目录，按环境分子目录。"""
    return _required_dir("MARENA_CONFIGS")
