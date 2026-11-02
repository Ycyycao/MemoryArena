"""读我们自己的 config。

上游 config 里 `<OPENAI_BASE_URL>` 这类尖括号占位符要换成环境变量的实际值——
这样 config 本身可以进版本库，密钥和 endpoint 留在 .env 里。
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from . import settings


# 占位符：`<MARENA_ENV_URL>`。只认全大写加下划线，免得把 config 里正常的尖括号
# 文本（比如 prompt 片段里的 `<thinking>`）也当成占位符。
_PLACEHOLDER = re.compile(r"<([A-Z][A-Z0-9_]*)>")


def _lookup(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"config 里的占位符 <{name}> 需要环境变量 {name}，但它没有设置")
    return value


def _resolve(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _resolve(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(v) for v in node]
    if isinstance(node, str):
        # 整串是一个占位符时原样替换（老行为）；夹在别的字符里也替换，
        # 这样一个目录变量能拼出底下的多个文件：`<MARENA_BROWSECOMP_DATA>/corpus.jsonl`。
        return _PLACEHOLDER.sub(lambda m: _lookup(m.group(1)), node)
    return node


def resolve_path(value: str | Path) -> Path:
    """config 路径：绝对路径原样用，相对路径按 $MARENA_CONFIGS 解析。"""
    path = Path(value).expanduser()
    return path if path.is_absolute() else settings.configs_dir() / path


def run_dir(cfg: dict, run_id: str) -> Path:
    """输出目录：`$MARENA_RESULTS/<memory-system>/<run-id>/`。

    按 memory system 分，不按数据集分：我们比的是 baseline，同一个 baseline 在几个
    环境上的 run 放在一起才好对照。环境已经写在 run-id 里（`travel-*`、`math-*`），
    不需要再占一层目录。
    """
    return settings.results_dir() / cfg["memory"]["memory_system_name"] / run_id


def load(path: str | Path, run_id: Optional[str] = None) -> dict:
    cfg = _resolve(json.loads(resolve_path(path).read_text(encoding="utf-8")))
    if run_id:
        cfg["output"]["json_output_dir"] = str(run_dir(cfg, run_id))
    return cfg


def hf_config(cfg: dict) -> str:
    """config 指定的数据集分支名，同时也是 data/memoryarena/ 下的子目录名。"""
    return cfg["task_specific"]["dataset"]["hf_config"]
