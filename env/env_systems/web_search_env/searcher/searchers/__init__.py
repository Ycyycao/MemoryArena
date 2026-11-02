"""
Searchers package for different search implementations.

上游在这里 eager import 全部 searcher，于是 `from searcher.searchers import SearcherType`
会把 `faiss_searcher` 一起拉进来，而它顶层 `import tevatron`——一个要从 git 装、
且会带上一串训练侧依赖的包。progressive_search 这条路只用 `OpenAISearcher`，
为一个用不到的 searcher 拖一个 git 依赖不划算，所以改成**按需 import**：
枚举里存模块路径，取 `.searcher_class` 时才真的 import。

对外行为不变：`get_choices()` 的取值和顺序、`get_searcher_class()` 的返回、
以及"pyserini 装了才有 bm25"这条规则，都和原来一样。区别只有报错时机——
选了一个依赖没装的 searcher，现在是取 `searcher_class` 时抛 ImportError，
而不是 import 这个包时就抛。
"""

import importlib
import importlib.util
from enum import Enum

from .base import BaseSearcher

# 原来靠 `try: from .bm25_searcher import BM25Searcher` 的成败决定 bm25 在不在
# 选项里，那条 import 唯一会失败的原因是 pyserini 没装。惰性化之后不能再靠
# import 探测（探测就等于 eager import），改成直接问 pyserini 在不在。
_BM25_AVAILABLE = importlib.util.find_spec("pyserini") is not None


class SearcherType(Enum):
    """Enum for managing available searcher types and their CLI mappings."""

    if _BM25_AVAILABLE:
        BM25 = ("bm25", ".bm25_searcher", "BM25Searcher")
    FAISS = ("faiss", ".faiss_searcher", "FaissSearcher")
    REASONIR = ("reasonir", ".faiss_searcher", "ReasonIrSearcher")
    OPENAI = ("openai", ".openai_searcher", "OpenAISearcher")
    CUSTOM = (
        "custom",
        ".custom_searcher",
        "CustomSearcher",
    )  # Your custom searcher class, yet to be implemented

    def __init__(self, cli_name, module_path, class_name):
        self.cli_name = cli_name
        self.module_path = module_path
        self.class_name = class_name

    @property
    def searcher_class(self):
        """真正 import 的地方。每次调用都走 `importlib`，模块缓存由 sys.modules 管。"""
        module = importlib.import_module(self.module_path, __package__)
        return getattr(module, self.class_name)

    @classmethod
    def get_choices(cls):
        """Get list of CLI choices for argument parser."""
        return [searcher_type.cli_name for searcher_type in cls]

    @classmethod
    def get_searcher_class(cls, cli_name):
        """Get searcher class by CLI name."""
        for searcher_type in cls:
            if searcher_type.cli_name == cli_name:
                return searcher_type.searcher_class
        raise ValueError(f"Unknown searcher type: {cli_name}")


__all__ = ["BaseSearcher", "SearcherType"]
