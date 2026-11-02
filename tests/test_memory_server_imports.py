"""The server imports a memory system only when it is created, so a run needs only that system's dependencies."""
import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_nothing_imports_memory_systems_at_module_level():
    assert not [n for n in ast.parse((REPO / "memory/memory_systems/__init__.py").read_text()).body
                if isinstance(n, (ast.Import, ast.ImportFrom))]
    tree = ast.parse((REPO / "memory/server.py").read_text())
    assert not [n for n in tree.body if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("memory_systems")]


def test_nothing_uses_litellm():
    hits = [p for p in REPO.rglob("*.py") if re.search(r"^\s*(import|from) litellm", p.read_text(errors="ignore"), re.M)]
    assert not hits, hits
