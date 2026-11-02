"""起/停 MemoryArena 的两个 server。

server 是长任务，放在 tmux session `memarena` 里，进程活过当前 shell。

两处和上游默认不一样：

- **env server 用 8011，不是上游的 8001。** 这台机器上 8001 被别的服务占了。
  上游把端口写死在 `env_server.py` 的 `__main__` 里，所以这里绕过 `__main__`，
  直接用 uvicorn 拉 app 对象。
- **三个窗口，不是两个。** 多出来的 `gateway` 是 LLM 网关（`harness.gateway`）：
  所有进程的 LLM 调用都从它过，按角色分入口，落一份完整轨迹。`.env` 里的
  `OPENAI_BASE_URL` 指向它的 `/agent` 入口，memory 和 env 两个窗口在这里被换成
  `/memory` 和 `/judge`——**角色靠入口路径区分，不用改任何被调用方的代码**。
  真正的上游只有网关知道（`MARENA_GATEWAY_UPSTREAM`）。
  网关没起来的话所有 LLM 调用都会失败，这是有意的：宁可响，不要静默绕过。
- **端口都在冷门段**（默认 18780/18781/18782），避开常见服务和系统临时端口
  （32768+）。改端口只改 `.env`：config 里写的是 `<MARENA_MEMORY_URL>` 占位符。
- **两个 server 都走 `harness.*_app`，不是上游的入口。** 区别只有观测：在 import
  上游 app 之前给 openai SDK 打标签、挂 request-id 中间件，让并行的几条 run 分得开。
  memory server 的 cwd 必须切到 `MemoryArena/memory`——上游用
  `from memory_systems import ...` 这种平级绝对导入，靠 uvicorn 把 cwd 放进 sys.path。

API key 通过 `tmux new-window -e` 传进去（走 argv，不进 shell history）：
`env_server.py` 自己不读 .env，但它要调 LLM 判分。
"""

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import run_math

from . import settings

SESSION = "memarena"

# 需要透传给 server 进程的凭证。只传当前环境里真有的。
PASSTHROUGH = [
    "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_BASE_URL",
    "OPENROUTER_API_KEY", "OPENROUTER_API_BASE_URL",
    "LETTA_API_KEY", "MIRIX_API_KEY", "MEM0_API_KEY",
    # memory 侧的补丁要用（落盘位置、memory 用的模型）
    "MARENA_RESULTS", "MARENA_MEMORY_STATE", "MARENA_REASONINGBANK_MODEL",
    "MARENA_MEM0_MODEL", "MARENA_MEM0_EMBEDDING", "MARENA_MEM0_HOSTED",
    "MARENA_SIMPLEMEM_MODEL", "MARENA_SIMPLEMEM_EMBEDDING",
    # env server 读本地数据；没有它就回落 HuggingFace，离线时 /env/initialize 直接 500
    "MEMORYARENA_DATA", "TRAVELPLANNER_DB",
    # SimpleMem 的 embedding 是本地 SentenceTransformers，要读 HF 缓存和挑卡
    "HF_HOME", "HF_HUB_CACHE", "HF_TOKEN", "HF_ENDPOINT", "CUDA_VISIBLE_DEVICES",
    # 网关自己要知道真上游；两个 server 不需要，它们只认 OPENAI_BASE_URL
    "MARENA_GATEWAY_UPSTREAM", "MARENA_GATEWAY_PORT",
    "MARENA_LLM_ROLE", "MARENA_LLM_JOURNAL_DIR",
]


def upstream_dir() -> Path:
    """上游仓库根目录。memoryarena 是 editable 装的，直接问包自己在哪。"""
    return Path(run_math.__file__).resolve().parent


def ports() -> tuple[int, int, int]:
    """memory, env, gateway。默认值挑在冷门段，见模块注释。"""
    return (int(os.getenv("MARENA_MEMORY_PORT", "18781")),
            int(os.getenv("MARENA_ENV_PORT", "18782")),
            int(os.getenv("MARENA_GATEWAY_PORT", "18780")))


def _alive(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/docs", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


def _gateway_alive(port: int) -> bool:
    """网关没有 /docs，用它自己的健康检查。"""
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/marena/health", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


def _tmux(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["tmux", *args], capture_output=True, text=True)


def _log_dir() -> Path:
    path = settings.results_dir() / "_servers"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _start_one(window: str, cwd: Path, app: str, port: int,
               env_override: dict[str, str] | None = None, module: bool = False) -> None:
    if _alive(port):
        print(f"[{window}] :{port} 已经在跑")
        return
    env_flags = []
    overrides = env_override or {}
    for name in PASSTHROUGH:
        value = overrides.get(name) or os.getenv(name)
        if value:
            env_flags += ["-e", f"{name}={value}"]
    _tmux("kill-window", "-t", f"{SESSION}:{window}")
    _tmux("new-window", "-d", "-t", SESSION, "-n", window, "-c", str(cwd), *env_flags)
    log = _log_dir() / f"{window}.log"
    if module:
        command = f"{sys.executable} -m {app} 2>&1 | tee '{log}'"
    else:
        command = (
            f"{sys.executable} -m uvicorn {app} --host 0.0.0.0 --port {port} 2>&1 | tee '{log}'"
        )
    _tmux("send-keys", "-t", f"{SESSION}:{window}", command, "C-m")
    print(f"[{window}] 启动中 :{port}   日志 {log}")


def gateway_role_url(role: str) -> str:
    """某个角色该用的 OPENAI_BASE_URL。"""
    return f"http://127.0.0.1:{ports()[2]}/{role}"


def start() -> int:
    memory_port, env_port, gateway_port = ports()
    if os.getenv("MARENA_GATEWAY_ROUTER", "passthrough") == "passthrough" \
            and not os.getenv("MARENA_GATEWAY_UPSTREAM"):
        print("MARENA_GATEWAY_UPSTREAM 没设置——网关不知道真上游在哪，见 .env",
              file=sys.stderr)
        return 1
    if _tmux("has-session", "-t", SESSION).returncode != 0:
        _tmux("new-session", "-d", "-s", SESSION, "-n", "shell")
    root = upstream_dir()

    # 网关必须先起：另外两个 server 的 OPENAI_BASE_URL 指着它
    _start_one("gateway", Path.cwd(), "harness.gateway", gateway_port, module=True)
    for _ in range(30):
        if _gateway_alive(gateway_port):
            break
        time.sleep(1)

    _start_one("memory", root / "memory", "harness.memory_app:app", memory_port,
               env_override={"OPENAI_BASE_URL": gateway_role_url("memory"),
                             "MARENA_LLM_ROLE": "memory"})
    _start_one("env", root, "harness.env_app:app", env_port,
               env_override={"OPENAI_BASE_URL": gateway_role_url("judge"),
                             "MARENA_LLM_ROLE": "judge"})
    print("等待就绪...")
    for _ in range(90):
        if _alive(memory_port) and _alive(env_port) and _gateway_alive(gateway_port):
            print("网关和两个 server 都起来了")
            return 0
        time.sleep(2)
    print(f"超时。看 {_log_dir()}/*.log", file=sys.stderr)
    return 1


def stop() -> int:
    if _tmux("kill-session", "-t", SESSION).returncode == 0:
        print("已停")
    else:
        print("没在跑")
    return 0


def status() -> int:
    memory_port, env_port, gateway_port = ports()
    print(f"{'gateway':<7} :{gateway_port} {'up' if _gateway_alive(gateway_port) else 'down'}")
    for name, port in (("memory", memory_port), ("env", env_port)):
        print(f"{name:<7} :{port} {'up' if _alive(port) else 'down'}")
    return 0


def logs() -> int:
    for path in sorted(_log_dir().glob("*.log")):
        print(f"===== {path} =====")
        print(path.read_text(encoding="utf-8", errors="replace")[-4000:])
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", nargs="?", default="start", choices=["start", "stop", "status", "logs"])
    raise SystemExit({"start": start, "stop": stop, "status": status, "logs": logs}[parser.parse_args().action]())
