"""EnvironmentClient 的同进程版：同样的方法、同样的返回形状，不走 HTTP。

`env_server.py` 对 step 做的事只有一件：把环境返回的 3 元组或 4 元组摊成
`{"observation", "reward", "done", "info"}`。这里照抄那段，所以 `run_travel.run_group`
拿它和拿 `EnvironmentClient` 跑出来的流程一样。用在一个进程里并行跑多组（每组一个环境对象），
不用起 env server。
"""

from typing import Any, Dict, Optional


class LocalEnvironmentClient:
    def __init__(self, env: Any):
        self.env = env

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        return self.env.reset(seed=seed)

    def step(self, action: Any, ground_truth: Any = None, need_judge: bool = False, **kwargs: Any) -> Dict[str, Any]:
        result = self.env.step(action, ground_truth=ground_truth, need_judge=need_judge, **kwargs)
        if not isinstance(result, tuple) or len(result) not in (3, 4):
            raise RuntimeError("Environment step() must return a 3- or 4-tuple")
        if len(result) == 4:
            observation, reward, done, info = result
        else:
            observation, reward, info = result
            done = getattr(self.env, "_done", False)
        return {"observation": observation, "reward": reward, "done": done, "info": info or {}}

    def get_observation(self) -> Dict[str, Any]:
        return self.env.get_observation()

    def close(self) -> Dict[str, Any]:
        self.env.close()
        return {"status": "ok"}
