"""跑实验的基础设施，跟 benchmark 本身无关。

这里的模块负责「怎么跑一条 run 并把它记下来」，不负责「benchmark 是什么」：

    settings     路径，全部来自 .env，代码里不拼路径
    config       读我们自己的 config
    runctx       让共享的 server 进程知道这次 LLM 调用替哪条 run 打的
    llm_journal  记录每一次 LLM 调用，不管打给哪家
    gateway      LLM 网关，所有进程的调用都从这里过
    serve        起/停 server
    trace        Recorder，把一条 run 的每一步记进 trajectory.json

上面七个只互相依赖，换一个 benchmark 照样能用。另外两个是给 uvicorn 用的 ASGI
入口，它们按定义就要 import 上游的 app，所以不在那条边界之内：

    env_app      env server 入口（装好 tagging 再 import env.env_server）
    memory_app   memory server 入口（同上，import memory 的 server）
"""
