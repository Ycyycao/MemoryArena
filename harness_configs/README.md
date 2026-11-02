# Configs

从 `configs/<env>_configs/` 拷贝后修改，按环境分子目录。上游的
`configs/` 是 tracked 文件，直接改会和 pull 冲突。

改动约定：

- `output.json_output_dir` 用绝对路径，指向 `results/<run-id>/`，run-id 和 config
  文件名一致；`marena-case --run-id` 会覆盖它。
- server 地址用 `<MARENA_MEMORY_URL>` / `<MARENA_ENV_URL>` 占位符，不要写死端口——
  端口的唯一来源是 `.env`。
- `<OPENAI_BASE_URL>` 这类尖括号占位符不用手填，`harness.config.load()` 会自动换成
  同名环境变量的值。
- `agent.agent_mode` 可以不写；写了只能是 `toolcall`（默认，上游原样的
  function-calling）或 `react`。也可以不进 config，跑的时候给
  `marena-travel-case --agent-mode react`，命令行优先，最终值会落进
  `trajectory.json` 的 `config` 里。
- 文件名用 kebab-case，能看出 env + memory system + model。

现有两份：

- `formal_reasoning/math-longcontext-gpt5mini.json` —— 给 `marena-case`
- `travel_planner/travel-longcontext-gpt5mini.json` —— 给 `marena-travel-case`

