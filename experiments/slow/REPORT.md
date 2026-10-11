# 慢脑（Slow Brain）实验报告

> 分支 `claude/exp-slow` · 2026-10 · 只改动 `duplexagent/plugins/slow/`、`experiments/slow/`、`configs/exp/slow/`、`bench_data/slow/`，没有改 `duplexagent/core/`。

## 0. 结论先行

- **推荐方案：`hybrid`（规则路由 + 带工具的 LLM agent）**。规则能完整解析的请求（算式、日期、单位，占评测集 1/3）毫秒级精确作答，其余交给 LLM agent。纯 CPU、Qwen3-4B-Instruct-2507（Q4）下 **64 题正确率 93.8%**（规则 22/22 全对，LLM 38/42），这 22 道可路由题从平均 17.7 s 降到约 1 ms。
- 同一模型下，**工具循环是正确率的主要来源**：无工具的单次调用基线（`openai_compat`）75.0%，且 8 题超时、56/64 的答案带 markdown 或换行、口语结论平均 207 字；工具 agent 92.2%，结论平均 17 字、无 markdown，98% 带详细版。
- **对话友好性指标**：接到任务 0 ms 内回报；81% 的任务（hybrid 84%）有“正在用计算器算 …”“写程序计算，得到 168”这类真实进度；运行中追加的补充采纳 7/8（无工具基线只有 1/8，其余 7/8 仍给旧答案）；取消响应约 1 ms，沙箱子进程零残留。
- 云端适配器 `claude`（Messages API）和 `claude_agent`（Claude Agent SDK + 进程内 MCP 工具）已写好，并用**脚本化 mock 服务**跑通了真实 SDK / 真实 Claude Code CLI 的工具调用、补充、取消和端到端场景；**没有 API Key，正确率、耗时和成本待补测**。
- 小模型风险：Qwen2.5-1.5B 基本不按协议调工具（51.6%），LLM 侧至少要 4B 级、做过原生工具调用训练的模型；CPU 上每个 LLM 任务 15–30 s，生产应上 GPU 或云端模型。
- 没有改 `duplexagent/core/`；第 7 节有 5 条向后兼容的接口变更请求，最重要的是“已结束任务的补充”和“把 partial/详细版带给快脑”。

## 1. 目标与设计

慢脑要“能干活”，更要“对对话友好”。按任务要求，所有候选共用一套对话友好的执行循环（`duplexagent/plugins/slow/agent.py`），后端可换：

| 要求 | 做法 |
|---|---|
| 持续回报有意义的进度 | 接到任务立刻回报 `已开始处理，正在分析问题`（不复述问题，免得快脑转述时重复）；每次调用工具前回报“正在用计算器算 23*47”，工具返回后回报“…，得到 1081”。快脑被问“好了吗”时转述的是**真实进展**，而不是“完成了第 1/2 步”。Claude 后端还会把 `thinking.display="updates"` 的进度笔记转成 `思路：…` |
| 流式中间结论（ConvFill 思路） | 每个工具结果同时发一条 `partial`（如 `用计算器算 23*47：23*47 = 1081`），融合层/快脑可以边等边用 |
| 尊重取消 | 循环里所有 await 都可取消；模型请求被取消时断开 HTTP；沙箱子进程在 `finally` 里按进程组 SIGKILL；`stream=true` 时断开流还能让本地推理服务立即停止生成 |
| 吸收 `task.amendments` | 模型思考期间每 50 ms 检查一次补充；新补充到达时**抢占**正在进行的模型调用（旧问题的答案没有意义），把补充作为 `用户补充：…` 注入后重来，并回报“收到补充「…」，按新条件重新考虑”；写最终答案时又来了补充，也会再跑一轮 |
| 结论精炼可口播 + 详细版 | 提示模型按 `结论：…（60 字内、口语、无 markdown）` + `详细：…` 作答；`result` 只放口语结论（去 markdown、下标数字转普通数字、去 `<think>`），详细版通过 `partial`（前缀 `详细：`）发出，并挂在 `task.detail` / `brain.details[task_id]` |
| 预算 | `max_steps` 轮工具后或超过 `time_budget_s`，要求模型“不再调工具，按已有信息作答” |
| 不可反问 | 慢脑看不到对话、无法追问；提示里要求“信息不全时按常见情况假设并在详细里说明” |

### 1.1 工具与沙箱（`toolbox.py`、`sandbox.py`）

| 工具 | 说明 |
|---|---|
| `calculator` | AST 白名单求值（四则、乘方、`sqrt/log/sin/factorial/comb…`、`π/e`、百分号、千分位），限制指数/阶乘规模，防 DoS |
| `python` | **隔离子进程**执行模型写的代码，见下 |
| `now` / `date_calc` | 当前时间（可配时区、可固定时间用于评测）、日期差、加减天数、星期/第几天/闰年 |
| `unit_convert` | 长度、重量（含斤/两）、体积、面积（含亩）、速度、数据、时间、温度 |
| `web_search` / `web_fetch`（`web=true` 才开） | 维基百科搜索 API（免 key）+ 网页正文抓取；只允许解析到公网地址的 http(s)（防 SSRF），正文截断 4000 字 |
| MCP 工具 | `mcp_servers=[{name, command, args}]` 接入任意 stdio MCP 服务器，工具名 `<server>__<tool>` |

Python 沙箱是多层防护，**不在引擎进程里执行任何模型代码**：
1. `python -I -S` 隔离模式，空环境变量，临时工作目录，代码从 stdin 传入；
2. rlimit：CPU 秒数、内存 512 MB、单文件 1 MB、64 个 fd、无 core；
3. 以 root 运行时，`unshare -n` 建立**无网络的网络命名空间**，再用 `setpriv` 降权到 `nobody`；
4. 子进程内 PEP 578 audit hook：禁止 socket、subprocess、exec/fork、ctypes、任何写文件、读 Python 安装目录以外的文件，以及导入 `socket/subprocess/ctypes/shutil/asyncio/…`；
5. 墙钟超时，超时或任务取消时杀整个进程组。

测试覆盖了：`import socket`、`os.system`、读 `/etc/passwd`、写文件、`__import__('subprocess')`、`().__class__.__base__.__subclasses__()` 找 Popen、死循环、2 GB 内存分配，全部被拦截；进程 uid 为 65534。说明：这是“防误伤 + 提高逃逸成本”的进程级沙箱，多租户/对抗场景应换 gVisor / nsjail / Firecracker 容器（见风险）。

### 1.2 候选实现

| impl | 文件 | 说明 |
|---|---|---|
| `mock` | 骨架自带 | 基线：等 3 秒，只会算纯算式 |
| `openai_compat` | 骨架自带 | 基线：一次 LLM 调用、无工具 |
| `router` | `router.py` | **无 LLM**：规则解析（中文口语运算符、中文数字、日期、单位）后直接调工具，毫秒级；解析不完整就拒答（不半懂乱算） |
| `agent` | `agent.py` | 带工具循环的 agent，任意 OpenAI 兼容端点（llama.cpp / llama-cpp-python / vLLM / Ollama / DeepSeek / 通义…）；`tool_mode="native"`（OpenAI tools）或 `"prompt"`（在系统提示里描述工具，解析 `<tool_call>{…}</tool_call>`，适配不支持 tools 的服务）；`stream=true` 可选 |
| `hybrid` | `router.py` | `router` 先接，规则搞不定的转给内层 LLM agent（`[slow.inner]` 任意 slow 配置） |
| `claude` | `claude.py` | 同一循环，后端换成官方 `anthropic` SDK 的 Messages API：默认 `claude-opus-5-5`、`effort=medium`、`fallbacks="default"`（服务端拒答回退）、`thinking.display="updates"`（进度笔记，账户不支持时自动降级一次）、可选服务端 `web_search/web_fetch`；完整保留 assistant content（含 thinking 块）回传 |
| `claude_agent` | `claude.py` | Claude Agent SDK（Claude Code harness）：我们的工具以**进程内 MCP 服务器**提供，内置 Bash/Read/Write/Edit 全部关闭（只能用沙箱工具，可选 WebSearch/WebFetch）；补充到达时 `interrupt()` 后追加用户消息；取消时 `disconnect()` |
| MCP 桥 | `mcp_tools.py` | 双向：作为客户端把外部 MCP 工具接给任一 agent；`python -m duplexagent.plugins.slow.mcp_tools` 把我们的沙箱工具箱作为 MCP 服务器提供给 Claude Code / Claude Desktop 等 |

每个候选一个配置：`configs/exp/slow/*.toml`（基于 `configs/default.toml`，只改 `[slow]`，LLM 候选把 `drain_timeout` 调到 120 s）。

## 2. 评测

### 2.1 数据（`bench_data/slow/`）

- `tasks.jsonl`：64 条，全部可自动判分。计算 11、单位换算 7、日期 9（“今天”固定为 2026-10-11）、事实 11、多步应用题 10、编程/枚举 10、规划 6（关键词 rubric，按“口语结论 + 详细方案”判分）。答案均用 Python 生成核对。字段兼容骨架的 `bench slow`（`answer_contains`），另加 `accept_any`（同义答案）、`reject_any`（如“不是闰年”）、`rubric`。判分前把“700万6652”这类口语数字展开成阿拉伯数字。
- `amendments.jsonl`：8 条，任务开始 1 秒后追加补充（“把 3 个人改成 4 个人”“不是八折是七折”“范围改成 1 到 50”…），判断最终答案是否体现补充（`adopted`）以及是否还是旧答案（`stale`）。
- `cancel.jsonl`：6 条，在 0.2–3 秒时取消，测取消响应时间与是否残留沙箱进程。

### 2.2 指标（`experiments/slow/bench.py`）

正确率（总体/分类）、端到端耗时 p50/p90、首次进度时间、每任务进度条数、**有信息量的进度占比**（排除“开始处理/完成了第 n 步”这类空话）、**最长静默**（两次更新之间最长间隔，决定快脑会不会“无话可说”）、工具调用数与**工具成功率**、补充采纳率、取消延迟、残留进程数、口语结论长度、详细版比例。

```bash
python experiments/slow/bench.py -c configs/exp/slow/hybrid_qwen3_4b.toml --name hybrid_qwen3_4b -v
python experiments/slow/leaderboard.py          # 汇总 results/*.json 成下表
python -m pytest -q experiments/slow/tests      # 慢脑专项测试（工具、沙箱、router、三种后端循环、取消、MCP）
```

### 2.3 硬件与模型

4 核 CPU（无 GPU）、15 GB 内存、Python 3.13。本地模型用 `llama-cpp-python[server]==0.3.36`（源码编译，`-DGGML_NATIVE=ON`）提供 OpenAI 兼容服务，Q4_K_M 量化：

```bash
pip install "llama-cpp-python[server]"   # CMAKE_ARGS="-DGGML_NATIVE=ON"
huggingface-cli download unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q4_K_M.gguf --local-dir models
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models
python -m llama_cpp.server --model models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf --n_ctx 8192 --n_threads 4 --port 8082 --model_alias qwen3-4b
python -m llama_cpp.server --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf --n_ctx 8192 --n_threads 4 --port 8081 --model_alias qwen1.5b
```

这台机器上 Qwen3-4B Q4 约 5.6 token/s 生成、约 30 token/s 预填（1.5B 约 10.6 token/s 生成），所以 LLM 候选的绝对耗时只代表“纯 CPU 小模型”的下限；正确率、进度、补充、取消这些行为指标与硬件无关。所有 LLM 评测串行执行（并发 1），避免互相抢 CPU。

## 3. 排行榜

主表（`python experiments/slow/leaderboard.py`，64 题 + 8 条补充 + 6 条取消；LLM 候选串行、4 核 CPU）：

| 候选 | 正确率 | 计算/单位/日期 | 事实 | 推理/编程 | 规划 | p50 耗时 s | p90 耗时 s | 首次进度 s | 进度/任务 | 有信息量进度占比 | 最长静默 p50 s | 工具调用/任务 | 工具成功率 | 补充采纳 | 取消延迟 p50 ms | 残留进程 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_qwen3_4b | 93.8% | 100% | 100% | 85% | 83% | 17.6 | 29.2 | 0.00 | 2.5 | 60% | 11.5 | 0.91 | 95% | 50% (80% 运行中) | 1.0 | 0 |
| hybrid_qwen3_4b_r1 | 93.8% | 100% | 100% | 80% | 100% | 15.4 | 29.9 | 0.00 | 2.5 | 60% | 12.1 | 0.92 | 92% | 50% (80% 运行中) | 1.0 | 0 |
| agent_qwen3_4b | 92.2% | 100% | 100% | 80% | 83% | 19.8 | 26.8 | 0.00 | 2.7 | 63% | 14.1 | 0.86 | 96% | 88% (88% 运行中) | 1.0 | 0 |
| agent_qwen3_4b_promptv1 | 85.9% | 90% | 100% | 90% | 33% | 16.2 | 23.9 | 0.00 | 2.7 | 63% | 10.8 | 0.84 | 94% | 100% (100% 运行中) | 1.0 | 0 |
| openai_compat_qwen3_4b | 75.0% | 70% | 91% | 85% | 33% | 27.7 | 120.1 | 0.00 | 1 | 0% | 27.7 | 0.00 | — | 12% (12% 运行中) | 1.0 | 0 |
| agent_qwen2_5_1_5b_prompt | 51.6% | 54% | 91% | 30% | 50% | 6.0 | 13.6 | 0.00 | 1 | 0% | 6.0 | 0.00 | — | 50% (50% 运行中) | 1.0 | 0 |
| router | 34.4% | 77% | 9% | 0% | 0% | 0.0 | 0.0 | 0.00 | 1.3 | 26% | 0.0 | 0.34 | 100% | 0% (— 运行中) | — | 0 |
| mock | 4.7% | 11% | 0% | 0% | 0% | 3.0 | 3.0 | 1.50 | 2 | 0% | 1.5 | 0.00 | — | 0% (0% 运行中) | 0.0 | 0 |
| agent_qwen3_4b_stream | — | — | — | — | — | — | — | — | — | — | — | — | — | 88% (88% 运行中) | 1.0 | 0 |

说明：
- “补充采纳”是最终答案体现补充条件的比例；括号内只统计补充到达时任务仍在运行的样本。`hybrid` 有 3 条补充落在已完成的规则任务上（规则毫秒级答完，补充到达时任务已结束），这是编排层问题，见接口变更请求 1；`router` 同理。
- `agent_qwen3_4b_stream` 只跑了补充和取消两部分，用来和非流式对比：补充到达后到出答案的平均耗时 **28.4 s → 22.6 s**。流式断开会让推理服务立刻停止生成旧答案；非流式请求会一直占着本地服务，直到生成完。
- `hybrid_qwen3_4b_r1` 是加入“截断 tool_call 防护 / 一下 修正 / km/s 单位”之前的同配置结果，与最终版持平（93.8%），说明结果稳定。
- `openai_compat` 的 8 个错误都是 120 s 超时：无工具模型对规划和枚举题会写长篇大论，在 5.6 token/s 下必然超时；超时的请求还会继续占用本地服务，拖慢后续任务。
- “有信息量进度占比”的分母包含“已开始处理”这类确认语。“最长静默”在 CPU 小模型上 p50 约 12 s，主要是单次推理耗时，换 GPU 或云端模型后会同比缩短。

### 3.1 消融

| 改动 | 前 | 后 | 说明 |
|---|---|---|---|
| 系统提示 v1 → v2（相对日期先调 `now`；不许反问，信息不全就假设；规划类在结论里给要点；数列先打印前几项核对下标） | 85.9% | 92.2% | 日期 78%→100%，规划 33%→83%（v1 有两题直接反问“请补充偏好”） |
| 规则路由：宽松 → 严格（表达式要覆盖整句） | 宽松版有 5 道错题，如把“先打八折再减 50”算成 −50 | 22/22 全对、0 错题 | 混合方案里错答比拒答危险得多 |
| 流式（`stream=true`） | 补充后平均 28.4 s | 22.6 s | 取消或抢占时服务端也停止生成 |
| KV 预热（`prewarm=true`，启动时把系统提示送进服务的前缀缓存） | 冷启动首个工具调用 24.8 s，总 42.6 s | 6.6 s，总 22.1 s | 只影响会话里第一个任务；评测里后续任务本来就命中前缀缓存 |
| `claude_agent` 预连一个空闲会话 + 后台断开 | 取消 0.8–3.3 s（取消落在 CLI 启动期间） | 约 0 ms，零残留 CLI 进程 | 同时省掉每个任务 1–3 s 的 CLI 启动 |

### 3.2 云端适配器的接口验证（scripted mock，不代表模型能力）

`experiments/slow/mock_llm.py` 同时实现 OpenAI Chat Completions（含流式）和 Anthropic Messages（含 SSE 流式、thinking 块、beta 拒绝模式）。`claude` 用真实的 `anthropic` SDK，`claude_agent` 用真实的 `claude-agent-sdk` 和它自带的 Claude Code CLI，只是把 `ANTHROPIC_BASE_URL` 指到 mock（结果在 `results/mock_interface/`）。mock 只会算算式，所以这里只看机制：

| 适配器 | 工具调用 | 补充被确认并注入 | 取消延迟 p50 / max | 残留进程 | 端到端场景 |
|---|---|---|---|---|---|
| `agent`（stream） | ✅ | 4/4 | 1 ms / 1 ms | 0 | 5/5（`agent_qwen3_4b` 真模型） |
| `claude` | ✅（thinking 块原样回传；同一轮的 tool_result 合并成一条 user 消息；beta 被拒时自动降级） | 4/4 | 1 ms / 1 ms | 0 | 4/5¹ |
| `claude_agent` | ✅（进程内 MCP 工具，内置 Bash/Read/Write 关闭） | 4/4 | 0 ms / 0.2 ms | 0 | 5/5 |

¹ `cancel` 场景假设任务 4 秒后还在运行；mock 1 秒就答完了，用户说“算了不用查了”时已经没有运行中的任务，规则快脑又因为句子里有“查”而重新委派（这是快脑的问题，不在慢脑范围）。`router` 失败的原因相同。

**待补测（需要 API Key）**：`claude`、`claude_agent` 在 64 题上的正确率、耗时、成本（`brain.usage` / `brain.cost_usd` 已记录）；`thinking.display="updates"` 进度笔记的实际质量；`fallbacks="default"` 的行为。补测命令：

```bash
export ANTHROPIC_API_KEY=...
python experiments/slow/bench.py -c configs/exp/slow/claude.toml --name claude_opus_5_5 -v --concurrency 4
python experiments/slow/bench.py -c configs/exp/slow/claude_agent.toml --name claude_agent_opus_5_5 -v --concurrency 4
python experiments/slow/bench.py -c configs/exp/slow/hybrid_claude.toml --name hybrid_claude -v --concurrency 4
```

## 4. 推荐方案

**默认用 `hybrid`**，内层按部署环境选：

| 场景 | 配置 | 理由 |
|---|---|---|
| 有云端 Key | `configs/exp/slow/hybrid_claude.toml`（内层 `claude`，`claude-opus-5-5`，`effort="medium"`；语音场景可试 `"low"`） | 规则层给出毫秒级的精确算术、日期、单位；Claude 负责推理、规划和多步工具；`fallbacks` 和进度笔记已接好 |
| 私有化 / 有 GPU | `hybrid` + `agent`（vLLM 或 llama.cpp `--jinja` 跑 Qwen3 系列，原生 tools，`stream=true`，`prewarm=true`） | 同一套循环；流式让取消和补充抢占立刻释放推理资源 |
| 纯 CPU / 离线（本次实测） | `configs/exp/slow/hybrid_qwen3_4b.toml` | 93.8%，可路由题毫秒级，LLM 题 15–30 s |
| 需要 Claude Code 生态（MCP、子代理、WebSearch） | `claude_agent` | 工具仍然只有沙箱工具和显式开启的 web；预连会话消除了 CLI 启动延迟 |

理由：
1. 正确率：工具循环带来 +17 个百分点（75.0% → 92.2%），规则前置再 +1.6 个百分点，同时把 1/3 的请求从十几秒降到毫秒。
2. 对话友好：只有工具循环能产出“正在算什么、算出了什么”的真实进度；补充抢占保证旧答案不会被说出口（stale 0%）；口语结论短、无 markdown，详细版单独提供。
3. 安全：模型写的代码只在降权、断网、受限的子进程里运行；规则层宁可拒答也不半懂乱算。
4. 可替换：后端（OpenAI 兼容 / Claude / Agent SDK）和工具来源（内置 / MCP）都是配置项，快脑和编排层无感。

## 5. 端到端验证

`python -m duplexagent eval scenarios/*.toml -c configs/exp/slow/<candidate>.toml`（结果在 `experiments/slow/results/e2e/`）：

| 候选 | basic | backchannel | barge_in | delegate | cancel |
|---|---|---|---|---|---|
| `hybrid_qwen3_4b`（推荐，真模型） | ✅ | ✅ | ✅ | ✅ | ✅ |
| `agent_qwen3_4b`（真模型） | ✅ | ✅ | ✅ | ✅ | ✅ |
| `hybrid`（内层 mock，`--set slow.impl=hybrid`） | ✅ | ✅ | ✅ | ✅ | ✅ |
| `claude_agent`（mock 服务） | ✅ | ✅ | ✅ | ✅ | ✅ |
| `claude`（mock 服务） | ✅ | ✅ | ✅ | ✅ | ❌¹ |
| `router` | ✅ | ✅ | ✅ | ✅ | ❌¹ |

另加了一个慢脑专用场景 `experiments/slow/scenarios/progress_status.toml`：用户委派“1 到 1000 之间有多少个质数”（规则做不了，转给 LLM），14 秒时问“算好了吗”，快脑转述进度，稍后把结果 168 融合回对话。`hybrid_qwen3_4b` 通过。

门槛：`python -m pytest -q`（10 个骨架测试）通过；`python -m pytest -q experiments/slow/tests`（26 个慢脑测试：工具、沙箱逃逸用例、规则、三种后端的工具循环/补充抢占/取消清理、真实 Agent SDK、MCP 往返、beta 降级）通过；默认配置下 `python -m duplexagent eval scenarios/*.toml` 仍是 5/5，没有退化。

## 6. 风险

1. **CPU 上太慢**：Qwen3-4B Q4 在 4 核上 5.6 token/s，LLM 任务 15–30 s，两次进度之间最长静默 p50 约 12 s。快脑要能撑场（已有“好了吗”→转述进度的路径）；生产需要 GPU（vLLM）或云端模型。本地服务优先选断连时会停止生成的实现，并开 `stream=true`、`prewarm=true`。
2. **小模型推理错误**：失败大多是不调工具的心算（相遇问题答 35 分钟、年龄题答 54 岁），或者“详细”里已经发现算错、“结论”却没改。缓解：换更大的模型，或加“结论与工具结果一致性”自检（下一步可做）。1.5B 级模型基本不会调工具，不建议作为内层。
3. **沙箱是进程级的**：降权和网络命名空间是内核级边界，但文件读取限制主要靠 audit hook 和 nobody 的权限位，而 audit hook 不是安全边界（CPython 文档明确说明）。多租户或对抗输入应换 gVisor / nsjail / Firecracker / 云端代码执行。没有 `unshare` 或非 root 运行时，会退化成只有 rlimit + audit hook。
4. **工具调用解析的服务端差异**：llama-cpp-python 0.3.36 的原生 tool call 偶尔丢掉代码参数里的反斜杠（`True\n` 变成 `Truenn`，有一题因此连错 3 次，模型最后靠心算答对），也会把 `<tool_call>` 写进正文。已加文本兜底解析和截断防护；生产建议用 vLLM（`--enable-auto-tool-choice`）或 llama.cpp server `--jinja`。
5. **云端适配器未实测**：只用 mock 验证了协议和机制。beta 特性（`fallbacks`、`thinking.display="updates"`）被拒时会自动降级一次；模型行为、成本和延迟要等有 Key 后补测。`claude_agent` 依赖 SDK 自带的 CLI，版本升级可能改变消息结构。
6. **联网工具弱**：`web_search` 目前用的是维基百科 API（免 Key，本环境已经遇到 429 限流）。实时信息（天气、股价、新闻）需要接带 Key 的搜索 API，或用 Claude 服务端 `web_search`（`server_web=true`）。评测集因此没有联网题。
7. **评测集偏“有标准答案”**：64 题以可自动判分为主，规划类只用关键词 rubric；开放式任务的质量需要 LLM 评审或人工打分。
8. **已结束任务上的补充会丢**：规则层毫秒级答完后，补充到达时任务已结束，编排层直接忽略（见接口变更请求 1）。

## 7. 接口变更请求

没有改 `duplexagent/core/`。以下请求都是向后兼容的增量，按优先级排列：

1. **补充到达时任务已结束**：`TaskOrchestrator.amend()` 现在会静默忽略非运行中的任务。建议返回 `bool`（或记一条 `task_amend_ignored` 事件），由快脑决定是否用“原问题 + 补充”重新委派；或者在结果尚未融合（`delivered=False`）时，由编排层自动用 `query + amendments` 重开任务并作废旧结果。`hybrid` 的 3 条补充就是这样丢的。
2. **让快脑看到中间结论**：编排层把 `partial` 收在局部变量里，只在没有 `result` 时拼起来用。建议 `TaskRecord` 增加 `partials: list[str]`（ConvFill 式“边等边用”需要）和 `detail: str | None`。目前详细版用前缀为 `详细：` 的 `partial` 发出，并以动态属性 `task.detail` 挂上（dataclass 允许，但不是正式字段）。
3. **`TaskUpdate.meta: dict`**（可选字段）：放工具名、第几步/预计几步、预计剩余时间等结构化信息，让快脑用自己的口吻播报进度，而不是转述慢脑写好的句子。
4. **主动预取钩子**（arXiv 2603.02206 思路）：`SlowBrain` 增加可选的 `async def observe(history: list[Message]) -> None`（默认空实现），由引擎在每轮对话后调用，慢脑就能预测接下来的话题并预热检索或计算。现在慢脑只看得到 `Delegate.query/context`，做不了预取。
5. **任务截止时间**：`Delegate` / `TaskRecord` 增加可选的 `deadline_s`，慢脑据此调整 `time_budget_s`（例如用户说“快点告诉我个大概就行”）。目前预算只能在配置里全局设置。
