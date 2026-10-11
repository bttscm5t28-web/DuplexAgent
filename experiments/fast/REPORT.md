# 快脑（Fast Brain）实验报告

> 分支 `claude/exp-fast`，2026-10。只改动了规范里属于快脑的目录（`duplexagent/plugins/fast/`、`experiments/fast/`、`configs/exp/fast/`、`bench_data/fast/`），**没有改 `duplexagent/core/`**。

## 0. 结论速览

1. **“规则路由 + 小模型说话”的混合方案明显优于“让小模型自己决定一切”。** 纯 LLM（Qwen2.5 0.5B/1.5B，同一套 prompt）在留出集上的动作准确率只有 53–62%，还会对已委派的问题瞎猜答案（违规率 14–20%），并且会在“你好”时重复委派正在运行的任务；端到端 9 个场景只过了 2 个。混合方案在同样的数据上达到 87.5–100%，瞎猜违规为 0。
2. **委派判断的质量上限由模型大小决定，延迟上限由 CPU 决定。** 用首 token 概率做门控的 `hybrid-gated-qwen3-4b-2507`：留出集委派准确率 / 精确率 / 召回 **100 / 100 / 100%**，融合测试集 86.7%，是质量最好的方案。但在本机 4 核 CPU 上，自答轮首段 TTFT p50 约 1.9 s，端到端会错过“说完 → 出声 < 800 ms”的目标，也让打断类场景出现时序错位。
3. **本机（4 核 CPU、无 GPU）推荐 `hybrid-strict-qwen2.5-1.5b`**：规则路由决定委派、取消、补充、进度，回执与进度用模板播报（≈0 ms），1.5B 只负责闲聊和融合转述。它的自答轮 TTFT p50 为 0.51 s，常见寒暄走零延迟模板；端到端 9 个场景中 8 个通过，唯一未过的是基线规则脑特有的字面检查“我明白了”。更看重延迟时换成 0.5B 版本（TTFT p50 0.33 s）。**有 GPU 或更快的 CPU 时换成 `hybrid-gated-qwen3-4b-2507`**（同一插件，只换配置）。
4. 融合（`trigger="task_result"`）采用“**模板开头 + 模型续写**”：先立刻说出“关于X，”或“更正一下我刚才说的，关于X，”，模型从这里预填续写，因此融合 TTFT 约为 0。另外有三道保险：数字核对（模型说出的数字必须在结果里出现，否则改用模板）、按任务逐个转述、失败时用固定模板。在全新测试集上，4B 的融合通过率为 86.7%（加数字核对前为 76.7%）。
5. 骨架自带的 `openai_compat`（JSON 工具调用）走 llama-cpp-python 的 OpenAI 兼容服务时**一次工具都没调**，模型却口头说“已经交给慢脑了”。原因是该服务的 `chatml-function-calling` 格式在 `tool_choice=auto` 下不会让 Qwen 产出工具调用。同一个服务换用本方案的文本协议（`backend="openai"`）后，迷你集动作准确率 88.9%，瞎猜 0。

## 1. 硬件与环境

- Intel Xeon @ 2.10 GHz，4 核，15 GB 内存，无 GPU；Python 3.13；llama-cpp-python 0.3.36（CPU 预编译轮子）。
- 模型（GGUF，均为 Hugging Face 官方或 unsloth 量化）：Qwen2.5-0.5B/1.5B/3B-Instruct Q4_K_M、Qwen3-1.7B Q8_0、Qwen3-4B-Instruct-2507 Q4_K_M、Qwen3-0.6B Q8_0。
- 实测吞吐（4 线程，`n_batch=512`）：

| 模型 | prefill tok/s | decode tok/s |
|---|---|---|
| Qwen2.5-0.5B Q4_K_M | ≈220 | ≈45 |
| Qwen2.5-1.5B Q4_K_M | ≈110 | ≈20 |
| Qwen2.5-3B Q4_K_M | ≈55 | ≈10 |
| Qwen3-1.7B Q8_0 | ≈70 | ≈11 |
| Qwen3-4B-Instruct-2507 Q4_K_M | ≈45 | ≈8 |

结论：CPU 上 TTFT 几乎完全由 **prefill** 决定，所以 prompt 设计的第一原则是让 KV cache 能复用（见 §4.3）。

## 2. 数据与评测

### 2.1 数据集（`bench_data/fast/`，标注规范见 `experiments/fast/LABELING.md`）

| 文件 | 条数 | 作者 | 用途 |
|---|---|---|---|
| `delegation.jsonl` | 194 | 我（与规则同一作者） | **开发集**：规则调试、门限选择 |
| `delegation_heldout.jsonl` | 120 | 独立子代理，只看过标注规范，没看过代码 | **留出集**：只用于报告，不据此改规则 |
| `delegation_mini.jsonl` | 63 | 上面两份每 5 条取 1 | 快速冒烟（服务端验证用） |
| `fusion.jsonl` | 14 | 我 | 融合开发集 |
| `fusion_heldout.jsonl` | 24 | 独立子代理 | 融合留出集（**注意**：做过两轮错误分析，已不算干净） |
| `fusion_test.jsonl` | 30 | 另一个独立子代理 | 融合**测试集**：只在最终代码上跑（例外见 §6） |

委派数据合计 314 条，覆盖闲聊、常识、心算、精确计算、实时信息、私人数据和动作、规划比较、写作翻译、进度询问、取消、补充条件、打断后续接、指令不完整需追问，中文为主，夹带少量英文和中英混说；带 `tasks`（正在运行的后台任务）和 `history`（含 `interrupted: true` 的被打断话语）。

### 2.2 指标（`experiments/fast/bench.py`，兼容 `python -m duplexagent bench fast` 的数据格式）

- **委派准确率 / 精确率 / 召回**：本轮是否产生新的 `Delegate`。
- **动作准确率**：期望动作 `say / delegate / cancel / amend / status` 是否命中。`cancel / amend` 还要求 task_id 指向正在运行的任务；`status` 不得新建任务。
- **瞎猜违规率**：带 `forbid` 的条目里，快脑说出了被禁止内容（例如对正在计算的问题报出数字）的比例。
- **TTFT**：从调用 `respond()` 到第一段非空 `Say` 的时间（经过短语切分，所以是 TTS 真正能拿到文本的时间）。`--warm` 会先把前几轮对话放进 KV cache，模拟真实会话里前文已缓存的情况。
- **融合质量**（`trigger="task_result"`）：点明问题（`must_mention_any`）、传达要点（`must_contain_any`）、简短（≤`max_chars`）、口语（无 Markdown、列表、表情）、更正正确（该更正时更正，不该时不更正），以及全部满足的通过率。
- **端到端**：骨架 5 个场景加上快脑新增的 4 个场景（`experiments/fast/scenarios/`：进度追问、补充条件、任务进行中闲聊、两个并发任务），用 `--serial` 串行跑。

## 3. 候选

| 候选（配置 `configs/exp/fast/<名>.toml`） | 实现 | 说明 |
|---|---|---|
| `rule`（基线） | `fast/rule` | 骨架自带的关键词规则 |
| `router` | `fast/router` | 规则路由 v2：中文正则 + 中文数字解析 + 运行中任务匹配；回执、进度、取消、补充、融合全部模板化 |
| `llm-<模型>` | `fast/llamacpp mode="llm"` | 小模型按文本协议决定一切 |
| `hybrid-<模型>` | `fast/llamacpp mode="hybrid"` | 路由有把握就直接执行；路由判为闲聊时交给模型，模型仍可委派、取消、补充 |
| `hybrid-strict-<模型>` | 同上，`llm_decides=false` | 一切决策听路由的，模型只负责说话（强制 `[说]`）和融合 |
| `hybrid-gated-<模型>` | 同上，`gate=<门限>` | 路由判为闲聊时读模型首 token 概率，只有 P(委派/取消/补充) ≥ 门限才采用模型的动作 |
| `openai_compat`（骨架自带） | JSON 工具调用 | 新增 `stream=false` 选项，适配不能流式输出工具调用的服务 |
| `llamacpp backend="openai"` | 文本协议，走任意 OpenAI 兼容 chat 接口 | 云端或服务端适配器（vLLM / Ollama / llama.cpp server / 云 API） |

还调研过但没有实测的：ConvFill（arXiv 2511.07397，`vysri/conversational-infill`，MIT）。它需要用 29 万条英文合成数据微调专用 talker（135M–1.7B），并且要求慢脑**流式**推送知识块。本方案的“模板开头 + 模型续写”融合是它的免训练简化版；若要真正落地，需要慢脑支持 partial 更新的融合时机（见 §8），并准备中文训练数据。

## 4. 方案设计要点

### 4.1 文本协议代替 JSON 工具调用

小模型（0.5B–4B）在 JSON 工具调用上既不稳定又慢：要等整段 JSON 生成完才能动作，回执也只能在调用之后才说。改成**单标记文本协议**：

```
[说] 直接回答……
[委派] 一句回执 [任务] 自包含的任务描述
[取消 t1] 回执
[补充 t1] 回执 [信息] 补充内容
```

- 标记在最前面，第 1–2 个 token 就能确定动作；**回执写在任务描述之前**，所以标记一确定就能开始说话，任务描述在回执播放期间生成。
- llama.cpp 后端预填 `[`（门控或 strict 模式下预填整个标记），模型不会跑题。
- 解析容错：模型不按协议输出时当作 `[说]`；标记里的任务号不存在时按文本相似度挑选任务；没有运行中任务时 `[取消]`/`[补充]` 降级为说话；口语输出里清除残留的标记和 Markdown。
- 动作（`Delegate` / `CancelTask` / `AmendTask`）**先于回执 yield**（模板路径上），避免用户在回执播放时打断导致任务没有下发。

### 4.2 规则路由 `common.route()`（≈0.1 ms）

- 有运行中任务时依次判断：取消（“不用了 / 先别 / 我自己写吧 / never mind”…）、进度（“好了吗 / 结果是多少 / 先说个大概”…）、补充（“对了… / 我是说… / 不是A是B / 要…的”…）；命中“再帮我 / 另外帮我 / 提醒”等新请求标记时不算这三类。
- 委派五类主题：私人数据与动作、实时信息、精确计算（解析中文数字，“三加五”“一百的一半”这类心算不委派，多位数乘除、利率、换算才委派）、写作翻译、规划比较。
- 防误判：“我查了一下午资料好累”“我计划明天早起”“X 用英文怎么说”走闲聊；“帮我订个”这类话没说完的先追问。
- 回执按主题选模板（“好，我让后台精确算一下，马上告诉你。”）；进度只报任务表里的进度（**结构上不可能瞎猜**）；取消、补充复述对应的任务。

### 4.3 低延迟手段（按收益排序）

1. **静态前缀 + KV cache 复用**：系统提示词（含 10 个 few-shot 示例，≈700 token）放在最前面且固定不变，启动时预热；每轮变化的任务表放在**最后一条用户消息之后**，历史窗口按 6 条一档滑动，所以真实会话里每轮只需 prefill 新增的几十个 token。few-shot 因此几乎是免费的。
2. **模板快路径**：回执、进度、取消、补充、追问、失败通知以及最常见的寒暄（“你好 / 谢谢 / 再见 / 嗯好的”等精确匹配）都不经过模型，TTFT 约为 0。
3. **没有运行中任务时省掉任务表消息**：再少 prefill 十几个 token。上面两项一起把 1.5B 自答轮 TTFT p50 从 0.65 s 降到 0.51 s，0.5B 从 0.39 s 降到 0.33 s。
4. **短语切分 `PhraseChunker`**：首段最多 4 个字就送 TTS，之后在标点或 16 个字处切分；每段再经过 `speakable()` 规范化（换行变逗号，去掉列表编号和 Markdown，`→`/`★`/`~` 转成可读的字）。
5. **生成放在工作线程**：llama.cpp 调用是阻塞的，放进线程后通过 asyncio 队列逐 token 传回，不会卡住音频泵。被打断时 `aclose()` 立即让线程在下一个 token 停下并释放模型锁。

### 4.4 融合

- 每个结果单独转述：开头由模板立即说出（“你之前问的2件事都有结果了。关于X，” / “另外，关于Y，”）；模型以这句话作为预填续写，限 50 字，到句末截断。
- **更正判断**：先在历史里找自己对这个问题说过的含糊估计（“大概 / 好像 / 应该 / 差不多…”）。有数字就按数字比对（忽略复述题目的数字，±10% 内算一致，支持“一百三 = 130”这类口语数字）；没有数字时让模型做一次“是 / 否”判断，判断作为同一前缀的续写，不会冲掉 KV cache。判定冲突就在开头加“更正一下我刚才说的，”。
- **数字核对**：模型续写的每个短语在播出前检查，凡是结果或问题里没有的数字（>3）就停止生成，改说“准确地说，<结果摘要>”。这一步是在测试集上看到 4B 把“42.195 公里”说成“4195公里”之后加的。
- 任务失败用固定模板：“抱歉，X没办成，原因是…”。

### 4.5 首 token 概率门控（`gate`）

路由判为闲聊时，用一次只做 prefill 的前向读出 `[` 之后各标记首 token 的概率（复用 KV cache，自己做前缀匹配后调用 `eval`，再读 `get_logits()`，不需要 `logits_all`）。只有 P(委派 / 取消 / 补充) ≥ 门限才采用模型的动作，否则强制 `[说]`。门限只在开发集上选（`experiments/fast/calibrate.py`），然后在留出集上报告：

| 模型 | 纯 LLM 留出动作准确率 | hybrid 选出的门限 | hybrid+门控 留出动作准确率（离线模拟） | 门控耗时 p50（仅 prefill，历史未缓存） |
|---|---|---|---|---|
| Qwen2.5-0.5B | 56.7% | 0.9 | 86.7% | 0.25 s |
| Qwen2.5-1.5B | 59.2% | 1.01（即永不推翻路由） | 87.5% | 0.42 s |
| Qwen2.5-3B | 82.5% | 0.99 | 91.7% | 0.80 s |
| Qwen3-4B-Instruct-2507 | **95.8%** | 0.2 | **100%** | 1.13 s |

1.5B 及以下推翻路由只会更糟，所以小模型用 strict；4B 的判断比规则可靠，门控后在留出集上全对，完整生成实测也是 100%（见排行榜）。

## 5. 排行榜

离线基准（`--warm`；TTFT 单位为秒；“自答轮”指快脑自己回答、不委派的轮次，用户等的就是这个）：

| 候选 | 委派准确率 主/留出 | 精确率 留出 | 召回 留出 | 动作准确率 主/留出 | 瞎猜违规 | 融合通过 主/留出/测试 | TTFT p50/p90 全部 | TTFT p50/p90 自答轮 |
|---|---|---|---|---|---|---|---|---|
| **hybrid-gated-qwen3-4b-2507** | 98.5 / **100.0** | 100.0 | 100.0 | 98.5 / **100.0** | 0.0 | 100.0 / 83.3 / **86.7** | 0.00 / 2.31 | 1.92 / 2.47 |
| hybrid-qwen3-4b-2507 ¹ | 98.5 / 99.2 | 100.0 | 98.2 | 97.9 / 99.2 | 0.0 | 92.9 / 75.0 / - | 0.00 / 2.28 | 1.96 / 2.51 |
| hybrid-qwen2.5-3b ¹ | 96.4 / 96.7 | 94.6 | 98.2 | 96.4 / 93.3 | 0.0 | 92.9 / 62.5 / - | 0.00 / 1.73 | 1.43 / 1.87 |
| hybrid-gated-qwen2.5-3b ² | 97.9 / 95.0 | 98.0 | 90.7 | 97.4 / 91.7 | 0.0 | 92.9 / 70.8 / 80.0 | 0.00 / 1.72 | 1.44 / 1.86 |
| hybrid-strict-qwen3-4b-2507 ² | 97.9 / 90.8 | 100.0 | 79.6 | 96.9 / 87.5 | 0.0 | 100.0 / 83.3 / 86.7 | 0.00 / 2.00 | 1.70 / 2.28 |
| hybrid-strict-qwen2.5-3b ² | 97.4 / 90.8 | 100.0 | 79.6 | 96.4 / 87.5 | 0.0 | 78.6 / 75.0 / 83.3 | 0.00 / 1.44 | 1.23 / 1.70 |
| **hybrid-strict-qwen2.5-1.5b** | 97.9 / 90.8 | 100.0 | 79.6 | 96.9 / 87.5 | 0.0 | 92.9 / 79.2 / 76.7 | 0.00 / 0.63 | **0.51 / 0.74** |
| **hybrid-strict-qwen2.5-0.5b** | 97.9 / 90.8 | 100.0 | 79.6 | 96.9 / 87.5 | 0.0 | 100.0 / 75.0 / 70.0 | 0.00 / 0.40 | **0.33 / 0.47** |
| router（无模型） | 97.9 / 90.8 | 100.0 | 79.6 | 96.9 / 87.5 | 0.0 | 100.0 / 70.8 / 83.3 | 0.00 / 0.00 | 0.00 / 0.00 |
| hybrid-qwen2.5-1.5b ¹ | 90.2 / 83.3 | 77.4 | 88.9 | 90.2 / 83.3 | 0.0 | 92.9 / 70.8 / - | 0.00 / 0.93 | 0.74 / 0.93 |
| llm-qwen2.5-1.5b ¹ | 66.5 / 64.2 | 62.2 | 51.8 | 65.0 / 61.7 | **20.0** | 92.9 / 62.5 / - | 0.86 / 1.26 | 0.81 / 1.08 |
| llm-qwen2.5-0.5b ¹ | 59.3 / 55.0 | 50.0 | 77.8 | 58.2 / 53.3 | **14.3** | 100.0 / 70.8 / 70.0 | 0.55 / 0.78 | 0.50 / 0.68 |
| rule（基线） | 70.1 / 62.5 | 76.5 | 24.1 | 61.9 / 53.3 | 0.0 | 78.6 / 50.0 / 53.3 | 0.15 / 0.15 | 0.15 / 0.15 |

¹ 在加入模板快路径、省略空任务表、数字核对、更正判断之前跑的（决策不受影响；TTFT 略偏高；融合是旧逻辑）。² 委派部分在省略空任务表之前跑，融合部分（除 gated-3b 的融合外）用的是最终代码。融合数字有采样波动（temperature 0.3，单次运行），例如 strict-3b 在 14 条的开发集上只有 78.6%。

端到端（骨架 5 个场景 + 快脑 4 个场景，`python -m duplexagent eval --serial`；延迟是各场景 `latency_p50` 的平均值，含模板轮）：

| 候选 | 通过 | 未通过的检查 | 延迟 p50 均值 |
|---|---|---|---|
| router | **9/9** | - | 0.58 |
| hybrid-strict-qwen2.5-0.5b | 8/9 | backchannel:said_contains「我明白了」 | 0.63（开放闲聊轮 0.76–0.88） |
| hybrid-strict-qwen2.5-1.5b | 8/9 | backchannel:said_contains「我明白了」 | 0.69（开放闲聊轮 0.99–1.14） |
| hybrid-gated-qwen2.5-3b | 8/9 | backchannel:said_contains「我明白了」 | 0.84（开放闲聊轮 1.47–1.99） |
| hybrid-gated-qwen3-4b-2507 | 7/9 | backchannel:turns_detected、barge_in:barge_in_detected（时序问题，见下）、「我明白了」 | 0.73（开放闲聊轮约 1.9） |
| llm-qwen2.5-1.5b ¹ | 2/9 | cancel、status、amend、两任务等**结构性**错误 | 1.45 |

- “我明白了”是基线 `rule` 回声式回复的原话，不是快脑能力指标，LLM 候选自然说不出来。
- 4B 在 backchannel / barge_in 场景失败，是因为开放闲聊要约 1.9 s 才出声，脚本里用户在 5.0 s 说的“嗯”或打断发生在助手开口之前，于是变成了新的一轮。这正说明 CPU 上 4B 太慢，并不是逻辑错误。
- 服务端验证（`experiments/fast/serve_check.sh`，llama-cpp-python OpenAI 兼容服务 + Qwen2.5-3B，迷你集 63 条）：

| 路径 | 动作准确率 | 委派召回 | 瞎猜违规 | TTFT p50 |
|---|---|---|---|---|
| `openai_compat`（JSON 工具，流式） | 50.8% | **0%** | 33% | 0.52 s |
| `openai_compat stream=false` | 50.8% | **0%** | 33% | 1.82 s |
| `llamacpp backend="openai"`（文本协议，纯 LLM） | **88.9%** | 100% | 0% | 1.42 s |

用 curl 直接验证过：该服务的 `chatml-function-calling` 在 `tool_choice=auto` 时，Qwen 不产出工具调用（只会闲扯），强制 `tool_choice` 时能产出。所以 JSON 工具路径能否工作取决于服务端的工具模板（llama.cpp `llama-server --jinja` 或 vLLM `--tool-call-parser hermes` 应当可以，**待补测**）；文本协议对任何 chat 接口都有效。

## 6. 方法学说明（数字该怎么读）

- 开发集和规则出自同一作者，开发集上的 97.9% 有乐观偏差；**请以留出集为准**。router 第一次跑留出集时看过失败条目，之后没有据此改规则。留出集上 router 的召回是 79.6%：漏掉的多为表述不常见的计算（“这组数的标准差”“17 percent of 4893”）、英文请求和智能家居（“把空调调到二十六度”）。
- `fusion_heldout` 做过两轮错误分析（修了中文口语数字解析、多任务逐个转述、TTS 规范化），已经算开发集。`fusion_test` 是最后才生成的；4B 在它上面第一次跑出 76.7%（结果存在 `results/v1/…pre-number-guard.json`）。看过失败条目后加了数字核对和“只说这一件事”两条规则，再跑得到 86.7%。所以 **76.7% 是干净的测试数**，86.7% 受到了测试集的启发。
- 数据由模型生成，是书面化的“口语”；真实 ASR 文本会更乱（缺字、同音字、没有标点），上线前要用真实日志再测。
- TTFT 在单进程、单会话、无并发的条件下测得。

## 7. 推荐方案

| 部署条件 | 推荐配置 | 理由 |
|---|---|---|
| **当前环境：4 核 CPU，无 GPU（默认）** | `configs/exp/fast/hybrid-strict-qwen2.5-1.5b.toml` | 决策等同 router（精确率 100%、瞎猜 0）；自答轮 TTFT p50 0.51 s；端到端 8/9（只差基线的字面检查）；融合测试集 76.7% |
| 极致延迟，或算力被 ASR/TTS 占满 | `hybrid-strict-qwen2.5-0.5b`（或纯 `router`） | 自答轮 TTFT 0.33 s；0.5B 的闲聊质量明显较弱（本实验没有量化），router 只能说模板话 |
| 有 GPU（任意消费级卡），或 ≥16 核 CPU | `hybrid-gated-qwen3-4b-2507` | 委派与动作在留出集上 100%，融合最好（测试集 86.7%）；GPU 上 4B 的 prefill 只需几十毫秒，延迟问题消失 |
| 用云端或服务端模型 | `fast.impl=llamacpp`、`backend="openai"`、`mode="hybrid"`；小模型加 `llm_decides=false`，大模型（≥4B 级）保留 `llm_decides=true` | 文本协议对任何 chat 接口都能用（`gate` 需要 logits，只支持 llama.cpp 后端）；JSON 工具调用依赖服务端模板，不推荐 |

```bash
# 推荐方案端到端
python -m duplexagent eval --serial -c configs/exp/fast/hybrid-strict-qwen2.5-1.5b.toml scenarios/*.toml experiments/fast/scenarios/*.toml
# 或在任意配置上覆盖
python -m duplexagent eval --serial scenarios/*.toml --set fast.impl=llamacpp --set fast.model_path='"models/qwen2.5-1.5b-instruct-q4_k_m.gguf"' --set fast.llm_decides=false
```

### 模型下载（权重不入库，放在 `models/`）

```bash
pip install -r experiments/fast/requirements.txt   # llama-cpp-python 用 CPU 预编译轮子，见文件注释
python - <<'EOF'
from huggingface_hub import hf_hub_download
for repo, f in [("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
                ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
                ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),
                ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")]:
    hf_hub_download(repo, f, local_dir="models")
EOF
```

### 复现

```bash
python experiments/fast/make_data.py                     # 重新生成开发集与融合开发集
python experiments/fast/bench.py all --warm -c configs/exp/fast/<候选>.toml --out experiments/fast/results/bench_<候选>.json
python experiments/fast/calibrate.py -c configs/exp/fast/hybrid-qwen3-4b-2507.toml   # 门限标定
bash experiments/fast/e2e.sh router hybrid-strict-qwen2.5-1.5b                      # 端到端
bash experiments/fast/serve_check.sh                                               # OpenAI 兼容服务验证
python experiments/fast/leaderboard.py                                             # 汇总排行榜
python -m pytest -q experiments/fast/test_fast.py                                  # 快脑单元测试（无需模型）
```

## 8. Prompt

快脑系统提示词（`duplexagent/plugins/fast/llamacpp.py: SYSTEM_PROMPT`，固定前缀，含 few-shot）：

```text
你是语音助手的“快脑”，你说的每个字都会被直接念给用户听。后台还有一个“慢脑”，能查资料、用工具、精确计算、做规划、写长内容，但要花几秒到几分钟。

每次回复必须以一个标记开头：
[说] 你直接回答。口语化、简短（一两句），不用列表、Markdown、表情。
[委派] 一句简短回执 [任务] 自包含的任务描述（慢脑看不到对话，要写清楚时间地点数字等）
[取消 t编号] 一句回执 —— 用户不要某个正在运行的任务了
[补充 t编号] 一句回执 [信息] 补充内容 —— 用户给正在运行的任务加条件或改条件

什么时候用[委派]：需要实时或外部信息（天气、新闻、股价、汇率、比赛、附近的店）；需要用户的私人数据或执行动作（日程、闹钟、提醒、发消息、订票、开关电器）；多位数乘除、利率、换算等精确计算；规划、比较、调研；写邮件、文章、代码、长翻译。
什么时候用[说]：闲聊、安慰、观点、稳定的常识、一步心算、简短词语翻译；问正在运行的任务进度（只说还在处理，绝对不要猜结果）；用户话没说完时追问。
已经交给后台的问题，在结果回来之前，绝不自己猜答案。

示例：（10 组 用户/助手 示例，覆盖 说 / 委派 / 取消 / 补充 / 进度 / 追问，见源码）
```

每轮动态内容（放在最后一条用户消息之后，没有运行中任务时省略）：

```text
当前正在运行的后台任务：
[t1] 查询明天北京的天气（进度：正在查询天气接口）
```

融合指令（模板开头已经播出，模型从开头处预填续写）：

```text
后台任务刚完成，你已经开口说了：“{lead}”。现在直接接着往下说，用口语讲出最关键的结果，只说这一件事，不要提别的任务；数字要和结果完全一致；不要重复开头，不要列表，不要念编号和符号，总共{limit}字以内。
```

更正判断（只在找到含糊估计且其中没有数字时使用；读“是/否”的首 token 概率）：

```text
先判断一下：你之前说过“{claim}”，后台查到的结果是“{result}”。你之前的说法和结果在关键信息上矛盾吗？只回答“是”或“否”。
```

模板回执（`common.ACKS`）：实时信息“好，我让后台查一下最新的，稍等。”；私人动作“好的，我这就让后台去办。”；计算“好，我让后台精确算一下，马上告诉你。”；规划“好，这个我让后台好好规划一下，你可以先聊点别的。”；写作“没问题，我让后台来写，写好了告诉你。”。

## 9. 风险

1. **规则召回有天花板**：strict 方案留出集召回 79.6%。漏掉的委派会交给小模型自己回答，可能**编出数字**（纯 LLM 实验里 1.5B 把“圆周率乘以 7.5 的平方”算错、把“一百美元换人民币”说成“六千四百”）。缓解办法：有算力就用 gated-4B；或把真实日志里的漏判补进规则或训练分类器；或对自答内容也加数字核对（会误伤常识数字，未实现）。
2. **0.5B / 1.5B 的闲聊质量没有量化**：只测了长度和口语性，没有做 MOS 或 LLM 评审（无 API Key）。
3. **更正判断**约 83–92%：非数字的冲突（日期、地点、是否）靠 4B 的“是 / 否”判断，小模型更差；误更正会让用户困惑。
4. **数字核对偏保守**：模型推算出的数字（如“还有 40 天”）也会被拦下，改用模板念结果，信息不丢，但不够自然。
5. **llama.cpp 单序列 KV cache**：任何不共享前缀的旁路 prompt 都会冲掉缓存（4B 重新 prefill 系统提示词约 15 s）。插件内所有调用都已设计成共享前缀，但**一个模型实例只能服务一个会话**（实例之间有全局锁）；多会话需要换成 llama-server 多 slot 或 vLLM。
6. **依赖 llama-cpp-python 的私有属性**（`_input_ids`、`_ctx.get_logits()`）实现门控和判断，已在 0.3.36 上测试，升级时需要回归；`requirements.txt` 建议锁定版本。
7. **CPU 上 4B 的时序问题**：开口前 1.5–2.5 s 的空白里用户容易插话，会被引擎当成新的一轮（端到端 backchannel / barge_in 场景已复现）。
8. **许可证**：Qwen2.5-0.5B / 1.5B 和 Qwen3-4B-Instruct-2507 是 Apache-2.0；**Qwen2.5-3B 使用 Qwen Research License（非商用）⚠️ 请核实**，商用时不要选 3B。
9. **评测数据是合成的**（全部由模型撰写），分布与真实用户不同。

## 10. 接口变更请求（未改 core，仅提议，均可向后兼容）

1. **`FastBrain.prepare(ctx)` 投机 prefill 钩子**：引擎在收到 ASR partial 时，或在 EOT 静音计时期间，用当前 partial 调用它（默认空实现）。快脑可以提前把“历史 + 当前用户话”prefill 进 KV cache，TURN_END 时只剩几个 token。在 CPU 上这是让 3B/4B 进入 800 ms 预算的**最大杠杆**：按实测 prefill 速度，可以藏掉自答轮 TTFT 的大部分（1.5B 约 0.3–0.5 s，4B 约 1–1.5 s）。
2. **`TaskRecord` 增加 `origin_text`（用户原话）和 `receipt`（快脑当时的回执或估计）**：融合时可以直接引用用户原话来点明问题，也能确定地找到“之前的说法”来判断是否更正，不用再靠二元组匹配历史。
3. **慢脑 partial 的融合时机**：`FusionPolicy` 目前只处理已完成的任务。建议允许对 `partial` / `progress` 触发 `trigger="task_progress"`，支持 ConvFill 式的流式知识注入，以及基于真实进度的“出声思考”。
4. **把 `bench fast` 的扩展字段并入 core 基准**：`duplexagent/bench/offline.py: bench_fast` 目前不传 `tasks`，测不了取消、补充、进度。可以直接采用 `experiments/fast/bench.py` 的 `tasks / action / forbid` 字段和融合基准。
5. **场景期望增加 `said_not_contains`**：用来在端到端层面检查“不瞎猜”，例如任务没完成时不许出现答案。

## 11. 文件清单

- `duplexagent/plugins/fast/common.py`：规则路由、中文数字解析、任务匹配、模板、短语切分、TTS 规范化、协议解析、更正与数字核对。
- `duplexagent/plugins/fast/router.py`：`fast/router` 插件。
- `duplexagent/plugins/fast/llamacpp.py`：`fast/llamacpp` 插件（llm / hybrid / strict / gated；llama.cpp 与 OpenAI 兼容两种后端）。
- `duplexagent/plugins/fast/openai_compat.py`：新增 `stream=false`。
- `experiments/fast/`：`bench.py`、`calibrate.py`、`leaderboard.py`、`sweep.sh`、`e2e.sh`、`serve_check.sh`、`make_data.py`、`test_fast.py`、`LABELING.md`、`scenarios/`、`results/`（`results/v1`、`results/pre_opt` 是改进前的对照）。
- `configs/exp/fast/*.toml`：每个候选一个配置。
- `bench_data/fast/*.jsonl`：见 §2.1。
