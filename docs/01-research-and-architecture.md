# 快慢脑全双工语音对话引擎：调研与架构方案

> 状态：调研 v0.1（2026-10）。标 ✅ 的项目本次检索已核实；标 ⚠️ 的出自记忆，用前请再核对版本与许可证。

## 0. 结论速览

1. **业界在 2025–2026 收敛到同一个形态**：常驻的低延迟“对话层”（快脑）负责听、说、轮次与打断，复杂任务以**异步工具调用**下发给后台“推理/执行层”（慢脑），结果回来后由快脑在合适时机口语化地融合进对话。OpenAI GPT-Live 的“交互层 + 委派层”、Gemini Live 的后台推理 + 异步工具、LiveKit 的 Talker-Reasoner 都是这个模式。
2. **端到端全双工模型（Moshi、MiniCPM-o、Freeze-Omni 等）适合做“快脑的耳朵和嘴”，但不适合单独承担复杂推理与工具**；今天在工程上可控、可调试、可换模型的方案仍然是 **级联（ASR → 轮次/打断判别 → 快脑 LLM → TTS）+ 异步慢脑**。
3. 真正的难点不在模型，而在 **会话状态机**：打断（barge-in）、回声消除、“慢脑结果什么时候插话”、被打断后上下文如何截断、多个慢任务并发时的优先级。
4. 推荐路线：**基于 Pipecat 或 LiveKit Agents 起步**（不自己写传输和音频管线），自研 “Turn Manager + Task Orchestrator（慢脑任务总线）+ Fusion Policy（融合策略）” 三个模块——这是差异化所在。

---

## 1. 业界做法与可借鉴经验

### 1.1 两条技术路线

| 路线 | 代表 | 优点 | 缺点 |
|---|---|---|---|
| 级联 / 半级联 | Pipecat、LiveKit Agents、TEN、FireRedChat ✅ | 每段可替换、可观测；能直接用最强文本 LLM 与工具；中文 ASR/TTS 选择多 | 延迟靠工程压缩（目标端到端 < 800ms）；副语言信息（语气、笑声）丢失 |
| 端到端全双工语音模型 | Moshi ✅（~200ms 实测延迟，双流并行建模）、MiniCPM-o 4.5 ✅（9B，1Hz 块式双工）、Freeze-Omni ✅、NemotronLabs VoiceChat ✅（带工具调用） | 天然支持重叠说话、附和语（嗯/对）、极低延迟 | 智力与工具能力弱（VoiceBench：Moshi 29.5 / Freeze-Omni 55.2 / MiniCPM-o 76.1 ✅）；难以控制和调试；中文与私有化部署受限 |

**借鉴**：Full-Duplex-Bench-v2 ✅（ACL 2026）发现全双工系统在双方同时说话时普遍会混乱——说明“轮次/打断判别”值得做成独立、可评测的模块，而不是寄望于一个模型全包。

### 1.2 快慢脑（Talker–Reasoner）模式

- **理论来源**：DeepMind《Agents Thinking Fast and Slow: A Talker-Reasoner Architecture》(Christakopoulou et al., 2024, arXiv 2410.08328) ⚠️——Talker 负责对话与快速反应，Reasoner 负责多步推理/规划并写入共享的“信念状态（belief state）”，Talker 读取最新信念状态作答。**“共享状态 + 异步写入”** 正是你需要的融合机制。
- **LiveKit 实现** ✅：快脑是普通 Agent；慢脑是独立 LLM 实例、**有自己的上下文**（不继承全部对话历史，降低成本）；两者唯一的连接是一个 **async function tool**。工具的第一次进度回报就“释放”对话，快脑说“我去查一下”后继续聊；慢脑完成后，快脑在**空闲时**主动补一轮，并点明“回答的是刚才哪个问题”。快脑被约束为**不许对已委派的问题瞎猜**。
  - 官方给出的“不适合”场景很有价值：答案必须核实后才能说出口（报价、医疗）；快脑无法填补等待；通话极短（< 30s）。
- **OpenAI** ✅：GPT-Live（2026-07）为“持续交互层 + 委派层（后台交给更大模型）”；Responses API 的工具支持 `async: true`，模型发出调用后继续工作，结果通过 `call_id` 在后续请求中回填——**用 call_id 关联任务与结果** 是值得照搬的协议设计。
- **Google Gemini Live** ✅：文档描述其在持续流式输出音频的同时进行后台推理和异步工具调用。

### 1.3 学术上的“边想边说”

| 工作 | 要点 | 可借鉴 |
|---|---|---|
| ConvFill / *Thinking While Speaking* (arXiv 2511.07397) ✅ | 小 Talker（135M–1.7B）即时回应，慢 Reasoner 以**流式知识块**注入；准确率距前沿模型 6.3% 以内，首响毫秒级；**代码/模型/数据开源**（GitHub: vysri/conversational-infill） | 慢脑不一定一次性返回，可以**流式推送中间结论**，快脑边说边吸收；超时插入“静默 token”避免胡编 |
| AsyncVoice Agent (arXiv 2510.16156) ✅ | 推理后端与语音前端解耦，叙述与推理并行 | 慢脑的“思考过程”可被快脑口语化播报（类似 Codex 显示进度） |
| Slow Thinker / Fast Talker (arXiv 2603.02206) ✅ | 慢脑异步**预测接下来 3–5 个话题**并预取文档 | 慢脑不仅被动接任务，还能**主动预热**，降低下一轮延迟 |
| Spoken LMs that Think Aloud (arXiv 2609.26488) ✅ | 主推理流 + 轻量“出声思考”流，显著减少无遮蔽静默 | “填充语”应由模型基于真实进度生成，而非固定话术 |
| Qwen2.5/3-Omni Thinker–Talker ⚠️ | 模型内部的 Thinker（理解与文本生成）与 Talker（语音生成）分工 | 注意：这是**模型内部**的分工，不等于你要的“快慢脑任务异步” |

### 1.4 轮次检测与打断——经验法则

- **VAD 只是第一层**：纯静音阈值会在用户停顿思考时抢话。业界做法是 **VAD（是否有声） + 语义/声学 End-of-Turn 模型（话说完没有）**。
- **打断要区分三类**：真打断（“不对，我是说…”）→ 立刻停 TTS；附和语（“嗯”“对”）→ 不停；噪声/回声 → 忽略。常用策略：用户语音持续 > 200–300ms **且** ASR 出现非附和词时才确认打断；在此之前先**压低音量（ducking）**而不是立刻切断。
- **被打断后要截断上下文**：只把**实际播出**的那部分文字写回对话历史（按 TTS 字级时间戳对齐），否则 LLM 以为自己说完了。Pipecat/LiveKit 都内建了这个逻辑——这是用框架而不是自研的重要理由。
- **回声消除（AEC）是全双工的前提**：没有 AEC，助手会被自己的声音打断。浏览器/WebRTC 自带 AEC；硬件端需用 WebRTC APM / SpeexDSP 或芯片自带 AEC。

---

## 2. 可靠的开源项目清单

### 2.1 编排框架（建议二选一作为底座）

| 项目 | 语言 | 亮点 | 建议 |
|---|---|---|---|
| **Pipecat** (pipecat-ai/pipecat) | Python | Frame 管线模型、可插拔服务极多、内建打断与上下文截断、Smart Turn 已成为默认轮次策略 ✅ | **首选**：管线透明，最易插入自定义“慢脑”处理器 |
| **LiveKit Agents** (livekit/agents) | Python / Node | WebRTC 传输成熟、官方 Talker-Reasoner/子代理委派示例 ✅、Turn Detector | 需要大规模实时传输/多端时首选 |
| **TEN Framework** (TEN-framework) ✅ | C++/Go/Python 多语言扩展 | 声网出品；TEN VAD、TEN Turn Detection（**中英双语**） | 中文场景重点参考其 VAD/轮次模型 |
| **FireRedChat** (arXiv 2509.06502) ✅ | — | 小红书，可插拔全双工，级联与半级联两种实现 | 参考其中文全双工控制模块设计 |

### 2.2 组件

| 环节 | 开源候选 | 说明 |
|---|---|---|
| VAD | Silero VAD ⚠️、**TEN VAD** ✅（ONNX/WASM，宣称语音→静音切换比 Silero 快数百毫秒，厂商自测） | 两个都试，按自己数据测 |
| 轮次检测 | **Pipecat Smart Turn v3.x** ✅（音频模型，BSD-2，23 语言，CPU 约 10ms，8MB 量化）；**LiveKit Turn Detector** ✅（基于 Qwen2.5-0.5B 的文本模型；v1-mini 开放权重）；**TEN Turn Detection** ✅（中英，Apache-2.0 附加限制） | 推荐 **音频 Smart Turn + 文本模型** 双信号融合 |
| 流式 ASR（中文） | FunASR / Paraformer-streaming、SenseVoice ⚠️、FireRedASR ⚠️、Whisper 系（流式需改造）⚠️；云：阿里/讯飞/Deepgram | 必须支持**流式 partial 结果**——轮次与打断判别依赖它 |
| 快脑 LLM | Qwen3 小尺寸（4B–8B）/ 云端 Haiku、Flash 级模型 ⚠️ | 目标 TTFT < 300ms；可用 ConvFill 思路微调专门的 talker |
| 慢脑 | 大模型（Claude Opus/Sonnet、GPT、Gemini Pro 等）+ 工具 / MCP / 代码执行 | 任何 agent 框架都可以（Claude Agent SDK、LangGraph…），通过任务总线接入 |
| 流式 TTS（中文） | CosyVoice 2/3、Fish-Speech、IndexTTS、F5-TTS ⚠️；云：火山/MiniMax/ElevenLabs | 需要**字级时间戳**（用于打断截断）与**首包 < 200ms** |
| 端到端全双工（研究/对照） | Moshi ✅、MiniCPM-o 4.5 ✅、Freeze-Omni ✅、VITA ⚠️ | 可作为后续“快脑升级”的对照组 |
| 快慢脑参考代码 | **ConvFill**（vysri/conversational-infill）✅、LiveKit subagent delegation 示例 ✅ | 最直接的参考实现 |
| 评测 | Full-Duplex-Bench v1/v2 ✅、VoiceBench ✅ | 建立自己的回归评测 |

---

## 3. 我们的架构方案

### 3.1 总体结构

```
          ┌──────────────────────────── Transport (WebRTC/WebSocket, AEC) ───────────────────────────┐
 mic ───► │ VAD ─► Streaming ASR ─┬─► Turn Manager ──► Fast Brain (small LLM) ──► Streaming TTS ──► │ ───► speaker
          │                       │   (EOT + 打断判别)       ▲     │  delegate(task)                     │
          │                       │                         │     ▼                                    │
          │                       │                 Fusion Policy ◄── Task Orchestrator (event bus) ──┐ │
          │                       │                 (何时/如何插话)       │  spawn / cancel / progress │ │
          │                       │                         ▲             ▼                            │ │
          │                       └── partial transcript ──►│      Slow Brain workers (big LLM + tools)┘ │
          │                                       Shared Session State (对话历史 / 任务表 / 信念状态)   │
          └──────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 模块职责

**① 音频与 ASR 层**
- 流式 ASR 输出 `partial` 与 `final`；保留词级时间戳和置信度。
- AEC 放在最前端；TTS 正在播放的参考信号同时送 AEC 与打断判别。

**② Turn Manager（轮次 / 打断）——一个显式状态机**

```
LISTENING ──(EOT概率>θ 且 静音>t)──► THINKING ──(快脑首token)──► SPEAKING
    ▲                                                                │
    │◄─────────────(播完)────────────────────────────────────────────┘
    │◄──(确认打断: 语音>250ms 且 非附和词)── SPEAKING ──(疑似打断)──► DUCKING ──(超时未确认)──► SPEAKING
```

- 输入信号：VAD、Smart Turn（声学 EOT）、文本 EOT 模型（基于 partial 转写）、当前助手状态。
- 输出事件：`user_turn_end`、`barge_in`、`backchannel`（附和，不打断）、`ignore`。
- 高级：**投机执行**——在 EOT 概率较高但未确定时提前让快脑开始生成（不播放），确认后直接出声，可省 200–400ms。

**③ Fast Brain（快脑）**
- 小模型、短 system prompt、精简上下文（最近 N 轮 + 会话摘要 + 任务表摘要）。
- 只有三种动作：**直接回答**、**委派** `delegate(task_spec)`、**澄清提问**。委派判据写进 prompt，并可以用一个轻量分类器兜底（需要工具/需要最新信息/多步推理/需要高准确度 → 委派）。
- 委派后必须给出**口头回执**（“好，我让后台去查，你可以继续说别的”），并**禁止对已委派问题给出猜测答案**（LiveKit 经验）。

**④ Task Orchestrator（慢脑任务总线）——核心自研模块**
- 每个任务：`task_id`、`origin_turn_id`、`spec`（由快脑生成的自包含任务描述，**不传全量历史**）、`priority`、`status`（queued/running/progress/done/failed/cancelled）、`deadline`。
- 事件：`task.created / task.progress / task.partial_result / task.done / task.failed / task.cancelled`。
- 支持：并发多任务；用户说“算了/不用查了”时快脑可调用 `cancel(task_id)`；用户补充条件时调用 `amend(task_id, info)`。
- 慢脑可以是任何 agent（大模型 + MCP 工具 + 代码执行）；流式回报中间结论（ConvFill 思路）。
- 可选：**主动预取**——根据近几轮对话预测后续话题并预热检索（arXiv 2603.02206 思路）。

**⑤ Fusion Policy（融合策略）——决定慢脑结果“何时、以何种方式”回到对话**

| 当前状态 | 结果到达时的处理 |
|---|---|
| LISTENING 且用户在说话 | 排队，不插话 |
| 空闲（双方都沉默 > 1s） | 快脑主动发起一轮：“刚才你问的 X，结果是……”（点明对应问题） |
| SPEAKING（快脑在聊别的） | 在当前句子边界后插入一句过渡语，或排到本轮结束；高优先级任务可在句界打断自己 |
| 结果与快脑之前的说法冲突 | 显式更正：“我刚才说的不准确，后台查到的是……” |
| 结果很长 | 快脑先口头讲摘要，详细内容推到屏幕/卡片（类 Codex 体验） |

- 实现方式：慢脑结果**不直接 TTS**，而是作为一条 `system/tool` 消息写入共享状态，由快脑“改写成口语”后说出——保证语气一致、长度可控。

**⑥ Shared Session State**
- 对话历史（只记录**实际播出**的助手文本）、任务表、慢脑写入的“信念状态/事实笔记”。
- 快脑每轮读取任务表摘要，从而能回答“查到了吗？”这类问题并播报进度。

### 3.3 延迟预算（目标：用户说完到助手出声 < 800ms）

| 环节 | 目标 |
|---|---|
| VAD + EOT 判定 | 150–300ms（静音阈值 + Smart Turn） |
| ASR final | 与 EOT 并行，< 100ms 额外开销 |
| 快脑 TTFT | < 300ms（小模型 / 投机执行 / prompt cache） |
| TTS 首包 | < 200ms |
| 网络 | < 100ms（WebRTC） |

### 3.4 分阶段实施

| 阶段 | 内容 | 验收 |
|---|---|---|
| **P0 骨架（1–2 周）** | Pipecat 管线：WebRTC + Silero/TEN VAD + 流式中文 ASR + 快脑 LLM + 流式 TTS；用框架自带打断 | 半双工可用对话，端到端 < 1.2s |
| **P1 轮次与打断** | 接入 Smart Turn + 文本 EOT；附和语识别；ducking；打断后上下文截断 | 抢话率、误打断率、漏打断率三项指标 |
| **P2 快慢脑** | `delegate/cancel/amend` 工具；Task Orchestrator（asyncio 事件总线）；慢脑接入大模型 + 工具；Fusion Policy | 慢任务执行期间可持续闲聊；结果在合适时机正确融合 |
| **P3 体验打磨** | 进度播报、投机执行、主动预取、屏幕卡片、多任务优先级 | 主观 MOS + Full-Duplex-Bench 风格回归集 |
| **P4 探索** | 用 ConvFill 方法微调专用 talker；评估把快脑换为端到端全双工模型（MiniCPM-o/Moshi 类） | 与 P3 级联方案对比 |

### 3.5 风险与对策

- **快脑过度委派 / 委派不足** → 记录委派决策，离线标注后调 prompt 或训练小分类器。
- **慢脑结果打断用户思路** → Fusion Policy 保守默认（只在空闲时插话），高优先级才在句界插入。
- **上下文不一致**（快脑说过的话与慢脑结论冲突）→ 慢脑结果带 `origin_turn_id`，快脑融合时必须先检查并更正。
- **中文轮次判别数据少** → 优先评测 TEN Turn Detection（中英）与 Smart Turn 多语言版，并尽早采集自有数据微调。
- **许可证** → TEN Turn Detection 为 Apache-2.0 附加限制；部分社区轮次模型为非商用；Moshi/MiniCPM-o 需查看各自权重许可。

---

## 参考资料

- LiveKit：Talker-Reasoner 模式 — https://livekit.com/blog/talker-reasoner-pattern-voice-agents
- LiveKit：Solving end-of-turn detection — https://livekit.com/blog/solving-end-of-turn-detection
- Pipecat Smart Turn — https://github.com/pipecat-ai/smart-turn ， https://docs.pipecat.ai/server/utilities/turn-detection/smart-turn-overview
- TEN Turn Detection — https://github.com/TEN-framework/ten-turn-detection ；TEN VAD — https://github.com/TEN-framework/ten-vad
- OpenAI 异步工具调用 — https://developers.openai.com/api/docs/guides/async-tool-calling
- Gemini Live extended thinking — https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live-extended-thinking.md.txt
- GPT-Live 范式分析 — https://yage.ai/share/gpt-live-voice-programming-paradigm-en-20260709.html
- ConvFill / Thinking While Speaking — https://arxiv.org/abs/2511.07397
- AsyncVoice Agent — https://arxiv.org/pdf/2510.16156v1
- Slow Thinker / Fast Talker — https://www.alphaxiv.org/overview/2603.02206
- Spoken Language Models that Think Aloud — https://arxiv.org/pdf/2609.26488
- FireRedChat — https://arxiv.org/pdf/2509.06502
- Moshi — https://arxiv.org/html/2410.00037v2
- NemotronLabs VoiceChat（含 VoiceBench 对比）— https://arxiv.org/pdf/2609.21967
- MiniCPM-o 4.5 — https://minicpmo45.modelbest.cn/
- Full-Duplex-Bench-v2 — https://preview.aclanthology.org/ingest-acl/2026.acl-short.4/
- Agents Thinking Fast and Slow (Talker-Reasoner) — https://arxiv.org/abs/2410.08328 ⚠️
