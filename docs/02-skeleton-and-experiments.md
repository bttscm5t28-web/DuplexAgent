# 骨架说明与五路实验规范

## 1. 骨架结构

```
duplexagent/
  core/
    types.py          共享数据类型（AudioFrame / Transcript / TurnEvent / Say / Delegate / TaskRecord / TTSChunk …）
    interfaces.py     8 类可替换模块的抽象接口
    registry.py       插件注册表：@register(kind, name)，配置里用 impl = "name" 选择
    engine.py         全双工引擎：状态机、打断、上下文截断、响应流水线
    orchestrator.py   慢脑任务总线：spawn / cancel / amend / 进度
    timeline.py       事件日志（所有指标都从它计算）
    llm.py            OpenAI 兼容 Chat 客户端（vLLM / Ollama / DeepSeek / 通义 / OpenAI 都可）
  plugins/<kind>/<name>.py   各模块实现（自动发现）
  metrics.py          端到端场景打分
  bench/offline.py    单模块离线基准（asr / fast / slow / tts）
configs/default.toml  基线：全部 mock，无需模型和网络即可跑通
scenarios/*.toml      端到端场景（含期望值）
bench_data/*.jsonl    单模块基准数据（种子集，各实验应扩充）
```

### 可替换模块

| kind | 接口 | 基线实现 | 作用 |
|---|---|---|---|
| `input` | `AudioInput.frames()` | `sim`（脚本化模拟用户）, `wav` | 麦克风 / 文件 / WebRTC |
| `output` | `AudioOutput.play()/stop()` | `sim`, `wav` | 扬声器 / 文件 / WebRTC |
| `asr` | `ASR.transcribe(frames) → Transcript*` | `mock`（读模拟真值） | 流式识别 |
| `turn` | `TurnDetector.on_audio / on_transcript → TurnEvent*` | `rule`（能量 VAD + 静音超时 + 附和词表） | VAD、说完判定、打断判别 |
| `fast` | `FastBrain.respond(ctx) → Say / Delegate / CancelTask / AmendTask*` | `rule`, `openai_compat` | 快脑 |
| `slow` | `SlowBrain.run(task) → TaskUpdate*` | `mock`, `openai_compat` | 慢脑 |
| `tts` | `TTS.synthesize(text*) → TTSChunk*` | `mock` | 流式合成（chunk 带文本对齐） |
| `fusion` | `FusionPolicy.select(state, ready)` | `idle` | 慢脑结果何时融合回对话 |

### 引擎关键行为（与模块无关，模块不用重复实现）

- 状态：LISTENING / THINKING / SPEAKING（由是否有响应任务、是否在放音推导）。
- `TURN_END` → 写入用户消息 → 启动快脑响应。
- 从开始响应起，助手就算“占着话轮”（`TurnContext.agent_speaking` 为真），不只是放音期间。THINKING 期间用户又开口 → **暂停出声**（`hold`），快脑继续想，再看轮次判别的结论：`BACKCHANNEL` → 恢复出声；`BARGE_IN` / `TURN_END` → 丢弃这次响应；如果用户是在上一轮结束后 `merge_window_ms`（默认 1 s）内接着说的，把新话**合并进上一轮**（`turn_merged`）一起回答，否则作为新的一轮；`hold_release_ms`（默认 2.5 s）内没有结论 → 恢复出声。旧行为是直接丢弃响应，可用 `engine.resume_policy = "cancel"` 恢复。助手还没出声时 `TurnContext.agent_audible` 为假，轮次插件此时不应根据 partial 提前判打断（反正已经暂停出声），而应等到这句话说完、用完整识别结果判断。
- `BARGE_IN` → 取消响应、`output.stop()`；**只把实际播出的文字写入历史**，并标记 `interrupted`。
- 快脑 `Delegate` → 任务总线异步执行；`FusionPolicy` 认为合适时以 `trigger="task_result"` 再触发快脑，把结果融合进对话；融合回复被打断时结果会重新排队：一个字都没播出时一定重排；说到一半被打断时，同一任务最多重排 `max_redeliver` 次（默认 2）。

### 运行

```bash
python -m duplexagent list                                   # 已注册插件
python -m duplexagent run -s scenarios/delegate.toml          # 跑一个场景，打印事件时间线
python -m duplexagent eval scenarios/*.toml --out r.json      # 跑全部场景并打分
python -m duplexagent bench fast --data bench_data/fast_delegation.jsonl
python -m duplexagent eval scenarios/*.toml --set turn.eot_silence_ms=300   # 任意覆盖配置
python -m pytest -q
```

场景里的用户话语可以带 `wav = "path.wav"`（16 kHz 单声道 int16），这样真实 ASR / VAD 也能用同一套场景评测；不带时用噪声模拟语音并附带真值文本（只适用于 `asr=mock`）。

---

## 2. 五路实验规范

每个方向一个独立 session、一个独立分支，**只在自己的地盘里改**，这样最终可以无冲突合并、任意组合。

| 方向 | 分支 | 自己的地盘 |
|---|---|---|
| ASR | `claude/exp-asr` | `duplexagent/plugins/asr/`、`experiments/asr/`、`configs/exp/asr/`、`bench_data/asr/` |
| 打断/轮次 | `claude/exp-turn` | `duplexagent/plugins/turn/`、`experiments/turn/`、`configs/exp/turn/`、`scenarios/turn/` |
| 快脑 | `claude/exp-fast` | `duplexagent/plugins/fast/`、`experiments/fast/`、`configs/exp/fast/`、`bench_data/fast/` |
| 慢脑 | `claude/exp-slow` | `duplexagent/plugins/slow/`、`experiments/slow/`、`configs/exp/slow/`、`bench_data/slow/` |
| TTS | `claude/exp-tts` | `duplexagent/plugins/tts/`、`experiments/tts/`、`configs/exp/tts/`、`bench_data/tts/` |

### 规则

1. **不改 `duplexagent/core/` 的接口**。确实需要时：只做向后兼容的增量改动，单独一个 commit，标题以 `core:` 开头，并在报告的「接口变更请求」一节说明原因。
2. 新实现放 `duplexagent/plugins/<kind>/<name>.py`，用 `@register("<kind>", "<name>")` 注册；可选依赖在函数内延迟 import，并写进 `experiments/<kind>/requirements.txt`。模型权重不入库（`models/` 已忽略），在报告里写下载方式。
3. 每个候选一个配置 `configs/exp/<kind>/<name>.toml`（基于 `configs/default.toml` 只改本模块）。
4. **门槛**：`python -m pytest -q` 通过；把候选换进骨架后 `python -m duplexagent eval scenarios/*.toml` 不退化（需要真实音频的方向用带 `wav` 的场景版本）。
5. 结果 JSON 存 `experiments/<kind>/results/`，`experiments/<kind>/REPORT.md` 维护：候选列表、指标排行榜、硬件环境、推荐方案与理由、风险、接口变更请求。
6. 运行环境：当前容器 4 核 CPU、15 GB 内存、无 GPU，可访问 PyPI 和 Hugging Face，**没有 LLM API Key**。CPU 上能跑的就实测；需要 GPU 或 API Key 的候选先写好适配器并用小模型/本地服务验证接口，在报告里注明待补测。

### 各方向主要指标

| 方向 | 主要指标 | 次要指标 |
|---|---|---|
| ASR | 中文 CER（`bench asr`）、实时模式下 final 相对说完的延迟 | 首个 partial 延迟、RTF、CPU 占用、中英混说、噪声鲁棒 |
| 打断/轮次 | 抢话率（停顿被切断）、误打断率（附和/噪声被当打断）、漏打断率、打断停止延迟、说完到响应延迟 | 中文语气词、长停顿思考、回声场景 |
| 快脑 | 首 token 延迟 TTFT、委派判断准确率/召回（`bench fast`） | 回复口语化程度、被打断后的衔接、融合结果的质量、不对已委派问题瞎猜 |
| 慢脑 | 任务正确率（`bench slow`）、端到端耗时 | 进度回报质量、工具调用成功率、成本、取消/补充的响应 |
| TTS | 首包延迟、RTF、文本对齐正确率（`bench tts`） | 中文自然度（MOS/UTMOS 等自动评分）、流式分句策略、可打断性 |

### 最终选型

五路结束后，在主分支合并所有插件，用 `--set` 组合各方向前两名跑全量场景 + 真实音频场景，按「端到端延迟 < 800 ms、误打断率、任务融合成功率、主观听感」挑出最终组合。
