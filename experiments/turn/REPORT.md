# 打断/轮次判别实验报告（`claude/exp-turn`）

## 0. 结论速览

| 用途 | 推荐 | 关键数据（真实语音 + 带延迟 ASR，wav_lagged） |
|---|---|---|
| **默认（有 ~500 MB 内存和 1–2 个 CPU 线程余量）** | `fusion`：Silero VAD + 中文文本规则 + Qwen2.5-0.5B「LiveKit 式」文本 EOT 融合 | 27/28 通过；抢话 10%（仅剩刻意设计的难例）；误打断 0；说完→TURN_END p50 0.36 s；打断停止 p50 0.35 s；两种家电噪声下综合分第一 |
| **轻量（纯 CPU，~0.1 ms/帧，无 LLM）** | `adaptive_silero`：Silero VAD + 中文文本规则 | 27/28；抢话 10%；误打断 0；TURN_END p50 0.34 s；打断停止 p50 0.35 s |
| 只能用标准库 | `adaptive`：能量 VAD + 中文文本规则 | 安静环境与上面持平，**有稳态噪声就失效**（吸尘器 10 dB 下 4/28） |

基线 `rule` 在同一条件下：15/28，**抢话率 100%**（10 个句中停顿全被抢话），误打断 21%，TURN_END p50 0.44 s，「停」的停止延迟约 0.7 s。

一句话：**最大收益来自“文本语义 → 自适应静音阈值”和“命令词/附和词/回声感知的打断判别”，其次是把能量 VAD 换成 Silero 以抗噪；Smart Turn v3（音频）在中文合成语音上不可靠；TEN Turn Detection 文本判别最准但 7B 在 CPU 上太慢。**

---

## 1. 候选与实现

全部候选都是 `duplexagent/plugins/turn/hybrid.py` 里同一个检测器（`HybridTurnDetector`）的不同预设，只换零件，便于公平对比：

```
VAD（energy | silero | ten）──► 有声/无声（带迟滞）
EOT 打分器（text 规则 | smart_turn 音频 | lm 文本 | ten_turn 文本）──► P(说完)，多个时加权平均
P(说完) ──► 自适应静音阈值：P≥p_high 用 min_silence_ms，P≤p_low 用 max_silence_ms，中间线性插值
打断（助手说话时）：
  命令词「停/等一下/不对/别说了/wait」──► 立刻 BARGE_IN（不等 300 ms）
  附和词或其前缀「嗯/对对对/原来…如此」──► 不打断，结束时发 BACKCHANNEL
  用户文本是助手正在说的话的片段（回声泄漏）──► 忽略
  其他文字 ──► 有声 ≥ barge_in_ms 后打断；完全没有文字 ──► 有声 ≥ barge_in_max_ms 后打断
慢模型在线程池里跑，on_audio 不阻塞；结果没回来前按保守阈值等待（上限 max_silence_ms）
自适应：TURN_END 后 1.5 s 内用户又开口（= 抢话了）→ 该会话后续等待时间 ×1.2（上限 1.6），之后缓慢回落
```

| 预设（`turn.impl`） | VAD | EOT | 许可证 | CPU 开销 |
|---|---|---|---|---|
| `rule`（基线） | 能量 | 固定 500 ms | — | ~0 |
| `silero` / `ten_vad` | Silero / TEN | 固定 500 ms（仅换 VAD） | MIT / Apache-2.0+附加条款 | 0.1 / 0.2 ms 每 20 ms 帧 |
| `adaptive` | 能量 | 中文文本规则（`lexicon.py`） | — | ~0 |
| `adaptive_silero` / `adaptive_ten` | Silero / TEN | 文本规则 | 同上 | 同上 |
| `smart_turn` | Silero | Pipecat Smart Turn v3.2 CPU 版（音频，Whisper-tiny 编码器） | BSD-2 | ~55 ms/次（每次停顿 1 次） |
| `lm_eot` | Silero | Qwen2.5-0.5B-Instruct int8 ONNX：用户话后接 `<|im_end|>`/句末标点的概率，log 校准（LiveKit 的方法，**不用 LiveKit 权重**） | Apache-2.0 | ~170 ms/次（2 线程） |
| `fusion` | Silero | 文本规则 + lm 平均；纯命令词/附和词直接判完 | Apache-2.0 / MIT | 同 lm |
| `fusion_st` | Silero | 文本规则 + lm + smart_turn(0.5 权重) | 同上 | lm + 55 ms |
| `ten_turn` | Silero | TEN Turn Detection（Qwen2.5-7B 微调）Q4_K_M GGUF，llama.cpp，读 finished/unfinished/wait 首 token 概率；也支持 OpenAI 兼容服务（vLLM） | Apache-2.0+附加条款 | ~0.9 s/次（4 线程） |

**没有跑的：LiveKit Turn Detector。** 其 LiveKit Model License 规定“只能与 LiveKit Agents 框架一起使用，不得单独或在其他框架中使用”，在本引擎里使用即违约，所以只借用其方法（`lm_eot`），不下载、不运行其权重。

实现细节：
- Smart Turn 的特征提取用 numpy 重写了 `WhisperFeatureExtractor`（与 transformers 输出最大误差 1.2e-7，模型概率完全一致），运行时不依赖 transformers/torch。
- Silero 直接用 `silero-vad` wheel 里的 ONNX（`pip install --no-deps silero-vad`，避免拉 torch）。
- 所有可选依赖都在函数内延迟 import，见 `experiments/turn/requirements.txt`；权重首次使用时从 Hugging Face 下载到 `models/`（已 gitignore）。

## 2. 评测

### 2.1 场景（`scenarios/turn/*.toml`，28 个，每个文件头写明期望）

| 类别 | 场景 | 期望 |
|---|---|---|
| 句中思考停顿（10） | 我想……订机票（0.9 s）、那个……、短停顿 0.55 s、长停顿 1.5 s、列举中“还有……”、报手机号中途、两次停顿、“因为……所以”、“大一点的……杯子”、**难例**“我今天有点累……而且还有点头疼” | 不抢话，只在整句后响应一次 |
| 完整短句/追问（2） | 今天星期几 / 谢谢 / 明天会下雨吗；回答后 1 s 内追问“那后天呢”“上海呢” | 每轮都响应，延迟尽量低 |
| 附和语（5） | 嗯/对/嗯嗯/好的；对对对/是的/没错/明白；哦/这样啊/哈哈/真的啊；嗯哼/原来如此/可以/我知道了；先“嗯”再“嗯那我换个问题” | 附和不打断；以附和词开头的真打断要打断 |
| 短命令打断（5） | 停、等一下、不对不对、别说了、wait | 快速停下 |
| 长回复打断（3） | 长回复中段“等等我是想问上海的”、“嗯不是这个意思”、“那明天呢” | 停下并接新轮次 |
| 噪声（3） | 助手说话时咳嗽 0.3 s/敲键盘 0.4 s、1.2 s 持续敲击、助手空闲时咳嗽和敲门 | 既不打断也不成轮次 |

真实语音版 `scenarios/turn/wav/*.toml`（含 5 个骨架场景的 `base_*.toml`）由 `experiments/turn/prepare_data.py` 生成：
- 语音：Piper TTS（zh_CN huayan / xiao_ya / chaowen 三个音色轮换）。**带停顿的句子整句合成后在词边界附近能量最低处切开**，保证前半句保留“没说完”的韵律，而不是单独合成“我想”得到的降调完结音。
- 噪声：ESC-50（Hugging Face 上 DynamicSuperb 镜像）里的 coughing / keyboard typing / door or wood knock；背景噪声 vacuum cleaner / washing machine。
- wav 不入库，`python experiments/turn/prepare_data.py` 约 30 s 重新生成。

### 2.2 评测条件（`experiments/turn/run_suite.py`）

| 条件 | 输入 | ASR |
|---|---|---|
| text_ideal | 骨架的噪声 PCM 模拟语音（强制能量 VAD，只考 EOT/打断逻辑） | mock，无延迟 |
| text_lagged | 同上 | partial +300 ms，final +200 ms |
| wav_ideal | 真实语音 + 真实噪声 | 无延迟 |
| **wav_lagged** | 真实语音 + 真实噪声 | partial +300 ms，final +200 ms（最接近线上） |
| wav_vacuum / wav_washer | 真实语音 + 持续吸尘器/洗衣机背景噪声，SNR 10 dB（`experiments/turn/noisy_sim.py`） | 带延迟，ASR 视为抗噪 |
| base_wav | 5 个骨架场景的真实语音版（门槛：不退化） | 无延迟 |

ASR 统一用 `experiments/turn/lagged_asr.py`（`asr.impl=lagged_mock`）：在骨架 mock 上加流式延迟，并修复 mock 在 wav 场景下的一个问题（真值字会落在录音前导静音帧上，导致立即吐出单字 final，见 §5）。

### 2.3 指标（`experiments/turn/evaluate.py`，补充 `metrics.py` 没有的）

- **抢话率**：有句中停顿的轮次里，用户还没说完就 TURN_END 的比例。
- **误打断率**：助手说话期间的附和/噪声导致中断的比例。
- **漏打断率**：应打断（kind=barge_in）却 0.8 s 内没停的比例。
- **噪声/附和误成轮次**：噪声，或助手说话时的附和，产生了 TURN_END 的次数。
- **漏判说完**：用户说完 3 s 内没有 TURN_END 的轮次数。
- **说完→TURN_END**、**说完→助手出声**：都从真值的用户说完时刻算起。
- **打断停止延迟**：从打断语开始到引擎 interrupt。
- **综合分**（越低越好）：各错误率之和 + 0.5 × [(TURN_END p50 − 0.3 s) + (停止 p50 − 0.3 s)]，用于排序。

另有离线文本 EOT 基准 `experiments/turn/bench_text_eot.py`：`bench/text_eot.jsonl`，100 句中文最小对（“帮我设一个 / 帮我设一个明天早上七点的闹钟”），与场景文本不重叠，规则没有在它上面调过。

## 3. 排行榜

完整表（7 个条件 × 11 个候选）见 `results/leaderboard.md`，原始数据见 `results/suite_<条件>.json`。

### 3.1 wav_lagged（真实语音 + 带延迟 ASR）

| 候选 | 综合分↓ | 通过 | 抢话率 | 漏打断 | 误打断 | 误成轮次 | 漏判说完 | TURN_END p50 / p90 | 出声 p50 | 停止 p50 / p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| adaptive_silero | **0.146** | 27/28 | 0.1 | 0 | 0 | 0 | 0 | 0.34 / 0.45 | 0.57 | 0.35 / 0.55 |
| adaptive_ten | 0.150 | 27/28 | 0.1 | 0 | 0 | 0 | 0 | 0.35 / 0.45 | 0.57 | 0.35 / 0.56 |
| fusion | 0.157 | 27/28 | 0.1 | 0 | 0 | 0 | 0 | 0.36 / 0.53 | 0.59 | 0.35 / 0.55 |
| adaptive | 0.196 | 27/28 | 0.1 | 0 | 0 | 0 | 0 | 0.33 / 0.36 | 0.56 | 0.46 / 0.54 |
| lm_eot | 0.253 | 26/28 | 0.2 | 0 | 0 | 0 | 0 | 0.35 / 0.48 | 0.59 | 0.35 / 0.56 |
| fusion_st | 0.291 | 26/28 | 0.2 | 0 | 0 | 0 | 0 | 0.42 / 0.59 | 0.65 | 0.36 / 0.56 |
| ten_turn | 0.675 | **28/28** | **0.0** | 0 | 0 | 0 | 3 | 1.44 / 1.73 | 1.69 | 0.36 / 0.54 |
| smart_turn | 0.751 | 21/28 | 0.7 | 0 | 0 | 0 | 0 | 0.35 / 1.56 | 0.57 | 0.35 / 0.54 |
| silero | 1.274 | 15/28 | 1.0 | 0 | 0.12 | 0 | 0 | 0.55 / 0.57 | 0.78 | 0.36 / 0.80 |
| ten_vad | 1.322 | 15/28 | 1.0 | 0 | 0.18 | 0 | 0 | 0.54 / 0.55 | 0.77 | 0.35 / 0.82 |
| rule（基线） | 1.461 | 15/28 | 1.0 | 0 | 0.21 | 2 | 0 | 0.44 / 0.45 | 0.67 | 0.47 / 0.69 |

### 3.2 抗噪（SNR 10 dB 持续背景噪声）

| 候选 | 吸尘器：综合 / 通过 / 误打断 / 漏判说完 | 洗衣机：综合 / 通过 / 误打断 / 漏判说完 |
|---|---|---|
| **fusion** | **0.31** / 26 / 0.05 / 0 | **0.57** / 22 / 0.20 / 0 |
| adaptive_ten | 0.36 / 26 / 0.11 / 0 | 0.69 / 25 / 0.00 / 12 |
| adaptive_silero | 0.46 / 25 / 0.15 / 0 | 0.59 / 22 / 0.25 / 0 |
| lm_eot | 0.42 / 25 / 0.05 / 0 | 0.67 / 21 / 0.20 / 0 |
| adaptive（能量 VAD） | 2.50 / **4** / — / 38 | 0.64 / 20 / 0.17 / 9 |
| rule（能量 VAD） | 2.62 / **4** / — / 38 | 3.70 / **0** / — / 41 |

能量 VAD 在吸尘器噪声下永远“有声”，几乎从不判说完。TEN VAD 对洗衣机噪声也有 17–25% 的帧判为语音，所以 `adaptive_ten` 在洗衣机条件下漏判说完 12 次。Silero 两种噪声都几乎 0 误报（离线：吸尘器、洗衣机在 SNR 5/10/20 dB 下有声帧比例均为 0.00）。

### 3.3 骨架门槛（base_wav，5 个原场景的真实语音版；以及原 `scenarios/*.toml`）

- 原 `python -m duplexagent eval scenarios/*.toml`（默认配置）每次推送前都 5/5 通过；`adaptive` 等纯 Python 候选换进去也 5/5。
- base_wav：`fusion`、`adaptive*`、`silero`、`ten_vad`、`rule` 均 5/5，**不退化**；`fusion` 的说完→出声 p50 0.52 s（基线 0.68 s），打断停止 0.21 s（基线 0.65 s）。
- 未达门槛（只列失败的检查项，原因未逐一深究）：`lm_eot` 4/5、`fusion_st` 4/5（delegate 的 `said_contains:1081`）；`smart_turn` 2/5（basic 的 `max_latency_p50`，cancel 的 `tasks_spawned`/`tasks_delivered`/`said_contains:已经取消`，delegate 的 `said_contains:1081`）；`ten_turn` 4/5（basic 的 `max_latency_p50`）。

### 3.4 文本 EOT 离线基准（100 句最小对）

| 打分器 | AUC | 准确率@0.5 | 未完被判完（P≥0.75） | 已完被判未完（P≤0.25） | 延迟 p50 |
|---|---|---|---|---|---|
| 文本规则 | 0.775 | 0.67 | 4% | 0% | <0.1 ms |
| Qwen2.5-0.5B（lm，log 校准） | 0.930 | 0.85 | 4% | 8% | 168 ms |
| Qwen2.5-1.5B（同法） | 0.893 | 0.85 | 4% | 14% | 314 ms |
| TEN Turn Detection 7B Q4 | **0.948** | **0.90** | 6% | 2% | 498 ms（独占 4 核） |
| 规则 + lm 平均（= fusion） | 0.939 | **0.92** | 4% | — | 168 ms |
| 规则 + TEN | 0.957 | 0.91 | 4% | — | 498 ms |

规则的问题是“没把握时一律给 0.7”：它几乎不会把未完句当成已完，但会对很多已完的陈述句多等几百毫秒。LM 补的正是这部分。1.5B 并不比 0.5B 好。零样本“这句话说完了吗？是/否”分类在 0.5B 上接近随机，已放弃。

## 4. 推荐方案与参数

**默认：`configs/exp/turn/fusion.toml`**

```toml
[turn]
impl = "fusion"
vad = "silero"            # vad_threshold 0.5，关闭阈值 0.35（迟滞）
eot = ["text", "lm"]
eot_weights = { text = 1.0, lm = 1.0 }
min_silence_ms = 200      # P(说完) ≥ 0.75 时
max_silence_ms = 2000     # P(说完) ≤ 0.15 时（“我想……”）
base_silence_ms = 500     # 打分器还在算时的下限
hangover_ms = 60          # Silero 自带平滑，不需要 100 ms
barge_in_ms = 300         # 有非附和文字时
barge_in_max_ms = 700     # 一个字都没识别出来时（Silero 不怕咳嗽，可以短）
# 默认开启：command_barge、echo_guard、lexicon_override、adapt
```

理由：
- 安静条件下与 `adaptive_silero` 并列第一档。
- 噪声条件下综合分最好，因为 LM 在 ASR 文本上补足了规则对“无标记陈述句”的犹豫。
- 骨架门槛 5/5。
- 依赖全部是宽松许可（MIT / Apache-2.0）。
- 代价：常驻约 500 MB 内存；每次停顿在后台线程跑约 170 ms，不阻塞音频循环。

**资源紧张时：`configs/exp/turn/adaptive_silero.toml`。** 安静环境综合分最好，只是噪声下比 fusion 稍差；只需 onnxruntime 和约 2 MB 的 Silero 模型。

**参数敏感点：**
- `min_silence_ms`：200–250 ms，再低抢话会上升。
- `max_silence_ms`：1.5–2 s，决定“我想……”能容忍多长的思考。
- `barge_in_max_ms`：配 Silero 用 700 ms；配能量或 TEN VAD 用约 1300 ms，否则咳嗽/敲击会误打断。

**不推荐：**
- `smart_turn`：在中文合成语音上把“我想…”判为说完（P=0.82），把“我今天有点累”判为没说完（0.12），抢话率 70%。真人中文录音可能更好，需补测。
- `ten_turn`：文本最准、抢话为 0，但 CPU 上 0.9 s/次使 TURN_END p50 达 1.44 s。若有 GPU（vLLM 服务，适配器已支持 `backend="openai"`），可作为 fusion 的第三路复测。
- 单独换 VAD（`silero` / `ten_vad`）：不解决抢话，只解决噪声。

## 5. 风险与局限

1. **合成语音偏差**：测试语音来自 Piper TTS，韵律比真人规整；“思考停顿”是在整句合成音频上切开再插静音的，没有真人的拖音和“呃——”。Smart Turn 这类声学模型的结论尤其需要用真人录音补测（建议录 50–100 条自有数据）。
2. **ASR 是 mock**：文本无错字、按真值字符均匀出现。真实 ASR 会有错字和标点差异，partial 也会回改。`lagged_mock` 只模拟了延迟。文本规则依赖尾词，ASR 把“我想”识别成“我像”时会失效；可以用 `asr.error_rate` 或 ASR 方向的插件补测。
3. **背景人声 / 电视**：4 人混叠人声即使在 20 dB SNR 下，Silero 和 TEN VAD 也有 98–99% 的帧判为语音，能量 VAD 在 10 dB 下也有 69%。这个条件已从评测中删除（任何 VAD 都会正确地认为“有人在说话”），需要说话人相关的 VAD 或声纹/方向信息，留作风险。
4. **回声**：没有 AEC 时助手声音漏进麦克风。现在只有文本级的 echo_guard（用户文字是助手正在说的话的片段就忽略），它依赖 ASR 先识别出回声内容，没有声学层防护。
5. **附和语在噪声中**：洗衣机噪声下 `fusion` 仍有 20% 的附和被当打断，主要是“嗯/哦”这类单字在噪声里被 VAD 拖长、超过 300 ms 后文字还没到。可考虑附和语专门的声学分类，或 §6 的 ducking。
6. **难例未解**：“我今天有点累……而且还有点头疼”前半句语义完整，所有文本模型都会抢话（只有 TEN 7B 没有）。这是延迟和抢话之间的固有取舍。
7. **规则过拟合**：中文词表是对着场景写的。在没调过的离线基准上规则只有 AUC 0.775，说明它在新说法上会退化到“多等一会儿”（安全方向），LM 融合能弥补。
8. **许可证**：
   - TEN VAD / TEN Turn Detection 是 Apache-2.0 附加条款，禁止用于与 Agora 竞争的产品。
   - LiveKit Turn Detector 不可用于非 LiveKit 框架。
   - Piper 是 GPL-3.0，只在离线造数据时使用。
   - ESC-50 是 CC BY-NC 3.0，只用于评测、不入库。
   - TEN VAD 在 Linux 上需要系统包 `libc++1`。

## 6. 接口变更请求

1. **已提交 `core:` commit `5fb19d0`：引擎时钟在所有插件 `start()` 完成后才开始。**
   - 原因：原先 `_t0` 在 `start()` 之前设定，插件加载模型（这里约 3.8 s）会让所有时间线事件相对模拟输入整体偏移，`metrics.py` 和我的打分都会误判延迟和打断窗口。
   - 影响：任何在 `start()` 里加载模型的方向（ASR/TTS 同样）都受益；接口不变，对瞬时启动的插件零影响。
2. **（请求，未改）`TurnContext` 增加 `agent_played_text`（已实际播出的文字）和 `last_agent_message`（助手上一条完整消息）。**
   - 现在 `agent_text` 是快脑已生成的全部文字，说完即清空：回声判断会把“还没播的字”也当成可能的回声，上下文感知模型拿不到助手刚问过什么（目前在插件里自行缓存）。
3. **（请求）新增 `TurnEventType.BARGE_CANDIDATE` / `DUCK`。** 引擎收到后先压低助手音量，确认后再停（docs/01 §1.4 的 ducking）。这样可以把 `barge_in_ms` 降到 100–150 ms 而不增加附和误打断。
4. **（请求）`AudioFrame` 或 `TurnContext` 带上当前播放的参考信号**（TTS PCM），便于做声学层回声判别或接 AEC。
5. **（建议给 ASR 方向）骨架 `asr=mock` 在 `wav = ...` 场景下会把首个真值字落在前导静音帧上并立即吐出单字 final**（静音计数从上一句延续）。`experiments/turn/lagged_asr.py` 里的修复（只在有声帧上放出字、听到声音后才断句）可以并回 `plugins/asr/mock.py`。
6. **（建议）sim 输入支持持续背景噪声和重叠音频。** 目前在 `experiments/turn/noisy_sim.py` 里以子类实现，通过 `frame.clean_pcm` 给 mock ASR 传干净信号。

## 7. 复现

```bash
apt-get install -y libc++1                                   # 仅 TEN VAD 需要
pip install -r experiments/turn/requirements.txt
pip install --no-deps silero-vad g2pW
pip install llama-cpp-python                                 # 仅 ten_turn
python experiments/turn/prepare_data.py                      # 生成 wav、场景、背景噪声（~30 s）
python experiments/turn/run_suite.py                         # 全部条件×候选，约 2.5 h，写 results/ 与 leaderboard.md
python experiments/turn/run_suite.py --cond wav_lagged --config fusion --config rule
python experiments/turn/bench_text_eot.py --scorer text --scorer lm --scorer ten_turn
python experiments/turn/make_configs.py                      # 由 configs/default.toml 重新生成 configs/exp/turn/*.toml
python -m pytest -q experiments/turn/tests                   # 检测器单元测试（标准库部分）
```

模型下载（首次运行自动下载到 `models/`）：
- `pipecat-ai/smart-turn-v3`：`smart-turn-v3.2-cpu.onnx`
- `onnx-community/Qwen2.5-0.5B-Instruct`：`onnx/model_quantized.onnx` 与 tokenizer
- `sheldonrobinson/TEN_Turn_Detection-Q4_K_M-GGUF`
- `rhasspy/piper-voices`：zh_CN 三个音色

## 8. 硬件环境

4 核 Intel Xeon @ 2.10 GHz，15 GB 内存，无 GPU；Python 3.13，onnxruntime 1.29，numpy 2.5，llama-cpp-python 0.3.36（源码编译）。

所有延迟都是在线实时测得（模拟用户按 20 ms 帧实时推送）。轻量候选 8 个场景并发，模型类 3 个并发，ten_turn 单独跑，以避免 CPU 争用污染延迟。
