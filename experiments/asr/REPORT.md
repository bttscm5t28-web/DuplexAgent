# ASR 实验报告（流式中文 / 中英混说）

分支 `claude/exp-asr`。目标：在骨架里实现并对比多个中文流式 ASR 插件，选出最适合全双工对话的方案。

## 结论先行

**推荐：`sherpa_offline` + SenseVoice-Small（`configs/exp/asr/sensevoice.toml`）**——Silero VAD 分段 + 每 300 ms 重解码出 partial 的伪流式。

- 唯一在 **5/5 真实音频场景**全部通过、同时**中英混说最好之一**（ASCEND MER 11.9%）的方案；
- **首个 partial 最快**（说话开始后 0.20 s，p50），这是打断判别（`barge_in_ms=300` 内要拿到文字）的关键；
- 说完到 final 0.62 s（p50），落在轮次检测的等待窗口内（eot 500 ms + final_wait 400 ms）；
- **CPU 只占 0.7 核、内存 380 MB、RTF 0.043**，4 核机器上给快脑 / TTS 留足余量；纯 ONNX，无 torch 依赖。

**备选：**

| 用途 | 方案 | 理由 |
|---|---|---|
| 纯中文、要最低延迟 | `paraformer_vad`（Paraformer-large 离线 + VAD） | 5/5 场景通过，final 0.57 s、首 partial 0.20 s，RTF 0.036 最省；但中英混说（MER 25.9%）和强噪声明显差 |
| 真流式、最低 CPU | `sherpa_zipformer_ctc_zh`（流式 Zipformer-CTC） | AISHELL 2.5%、final 0.53 s、0.34 核；但首 partial 0.56 s（打断场景超时）、中英混说差 |
| 有 GPU 时追求准确率 | `firered_aed2` / `funasr_nano` 做二遍（final）解码 | 准确率最高（Wenet 5.6% / 7.3%），附和词识别最好；CPU 上 final 延迟 1.4–2.7 s 不可用，**待 GPU 补测** |

不推荐：faster-whisper 全系（CPU 上 CER 10–13%、繁体/幻觉、RTF ≥ 0.35，final 3–10 s）；FunASR 原生 torch 流式（与 sherpa 导出的同一模型相比内存 3.2 GB、首 partial 1.0 s，且噪声下差）。

## 硬件与软件环境

- 4 × Intel Xeon @ 2.30 GHz（无超线程），15 GB 内存，无 GPU；Python 3.13。
- sherpa-onnx 1.13.8 / onnxruntime 1.29.0 / faster-whisper 1.2.1 / funasr 1.4.16 + torch 2.14.1 CPU。
- 所有候选 `num_threads = 2`；评测串行执行（同一时刻只有一个模型在跑），实时延迟不受争抢影响。

## 候选与插件

| 配置 `configs/exp/asr/*.toml` | 插件 `impl` | 模型（均为 Hugging Face 上的 int8 ONNX，除注明外） | 方式 |
|---|---|---|---|
| sherpa_zipformer_zh | sherpa_online | streaming-zipformer-zh 2025-06-30 | 真流式 transducer |
| sherpa_zipformer_zh_xlarge | sherpa_online | streaming-zipformer-zh-xlarge 2025-06-30 | 真流式 transducer |
| sherpa_zipformer_ctc_zh | sherpa_online | streaming-zipformer-ctc-zh 2025-06-30 | 真流式 CTC |
| sherpa_zipformer_bilingual | sherpa_online | streaming-zipformer-bilingual-zh-en 2023-02-20 | 真流式 transducer |
| sherpa_paraformer_streaming | sherpa_online | FunASR paraformer-zh-streaming（中英，sherpa 导出） | 真流式（chunk 600 ms） |
| funasr_streaming | funasr_streaming | FunASR paraformer-zh-streaming 原生 PyTorch | 真流式 |
| sensevoice / sensevoice_2025 | sherpa_offline | SenseVoice-Small 2024-07-17 / 2025-09-09 | VAD 伪流式 |
| paraformer_vad | sherpa_offline | Paraformer-large zh 2025-10-07 | VAD 伪流式 |
| firered_aed / firered_aed2 / firered_ctc2 | sherpa_offline | FireRedASR-AED-L v1 / FireRedASR2-AED / FireRedASR2-CTC | VAD 伪流式 |
| qwen3_asr | sherpa_offline | Qwen3-ASR-0.6B | VAD 伪流式（LLM-ASR） |
| funasr_nano | sherpa_offline | Fun-ASR-Nano 2025-12（SenseVoice 编码器 + Qwen3-0.6B） | VAD 伪流式（LLM-ASR） |
| faster_whisper_small / medium / turbo | faster_whisper | Whisper small / medium / large-v3-turbo（CTranslate2 int8） | VAD 伪流式 |

插件实现要点（`duplexagent/plugins/asr/`）：

- **推理在独立工作线程**（`_util.Worker`），永远不阻塞引擎事件循环；同进程内同一模型只加载一次（`eval` 并发跑多个场景时共享）。
- **`sherpa_online`**：真流式模型。端点默认 `endpoint="vad"`：Silero VAD 判到 300 ms 静音后给流补 0.66 s 尾部静音、`input_finished()`、取最终结果、换新流。sherpa 自带的「尾部 blank」端点（`endpoint="model"`）也保留，但实测更差，见下文。
- **`sherpa_offline` / `faster_whisper`**（共用基类 `VadSegmentASR`）：Silero VAD 分段；段内每 `partial_ms` 新音频且工作线程空闲时重解码整段出 partial（慢模型自动退化为更少的 partial，而不会越积越多）；段结束时整段解码出 final。
- **`funasr_streaming`**：FunASR `AutoModel.generate(cache=…, is_final=…)` 原生流式，段边界同样由 Silero VAD 决定。
- **ITN**：配置默认 `itn = true`（「二十三乘以四十七」→「23乘以47」，否则慢脑 mock 算不出 1081）；SenseVoice / Fun-ASR-Nano 用模型自带 ITN，其他用 `_util.itn_zh`（保守的中文数字转写，不动「一下 / 一二线 / 十分 / 万一」）。CER 评测时统一关掉（参考文本里是汉字数字）。
- 输出统一清洗：去掉模型标签、CJK 之间的空格、假名/韩文（多语模型偶尔把「嗯嗯」识别成「嗯嗯う」）。

模型权重不入库，首次使用时自动从 Hugging Face 下载到 `models/asr/`（Silero VAD 到 `models/vad/`）；`DUPLEXAGENT_MODELS` 环境变量可改根目录。依赖见 `experiments/asr/requirements.txt`。

## 评测数据（`bench_data/asr/*.jsonl`，wav 由脚本生成、不入库）

`python experiments/asr/prepare_data.py` 一键下载并生成（`--scenarios` 生成场景音频）。所有 wav 16 kHz 单声道，峰值归一化到 0.7（相当于麦克风 AGC），首尾各留约 300 ms 静音。

| 集合 | 来源 / 许可 | 规模 | 特点 |
|---|---|---|---|
| aishell | AISHELL-1 test（HF `carlot/AIShell` 转存；Apache-2.0） | 200 句 15.6 min | 朗读、安静 |
| aishell_rt | aishell 的 1/4 子集 | 50 句 3.8 min | 实时延迟测试用 |
| wenet | WenetSpeech test_net（HF `TwinkStart/WenetSpeech`；**仅限非商业研究**） | 100 句 6.2 min | 网络视频、口语 |
| ascend | CAiRE/ASCEND test，language=mixed（CC BY-SA 4.0） | 100 句 7.4 min | 中英混说、口语 |
| short | ASCEND test 中 < 1.5 s 的短句（78 条附和词 + 215 条短实词） | 293 句 3.9 min | 附和词 / 短回答 |
| noisy_{white,babble}{10,0} | aishell 子集加白噪声 / 4 人 babble，SNR 10 dB、0 dB | 各 50 句 | 噪声鲁棒 |

## 指标（`experiments/asr/bench.py`，官方 `bench asr` 的超集）

- **CER**：与 `bench asr` 同一归一化 / 编辑距离；ASCEND 用 **MER**（汉字按字、英文按词）。
- **首 partial 延迟**：实时送入时，第一个识别结果相对「开始说话」（能量检测的语音起点）的时间。
- **final 延迟**：实时送入时，最后一个 final 相对「真正说完」（能量检测的语音终点）的时间。官方 `bench asr --realtime` 相对 wav 结尾，且把句中断句产生的每个 final 都算进去（会出现负值），所以这里只取最后一个 final。
- **RTF**：非实时送入，只计 final 解码（伪流式关掉 partial）；partial 重解码的真实开销体现在**实时 CPU 核数**里。
- **附和保留 / 短句丢失**：附和词（嗯 / 对 / 好的 / yeah…）被识别成 `turn.rule` 认得的附和词的比例（否则引擎误打断）；短实词（然后 / 没有 / 不是…）被识别为空或附和词的比例（用户的短回答被吞）。
- **场景**：`scenarios/asr/*.toml`——与 `scenarios/*.toml` 完全相同的 5 个场景，每句话换成 TTS（sherpa-onnx VITS 中文）合成的 wav，后续轮次顺延以保持原有间隔；在真实 ASR + `turn.rule` + 其余 mock 模块下端到端跑分。

## 排行榜

（`python experiments/asr/matrix.py summary` 生成，原始结果在 `experiments/asr/results/<候选>/`；按 AISHELL CER 排序）

| 候选 | AISHELL CER% | Wenet CER% | ASCEND MER% | 噪声均值 CER% | 白噪10/0 | babble10/0 | 附和保留% / 短句丢失% | RTF | 首 partial p50 (s) | final 延迟 p50/p90 (s) | CPU 核 | 内存 MB | 场景通过 | 场景响应延迟 p50 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| firered_aed ¹ | **0.8** | 7.8 | 14.7 | 31.1 | 4.4/36.1 | 2.0/81.7 | 77.3/3.6 | 1.373 | 1.28 | 9.26/18.93 | 1.89 | 2925 | 1/5 | 1.13 |
| firered_aed2 | 1.7 | **5.6** | **10.9** | 28.4 | 4.5/34.9 | 4.5/69.7 | 85.9/6.0 | 0.315 | 0.40 | 2.72/4.29 | 1.74 | 1701 | 1/5 | 1.13 |
| funasr_nano | 2.3 | 7.3 | 13.4 | 29.6 | 6.2/38.4 | 3.5/70.5 | 88.5/3.3 | 0.148 | 0.34 | 1.36/2.31 | 1.34 | 1745 | 3/5 | 1.13 |
| sherpa_zipformer_zh_xlarge | 2.4 | 12.2 | 25.5 | 29.9 | 7.5/34.6 | 8.2/69.5 | 74.4/9.3 | 0.312 | 0.63 | 0.65/0.80 | 0.82 | 1037 | 4/5 | 0.89 |
| sherpa_zipformer_ctc_zh | 2.5 | 9.6 | 29.0 | 32.5 | 8.6/39.6 | 9.2/72.6 | 69.2/6.5 | 0.122 | 0.56 | 0.53/0.67 | 0.34 | 312 | 4/5 | 0.79 |
| sherpa_zipformer_zh | 2.9 | 12.3 | 27.6 | 34.2 | 9.0/40.8 | 10.8/76.1 | 73.1/8.4 | 0.133 | 0.56 | 0.53/0.64 | 0.47 | 309 | 3/5 | 0.79 |
| sensevoice_2025 | 3.2 | 12.0 | 14.8 | 32.5 | 8.2/47.2 | 4.1/70.3 | 80.8/7.0 | 0.044 | **0.20** | 0.62/0.90 | 0.71 | 375 | 4/5 | 0.89 |
| funasr_streaming | 3.3 | 10.7 | 19.6 | 42.0 | 7.8/42.8 | 15.1/102.5 | 65.4/5.6 | 0.143 | 1.00 | **0.51**/0.62 | 0.60 | 3187 | 3/5 | 0.79 |
| sherpa_zipformer_bilingual | 3.5 | 10.5 | 25.4 | 37.7 | 10.8/56.1 | 9.8/74.2 | 59.0/17.7 | 0.071 | 0.78 | **0.51**/0.61 | 0.39 | 307 | 3/5 | 0.75 |
| **sensevoice** ★ | 3.6 | 8.4 | 11.9 | **27.9** | 6.4/42.2 | 3.6/59.2 | 75.6/6.5 ² | 0.043 | **0.20** | 0.62/0.90 | 0.69 | 377 | **5/5** | 0.87 |
| firered_ctc2 | 4.3 | 10.2 | 15.4 | 30.9 | 9.1/43.7 | 7.9/63.0 | 75.6/5.6 | 0.209 | 0.34 | 1.94/2.98 | 1.73 | 984 | 2/5 | 1.13 |
| paraformer_vad | 4.7 | 11.9 | 25.9 | 38.3 | 8.3/60.7 | 7.5/76.9 | 80.8/7.9 | **0.036** | **0.20** | 0.57/0.76 | 0.56 | 355 | **5/5** | 0.81 |
| sherpa_paraformer_streaming | 5.2 | 12.1 | 20.3 | 42.3 | 10.8/34.9 | 25.0/98.4 | 71.8/7.0 | 0.100 | 0.40 | **0.51**/0.61 | **0.28** | 364 | 3/5 | 0.77 |
| qwen3_asr | 5.3 | 9.8 | 15.3 | 32.5 | 5.8/39.4 | 5.0/80.0 | **89.7**/6.5 | 0.246 | 0.66 | 2.20/3.38 | 1.80 | 1730 | 2/5 | 1.13 |
| faster_whisper_turbo ¹ | 10.1 | 19.4 | 26.1 | — ³ | — | — | 95.5/10.7 | 1.249 | 12.72 | 10.17/20.68 | 1.89 | 2112 | 0/5 | — |
| faster_whisper_medium ¹ | 10.2 | 21.2 | 54.7 | 94.4 | 20.7/103.9 | 13.5/239.6 | 86.4/7.1 | 0.857 | 8.49 | 9.38/15.39 | 1.91 | 2096 | 0/5 | — |
| faster_whisper_small | 13.0 | 30.8 | 39.4 | 78.2 | 27.9/81.3 | 21.3/182.5 | 84.6/8.8 | 0.350 | 1.68 | 3.08/3.94 | 1.94 | 857 | 2/5 | 1.13 |

¹ 太慢，每个数据集只跑前 50 句。² 加入假名/韩文过滤后重测（过滤前 70.5/2.8：原先被识别成假名乱码的短句现在变成空，计入「丢失」）。³ turbo RTF > 1，跳过噪声集。
「场景响应延迟」= 用户说完到 agent 开口；1.13 s 意味着 ASR final 没赶上 `turn.rule` 的等待窗口（500 + 400 ms），引擎只能拿 partial 文本兜底。

### 参数扫描（实时，aishell_rt）

| 变体 | 首 partial p50 | final 延迟 p50/p90 | finals/句 | CPU 核 | 结论 |
|---|---|---|---|---|---|
| sensevoice（endpoint 300 ms, partial 300 ms） | 0.20 | 0.62/0.90 | 1.30 | 0.69 | 默认 |
| sensevoice endpoint 200 ms | 0.22 | 0.56/0.72 | 1.40 | 0.68 | 快 60 ms，但断句更碎；轮次检测负责合并，可按需选 |
| sensevoice endpoint 500 ms | 0.22 | 0.88/1.05 | 1.14 | 0.80 | 太慢 |
| sensevoice partial 150 ms | 0.22 | 0.64/0.89 | 1.30 | 1.10 | CPU +60%，首 partial 不变（首 partial 受 VAD 起点限制），不值 |
| zipformer_zh endpoint="model"（sherpa 自带, 400 ms） | 0.56 | 0.94/1.24 | 1.84 | 0.45 | CER 2.6%（VAD 端点 2.9%）略好，但 final 慢 0.4 s、句子被切得更碎（「你\|好」） |
| firered_aed2 关 partial | — | 1.91/3.05 | 1.30 | 0.60 | 排队等 partial 只占一部分，整段解码本身就要 ~2 s |
| funasr_nano 关 partial | — | 1.14/1.51 | 1.30 | 0.32 | 同上 |

## 分析

1. **准确率 vs 延迟的分界很清楚。** LLM-ASR / 大 AED（FireRedASR2、Fun-ASR-Nano、Qwen3-ASR）在 Wenet、ASCEND、附和词上都明显更准，但在 4 核 CPU 上整段解码要 1–2 s，final 普遍晚于轮次检测的等待窗口，场景只过 1–3 个。它们是 GPU 部署时的二遍解码候选，不是 CPU 上的主 ASR。
2. **全双工最看重「首 partial」，不是 final。** `turn.rule` 要在用户开口 300 ms 内拿到非附和文字才判打断，否则要等到 700 ms；所有首 partial ≥ 0.4 s 的方案都在 `barge_in`（`max_barge_in_stop_latency=0.6`）场景失败。真流式 transducer 受 chunk + 右上下文所限（0.56–0.78 s），反而不如「VAD 一起点就立刻解码一次」的伪流式（0.20 s）。
3. **SenseVoice 的伪流式代价可接受。** 段内反复重解码是 O(n²)，但 SenseVoice 非自回归、一次解码几十毫秒，实时 CPU 只有 0.7 核；`max_segment_s=20` 兜底。
4. **端点用 VAD，别用流式模型自带的。** sherpa 的「尾部 blank」端点会被 transducer 的延迟出字骗到（字间也有一串 blank），既把一句话切开（1.84 finals/句），final 又慢 0.4 s；流式 Paraformer 没有 blank，自带端点直接在词中触发（CER 31%）。VAD + 冲刷让所有流式模型的 final 都落在 0.5–0.65 s。
5. **中英混说**：SenseVoice（11.9%）、FireRedASR2（10.9%）、Fun-ASR-Nano（13.4%）远好于纯中文的 zipformer / Paraformer（25–29%），后者把英文词音译成汉字。
6. **噪声**：10 dB 下多数模型仍可用（3.6–11%），0 dB 全部崩溃（35–100%）；babble 0 dB 下 Paraformer 系最差（流式 98%，会把背景人声当成用户在说话）。SenseVoice 噪声均值最好（27.9%）。
7. **附和词**：Qwen3-ASR / Fun-ASR-Nano 最好（≈ 89%）；SenseVoice 2024 版的漏识主要是把「嗯」输出成 um / hm / oh，或误判语种（已加过滤）；「对」单字常被丢成空（各模型都有）。
8. **SenseVoice 2024 vs 2025**：2025 版 AISHELL 更好（3.2 vs 3.6）、附和词更好（80.8 vs 75.6），但 Wenet（12.0 vs 8.4）和中英混说（14.8 vs 11.9）明显退步，场景 basic 也没通过（第一句没得到预期的寒暄回复）。综合仍推荐 2024 版。

## 风险

- **场景音频是 TTS 合成的**（单一音色、干净、语速匀），比真人好识别；附和「嗯」等短音 TTS 也不自然。真人录音 + 回声场景还需要补测（需要 AEC，属于 turn / io 方向）。
- **AISHELL 可能被大模型训练用过**（FireRedASR、Fun-ASR 报告都以 AISHELL 为主测试集），AISHELL CER 偏乐观；Wenet / ASCEND 更能说明问题。
- **VAD 断句**：300 ms 静音就断，句中停顿会把一句话切成多个 final（约 1.3 个/句）。引擎侧 `turn.rule` 会把同一轮的多个 final 拼起来，但 ASR 看不到跨段上下文，长句准确率会略降；想要更长上下文可调大 `endpoint_ms`，代价是延迟。
- **Silero VAD 在强噪声下**会把背景人声当作语音，partial 不停刷新，CPU 上升；这时应由打断判别（turn 方向）结合附和词表 / 语义来过滤。
- **伪流式的 partial 不单调**：每次整段重解码，前面已出的字可能被改写。依赖 partial 做早响应（如快脑预热）时要注意。
- **许可证**：WenetSpeech 仅限非商业研究；ASCEND 是 CC BY-SA 4.0（我们只存了文本与派生 jsonl）；SenseVoice 模型许可是 FunASR 模型许可证（可商用，需署名），FireRedASR / Qwen3-ASR 是 Apache-2.0；商用前需再核对。
- **GPU 待补测**：FireRedASR v1/v2、Fun-ASR-Nano、Qwen3-ASR、Whisper large-v3-turbo 的结论只代表 CPU。onnxruntime-gpu / CUDA provider 只需把配置里 `provider = "cuda"`，插件已支持。

## 接口变更请求

1. **已做（向后兼容，单独 commit `core: restart the session clock after plugins have started`）**：`DuplexEngine.run()` 原来在插件 `start()` 之前就开始计时。真实 ASR 加载模型要 1–10 s，场景里每个事件都被整体推迟这么久，而 `metrics.score` 用场景自己的时间判打断窗口和停止延迟，导致打断类指标失真。改为所有插件启动完后重置 `_t0`。mock 插件启动是瞬时的，基线结果不变（`pytest` 和 `eval scenarios/*.toml` 都通过）。
2. **建议（未改，属于 `duplexagent/bench/`）**：`bench_asr --realtime` 的 `final_after_speech_end_s` 把每个 final 都算进去（句中断句会产生负值），建议只取每句最后一个 final；`first_partial_s` 从 wav 开头计时，包含了句首静音，建议相对语音起点计时。`experiments/asr/bench.py` 已按这两种方式实现。
3. **建议给 turn 方向**：`turn.rule` 的附和词表补充 `um / hm / oh / 嗯哼 / 对呀 / 是的 / 对的`；真实 ASR 输出里这些都很常见（见 `results/*/short.json`）。
4. **可选**：`Transcript` 增加可选字段 `lang`（SenseVoice / Whisper 都会给出），便于快脑决定用中文还是英文回复；默认 `None`，向后兼容。本分支没有改。

## 复现

```bash
pip install -r experiments/asr/requirements.txt            # torch 用 CPU 源：--index-url https://download.pytorch.org/whl/cpu
python experiments/asr/prepare_data.py                     # 评测数据（约 1 GB parquet 缓存在 models/data/）
python experiments/asr/prepare_data.py --scenarios         # scenarios/asr/wav/*.wav
python experiments/asr/matrix.py run --rt                  # 全部候选 × 全部数据集（4 核上约 6 小时；慢模型加 --limit 50）
python experiments/asr/matrix.py scenarios                 # 端到端场景
python experiments/asr/matrix.py summary                   # 排行榜 -> results/leaderboard.md
# 单独换进骨架：
python -m duplexagent eval scenarios/asr/*.toml -c configs/exp/asr/sensevoice.toml --serial
python -m duplexagent bench asr --data bench_data/asr/aishell_rt.jsonl -c configs/exp/asr/sensevoice.toml --set asr.itn=false --realtime
python -m pytest -q experiments/asr/tests
```

注意：带真实模型跑 `eval` 时加 `--serial`，否则 5 个场景同时抢 CPU，延迟不具代表性。TTS 合成有随机性，重新生成场景音频后个别字的识别结果可能不同。
