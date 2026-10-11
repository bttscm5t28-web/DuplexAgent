# TTS 方向实验报告

> 分支 `claude/exp-tts`。本文件中所有数字均来自 `experiments/tts/results/*.json`，可用下文命令复现。

## 0. 排行榜（60 句，`eval_tts.py --cps 20`，按 CER 排序）

| 候选 | n | 首包 p50 / p90 (s) | RTF p50 | CER | UTMOS | 对齐命中 | 截断误差(字) | 流式首包 p50 (s) | 卡顿次数 | 取消→空闲 p90 (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| edge | 60 | 0.426 / 0.628 | 0.202 | 0.0125 | 3.50 | 0.982 | 0.57 | 0.817 | 9 | 0.001 |
| cosyvoice2-http | 10 | 11.337 / 15.675 | 6.789 | 0.0135 | 2.96 | 0.959 | 0.29 | – | – | 0.000 |
| matcha-zh-en | 60 | 0.131 / 0.176 | 0.060 | 0.0319 | 2.56 | 0.956 | 0.52 | 0.392 | 0 | 0.001 |
| zipvoice | 60 | 1.181 / 1.426 | 0.558 | 0.0360 | 1.61 | 0.863 | 0.63 | 1.233 | 43 | 0.000 |
| kokoro | 60 | 0.893 / 1.161 | 0.373 | 0.0499 | 2.91 | 0.984 | 0.46 | 0.863 | 29 | 0.646 |
| kokoro-int8 | 60 | 2.480 / 3.152 | 1.042 | 0.0540 | 2.12 | 0.983 | 0.47 | 1.873 | 58 | 2.783 |
| matcha-zh-baker | 60 | 0.167 / 0.217 | 0.072 | 0.0713 | 3.08 | 0.934 | 0.57 | 0.407 | 1 | 0.114 |
| melo-zh-en | 60 | 0.515 / 0.663 | 0.279 | 0.0831 | 2.64 | 0.949 | 0.54 | 0.602 | 25 | 0.000 |
| piper-huayan | 60 | 0.157 / 0.241 | 0.071 | 0.0949 | 3.66 | 0.800 | 0.70 | 0.396 | 2 | 0.001 |
| vits-aishell3 | 60 | 0.133 / 0.186 | 0.042 | 0.2597 | 1.83 | 0.841 | 0.61 | 0.385 | 0 | 0.625 |

- 首包/RTF：文本瞬时给完（同 `bench tts`）；**流式首包/卡顿**：按 20 字/秒喂文本并实时播放。
- `cosyvoice2-http` 只跑了 10 句短句（CPU 上每句 7–16 s），不能与其它行直接比较；`kokoro-int8` 为第一轮结果（第二轮未重跑，结论不受影响）。
- `results/round1/` 是第一轮结果（之后收紧了停顿锚点、加了限幅器、ZipVoice 换成干净的提示音频、MeloTTS 改 4 线程），仅供对照。
- CER 有约 ±0.5 个百分点的波动（Matcha/VITS 采样带噪声，两轮 matcha-zh-en 分别为 2.8% / 3.2%）。

## 1. 结论速览

**推荐：CPU 部署默认用 `tts.impl = "sherpa"`, `preset = "matcha-zh-en"`**（`configs/exp/tts/matcha-zh-en.toml`）。

- 唯一同时满足"首包 p50 ≈ 130 ms（p90 176 ms）、RTF ≈ 0.06（2 线程）、本地 CER 最低（3.2%）、中英混读可用、对齐命中 96%"的本地方案；端到端 5 个场景全过，响应延迟只比 mock 基线多 0.09–0.17 s。
- 短板：16 kHz 声码器，UTMOS 2.56（偏机械）；**模型来源（dengcunqin/matcha_tts_zh_en）未声明许可证，商用前必须确认**（见风险）。
- 第二梯队：
  - `matcha-zh-baker`：同样快（首包 0.17 s），音色更好（UTMOS 3.08），纯中文 CER ≈ 1–2%；但英文只能用近音字硬读（mixed CER 33%），且标贝数据仅限非商用。适合"纯中文、追求音色"的演示。
  - `edge`（微软在线音色，免 Key）：可懂度和自然度最好（CER 1.25%、UTMOS 3.50）、对齐用服务端词边界（命中 98%），但首包 0.43–0.56 s、端到端延迟 1.1–1.6 s、非官方接口无 SLA，只适合作对照/演示。
  - `kokoro`（Apache-2.0，许可最干净）：CER 5%、UTMOS 2.91、对齐最好，但 CPU 上 RTF 0.37、首包 0.9 s，20 字/秒流式时 29 次卡顿，端到端 3/5 场景失败——**需要更强 CPU 或 GPU**。
- GPU 方向：`CosyVoice 2`（Apache-2.0）通过 `openai_speech` + `servers/cosyvoice_server.py` 已跑通：10 句短句 CER 1.35%、UTMOS 2.96，但 CPU 上 RTF ≈ 5–7、首包 7–16 s，**待 GPU 补测**（官方称 GPU 流式首包约 150 ms）。云端 CosyVoice（`dashscope`）待 API Key 补测。
- 不推荐：`vits-aishell3`（CER 26%）、`piper-huayan`（UTMOS 最高 3.66 但 CER 9.5%、不会读英文、停顿与标点不对应导致对齐差）、`melo-zh-en`（CPU 上 RTF 0.28、CER 8.3%）、`zipvoice`（int8 蒸馏版 RTF 0.56、UTMOS 1.6）、`kokoro-int8`（int8 在此 CPU 上比 fp32 慢 3 倍）。

## 2. 硬件与环境

- 4 核 CPU（无 GPU）、15 GB 内存，Python 3.13，onnxruntime 1.29、sherpa-onnx 1.13.8。
- 所有本地候选都在 CPU 上实测；每个候选单独顺序跑（不并发），ORT 线程数见各自配置（Matcha/VITS/Piper 2 线程，MeloTTS/Kokoro/ZipVoice 4 线程）。评测之后对归一化做了两处小修（`23 * 47` 的 `*` 读作"乘"、中文之间的空格不再送给模型），不改变 60 句基准的输入。
- 无 API Key：DashScope CosyVoice 适配器未能实测；OpenAI 兼容适配器用本地桩服务（单测）和本地 CosyVoice 2 服务验证过，OpenAI 官方服务待补测；edge-tts 是免 Key 的在线服务，已实测。

## 3. 实现概览（`duplexagent/plugins/tts/`）

| 文件 | 内容 |
|---|---|
| `textnorm.py` | 中文文本归一化，**保留与原文的映射**：每个"对齐单元"记录原文切片 `orig`、送给模型的读法 `spoken`、预计时长权重 `weight`。覆盖整数/小数/负数/千分位、百分比与区间（`10-20%`）、日期/年份/时刻、手机号与长数字串（逐位读，"1"读"幺"）、货币（`¥3.5元` 不重复"元"）、单位（km/h、ml、℃…）、分数、"两/二"、`#号`、英文三种模式（`keep` 双语模型直接读、`spell` 缩写逐字母、`zh` 用中文近音字给纯中文模型）。 |
| `streaming.py` | 所有真实后端共用的流式基类 `StreamingTTS`：增量分段器 `Segmenter`、按段合成、字级对齐切块、重采样到 16 kHz、模型进程内共享缓存、取消。后端只需实现 `_load()` 和 `_synth(text, stop)`（或异步的 `_synth_async`），可选返回时间标记 marks 以获得精确对齐。 |
| `sherpa.py` | `impl="sherpa"`：sherpa-onnx 的 8 个预设（Matcha zh-en / zh-baker、MeloTTS、VITS AISHELL-3、Piper huayan、Kokoro v1.1-zh fp32/int8、ZipVoice）。 |
| `edge.py` | `impl="edge"`：微软 Edge 在线神经音色（edge-tts，免 Key），使用服务端返回的 WordBoundary 做精确对齐。 |
| `openai_speech.py` | `impl="openai_speech"`：OpenAI 兼容 `/v1/audio/speech`（纯标准库 HTTP）。GPU 级大模型（CosyVoice、Fish-Speech、IndexTTS、F5-TTS 等）以独立服务形式接入，不把它们的依赖拉进引擎进程。 |
| `dashscope.py` | `impl="dashscope"`：阿里云百炼 CosyVoice 云端（dashscope SDK），待补测。 |

配套：`experiments/tts/servers/cosyvoice_server.py`（CosyVoice 2/3 的 OpenAI 兼容最小服务）、`eval_tts.py`（评测）、`download.py`（下载模型）、`run_all.sh`/`sweep.sh`（批量评测/消融）、`leaderboard.py`（生成排行榜）、`test_tts.py`（单测，无需模型）。

### 3.1 流式分段策略（首包 vs 韵律）

快脑是逐小段（几个字）给文本的。`Segmenter` 的规则：

1. **首段要短**：遇到第一个标点且已累计 ≥ `first_min_chars`（默认 4）个字就切；超过 `first_max_chars`（默认 14）仍无标点，就在"软断点"（的/了/是/在/和… 等虚词之后或空格）处切。
2. **后续段要长**：在 `max_chars`（默认 60）以内取**最后一个句末标点**；没有句末标点时，只有当分句已经足够长（≥ 2×`min_chars`）或缓冲超过上限才在逗号处切——尽量整句合成，韵律更自然。
3. **截止时间驱动的"紧急切分"**：记录已交给播放端的音频何时播完（`horizon`）。如果在播完前 `lead_ms`（默认 250 ms）还没有可切的段（快脑卡住了），就取当前最长的安全前缀立即合成，避免播放断档；首段则在收到首个文字后最多等 `first_wait_ms`（默认 300 ms）。
4. **任何时候都不在数字或英文单词中间切**（"3." 可能变成 "3.5"、"Pyth" 可能变成 "Python"）。
5. 预合成上限 `max_ahead_s`（默认 6 s）：领先播放太多就暂停合成，被打断时浪费的算力有限。

### 3.2 字级时间对齐（打断截断依赖它）

引擎在每个 chunk **播放完毕后**才把 `chunk.text` 记入"已说出"，所以 chunk 越细、文字与音频的对应越准，打断时写入历史的文本越准确。做法：

- 每段合成后，把这段音频切成"每个对齐单元一个 chunk"（汉字基本是一字一块，`23` 这类数字是一个单元）。
- 有时间标记的后端（edge-tts 的 WordBoundary）直接按标记切。
- 没有时间标记的后端（sherpa-onnx 各模型都不暴露时长）：用能量 VAD 找出段内的停顿，把标点单元**锚定**到离预测位置最近的停顿上，锚点之间按音节权重（汉字 1、英文按元音组数、数字按读法字数、标点按停顿长度）比例分配。
- `TTSChunk.text` 拼起来永远等于原文（`bench tts` 的 alignment_ok 恒为 1），归一化不会丢字。

### 3.3 取消

`synthesize()` 是异步生成器，生产者是独立 task：引擎取消响应 → 生成器关闭 → 立即停止产出、取消生产者、置 `stop` 事件。sherpa-onnx 的回调在句间检查 `stop` 并中止；每个模型一个单线程执行器，已在算的那一段算完即停，不会继续合成后续段。

## 4. 评测方法

- 数据：`bench_data/tts/sentences.jsonl`，60 句，分 6 类：short（短句 10）、oral（口语 10）、numbers（数字/日期/金额/电话 15）、mixed（中英混合 12）、long（长句 8）、symbols（符号 5）。
- 官方指标：`python -m duplexagent bench tts -c configs/exp/tts/<name>.toml --data bench_data/tts/sentences.jsonl`（首包、RTF、alignment_ok），结果在 `results/bench_<name>.json`。
- 扩展指标：`python experiments/tts/eval_tts.py -c configs/exp/tts/<name>.toml --cps 20 --out results/<name>.json`
  - **首包**：从第一段文字到第一个音频 chunk（文本瞬时给完，同 `bench tts`）。
  - **RTF**：合成耗时 / 音频时长。
  - **CER（可懂度）**：用 SenseVoice-Small（sherpa-onnx int8）回识别；参考文本和识别结果都先过同一个归一化器（"23"≡"二十三"），去标点空格后算字错率。
  - **UTMOS**：UTMOS22-strong 的 ONNX 版（MIT），16 kHz。注意它在英文数据上训练，**对中文只适合做相对比较**，绝对值偏低。
  - **对齐命中率**：SenseVoice 的 CTC 时间戳经校准（实测比真实起点早约 0.10 s，见 `eval_tts.py` 注释）后，落在"承载该字的 chunk"时间范围 ±80 ms 内的比例。
  - **截断误差（字）**：每 250 ms 模拟一次打断，比较"按 chunk 记为已说出的字数"与"ASR 认为已听到的字数"，取平均绝对差。这正是引擎写入历史的文本的误差。
  - **流式首包 / 卡顿**：`--cps 20` 时按 20 字/秒（4 字一包）喂文本、按实时播放消费，统计首包和首包之后的播放断档（> 50 ms 记一次）。
  - **取消→空闲**：首个 chunk 到达后关闭流，到合成线程空闲的时间。
- 端到端：`python -m duplexagent eval scenarios/*.toml -c configs/exp/tts/<name>.toml`。

## 5. 结果

### 5.1 官方 `bench tts`（60 句，`results/bench_<name>.json`）

| 候选 | 首包 mean / p50 / p90 (s) | RTF mean / p50 / p90 | alignment_ok |
|---|---|---|---|
| matcha-zh-en | 0.118 / 0.120 / 0.160 | 0.067 / 0.056 / 0.080 | 1.0 |
| vits-aishell3 | 0.125 / 0.119 / 0.171 | 0.054 / 0.041 / 0.063 | 1.0 |
| piper-huayan | 0.143 / 0.140 / 0.199 | 0.075 / 0.067 / 0.081 | 1.0 |
| matcha-zh-baker | 0.160 / 0.159 / 0.206 | 0.080 / 0.070 / 0.092 | 1.0 |
| melo-zh-en | 0.474 / 0.469 / 0.651 | 0.275 / 0.264 / 0.309 | 1.0 |
| edge（在线） | 0.592 / 0.564 / 0.771 | 0.291 / 0.279 / 0.432 | 1.0 |
| kokoro | 0.815 / 0.848 / 1.075 | 0.367 / 0.355 / 0.407 | 1.0 |
| zipvoice | 1.172 / 1.182 / 1.399 | 0.600 / 0.570 / 0.781 | 1.0 |
| kokoro-int8 | 2.346 / 2.352 / 3.149 | 1.046 / 1.043 / 1.152 | 1.0 |

（alignment_ok 只检查 chunk 文本能否拼回原文，基类保证恒为 1；真正的对齐质量看下文的"对齐命中/截断误差"。）

### 5.2 分类 CER

| 候选 | CER long | CER mixed | CER numbers | CER oral | CER short | CER symbols |
|---|---|---|---|---|---|---|
| edge | 0.0080 | 0.0504 | 0.0000 | 0.0000 | 0.0000 | 0.0128 |
| cosyvoice2-http | – | – | – | – | 0.0135 | – |
| matcha-zh-en | 0.0281 | 0.0853 | 0.0227 | 0.0055 | 0.0000 | 0.0128 |
| zipvoice | 0.0100 | 0.1434 | 0.0057 | 0.0275 | 0.0405 | 0.0000 |
| kokoro | 0.0261 | 0.1550 | 0.0113 | 0.0330 | 0.0811 | 0.0385 |
| kokoro-int8 | 0.0220 | 0.1899 | 0.0113 | 0.0440 | 0.0541 | 0.0256 |
| matcha-zh-baker | 0.0240 | 0.3333 | 0.0028 | 0.0110 | 0.0135 | 0.0128 |
| melo-zh-en | 0.0741 | 0.1899 | 0.0453 | 0.0604 | 0.0541 | 0.0385 |
| piper-huayan | 0.0762 | 0.3178 | 0.0227 | 0.0385 | 0.0135 | 0.0128 |
| vits-aishell3 | 0.1844 | 0.7132 | 0.0992 | 0.1923 | 0.1351 | 0.2436 |

- 数字类：所有模型都 ≤ 4.5%（除 aishell3），说明归一化有效（数字/日期/金额/电话/单位都在客户端转成读法）。
- 中英混合是本地模型的主要短板：双语模型 8–19%，纯中文模型 30% 以上（近音字读英文）；只有 edge 低于 5%。

### 5.3 对齐方法消融（matcha-zh-en，`results/sweep_matcha-zh-en_align_*.json`）

| 方法 | chunk/s | 对齐命中 | 截断误差（字） |
|---|---|---|---|
| `segment`：一段一个 chunk（不做字级对齐） | 0.36 | 1.000* | **6.88** |
| `uniform`：每个字等分 | 4.53 | 0.880 | 0.61 |
| `weight`：按音节/停顿权重分 | 4.53 | 0.917 | 0.57 |
| `anchor`（默认）：权重 + 停顿锚定标点 | 4.53 | **0.954** | **0.53** |

\* 一段一个 chunk 时"字落在所属 chunk 内"必然成立，命中率无意义；关键看截断误差：不切块时打断后写入历史的文本平均差 6.9 个字，切成字级后降到约半个字（chunk 播完才计入"已说"，本身就有约半字的保守偏差）。

各候选的对齐命中：有词边界的 edge 98%，Kokoro 98%，Matcha 93–96%，MeloTTS 95%；Piper（80%）、AISHELL-3（84%）、ZipVoice（86%）较低——这几个模型在逗号处不一定停顿、或在非标点处停顿，锚点会被拒绝而退化为按权重分配。

### 5.4 分段策略扫描（matcha-zh-en，`results/sweep_matcha-zh-en_seg_*.json`）

| 分段参数 first_min-first_max-min-max | 首包 p50 (s) | 流式首包 p50 (s, 20 字/秒) | 卡顿 >50 ms 次数 | CER | UTMOS |
|---|---|---|---|---|---|
| 1-6-4-20（极短） | **0.092** | **0.279** | 0 | 0.0374 | 2.50 |
| 2-8-6-30 | 0.102 | 0.381 | 2 | 0.0360 | 2.57 |
| **4-14-10-60（默认）** | 0.141 | 0.391 | 0 | **0.0291** | 2.59 |
| 8-24-20-80 | 0.159 | 0.399 | 3 | 0.0291 | **2.65** |
| 40-120-40-120（≈整句） | 0.237 | 0.395 | 4 | 0.0312 | 2.60 |

- 段越短首包越快，但每段都是独立韵律单元（句尾降调），切得太碎时 CER/UTMOS 略变差（差异与 ±0.5 个百分点的 CER 波动同量级，趋势一致）；段长超过一个分句后收益很小。
- 流式场景下首包主要受"文字到达速度"限制：20 字/秒时前 4 个字就要 0.2 s；整句策略也只有 0.395 s，是因为 `first_wait_ms=300` 的截止时间触发了紧急切分——这说明截止时间机制兜住了"等句号"的最坏情况。
- 默认值在首包与质量间折中；对 Matcha 这种极快的模型，"首段 2–4 字 + 后续整句"都可以。对 RTF 较高的模型（Kokoro/Melo）应把 `first_max_chars` 调小、`lead_ms` 调大。

### 5.5 端到端场景（`python -m duplexagent eval scenarios/*.toml -c configs/exp/tts/<name>.toml --serial`，`results/e2e_*.json`）

| TTS | 通过 | 响应延迟 p50（5 个场景） | 打断停止延迟 |
|---|---|---|---|
| mock（基线） | 5/5 | 0.732–0.733 | 0.32 |
| matcha-zh-en | 5/5 | 0.827–0.901 | 0.32 |
| matcha-zh-baker | 5/5 | 0.866–0.916 | 0.32 |
| piper-huayan | 5/5 | 0.802–0.926 | 0.32 |
| edge | 5/5 | 1.126–1.601 | 0.32 |
| kokoro | **2/5** | 1.509–1.849 | – |

- mock TTS 本身带 80 ms 假首包；matcha-zh-en 比 mock 多 0.09–0.17 s，即真实 TTS 在端到端里占约 0.17–0.25 s（含等快脑凑够首段文字的时间）。
- Kokoro 失败原因：首段音频 1.5 s 以上才出来，场景里用户 5.0 s 处的打断落在 THINKING 阶段（还没开口），被当成"用户接着说"而不是打断，`barge_in_detected`、`turns_detected` 失败。说明 CPU 上 RTF 0.37 的模型不适合全双工。
- 打断后写入历史的文本是字级截断的，例如 barge_in 场景：`played='嗯，你说的是'`；cancel 场景：`'好的，我'`。

### 5.6 取消

所有本地后端在关闭流时生成器立即停止（`cancel_close` ≈ 0）。合成线程回到空闲的时间 p90：Matcha ≤ 0.11 s，Kokoro 0.65 s，AISHELL-3 0.63 s——这是"已在算的那一段"算完的时间，不影响放音（放音已停），只是浪费 CPU；Kokoro 这类慢模型建议减小 `max_chars`。

## 6. 复现

```bash
pip install -r experiments/tts/requirements.txt
python experiments/tts/download.py all eval          # 模型权重到 models/（不入库）
bash experiments/tts/run_all.sh                      # 全部候选（约 1 小时）
bash experiments/tts/sweep.sh matcha-zh-en           # 分段/对齐消融
python experiments/tts/leaderboard.py                # 排行榜
python -m pytest -q experiments/tts                  # 单测（无需模型）
```

## 7. 候选列表与许可证

| 候选 | 实现 | 采样率 | 中英 | 许可证 / 数据 | 状态 |
|---|---|---|---|---|---|
| Matcha-TTS zh-en（dengcunqin，icefall 格式） | `sherpa` / `matcha-zh-en` | 16 kHz | 中+英 | 代码 MIT/Apache；**权重未声明许可** | 实测，推荐 |
| Matcha-TTS zh-baker（icefall） | `sherpa` / `matcha-zh-baker` | 22.05 kHz | 中 | Apache-2.0 代码；标贝数据**仅限非商用** | 实测 |
| MeloTTS zh_en | `sherpa` / `melo-zh-en` | 44.1 kHz | 中+英 | MIT | 实测 |
| VITS AISHELL-3（icefall） | `sherpa` / `vits-aishell3` | 8 kHz 内容 | 中 | Apache-2.0 | 实测，质量差 |
| Piper zh_CN huayan | `sherpa` / `piper-huayan` | 22.05 kHz | 中 | piper-voices MIT；huayan 数据许可未知 | 实测 |
| Kokoro-82M v1.1-zh（fp32 / int8） | `sherpa` / `kokoro`、`kokoro-int8` | 24 kHz | 中+英 | **Apache-2.0** | 实测；CPU 太慢 |
| ZipVoice-Distill int8（零样本克隆） | `sherpa` / `zipvoice` | 24 kHz | 中+英 | 代码 Apache-2.0；训练数据 Emilia 为 CC BY-NC 4.0 | 实测；CPU 太慢 |
| Edge 在线神经音色 | `edge` | 24 kHz mp3 | 中+英 | 微软服务条款，非官方接口 | 实测（在线） |
| CosyVoice 2（0.5B） | `openai_speech` + `servers/cosyvoice_server.py` | 24 kHz | 中+英 | Apache-2.0 | CPU 实测 10 句；**待 GPU 补测** |
| CosyVoice 3（Fun-CosyVoice3-0.5B） | 同上（server 用 `AutoModel` 自动识别） | 24 kHz | 多语 | Apache-2.0 | 未下载（9.8 GB），待 GPU 补测 |
| 阿里云百炼 CosyVoice（云端） | `dashscope` | 16 kHz PCM | 中+英 | 商用付费 | 已写适配器，**待 API Key 补测** |
| OpenAI / 任意兼容 `/v1/audio/speech` 的服务 | `openai_speech` | 可配 | – | – | 已用本地桩服务和 CosyVoice 服务验证接口 |
| Fish-Speech / OpenAudio S1、IndexTTS-2、F5-TTS、Spark-TTS | 以服务形式经 `openai_speech` 接入（需各自的兼容服务或薄封装，同 cosyvoice_server） | – | – | Fish/OpenAudio 权重 CC BY-NC-SA；IndexTTS 自定义许可（商用需联系）；F5-TTS 权重 CC BY-NC；Spark-TTS 权重 CC BY-NC-SA | 未测：均为 0.5B 以上自回归/流匹配模型，参照 CosyVoice 2 在 CPU 上 RTF≈6，需 GPU |

模型下载：`python experiments/tts/download.py <preset>|all|eval`（sherpa-onnx 官方 release 与 Hugging Face）。CosyVoice：
```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice models/third_party/CosyVoice
python -m venv models/third_party/cv-venv   # 独立环境；依赖见仓库 requirements.txt（CPU 版 torch 2.6 可用）
# 权重：huggingface.co/FunAudioLLM/CosyVoice2-0.5B → models/third_party/CosyVoice2-0.5B
models/third_party/cv-venv/bin/python experiments/tts/servers/cosyvoice_server.py --port 8091
python experiments/tts/eval_tts.py -c configs/exp/tts/cosyvoice2-http.toml --limit 10
```

## 8. 风险

1. **许可证**：推荐的 matcha-zh-en 权重未声明许可；baker 数据非商用。若要商用，路线是 Kokoro（Apache，需更强算力）/ MeloTTS（MIT，质量一般）/ CosyVoice（Apache，需 GPU），或用 Matcha 代码（MIT）在自有/可商用数据上重训。
2. **自动评分的局限**：UTMOS 是英文训练的 MOS 预测器，对中文只可作相对参考（例如 Piper 3.66 最高但可懂度差，ZipVoice 的分数受参考音频音质影响很大：原自带新闻提示音频本身只有 1.57 分）。CER 用的 ASR 与参考都过同一个归一化器，对归一化器自身的错误不敏感（如果把"2个"读成"二个"，ASR 也可能写成"二个"）。没有做主观 MOS 听测。
3. **对齐是启发式的**：sherpa-onnx 模型不暴露时长；在不按标点停顿的模型上（Piper/AISHELL-3）只能按权重比例分配，截断误差约 0.6–0.7 字。若需要更准：Kokoro/VITS 的时长预测器输出可以直接导出（需改 ONNX 导出或用 PyTorch 版），或对合成音频跑一次 CTC 强制对齐（用 ASR 模型，耗时约 RTF 0.02，可以只对非首段做）。
4. **段间韵律**：每段独立合成，段首段尾会有"重新起调"的感觉；默认参数让后续段尽量是整句。真正的流式文本输入（CosyVoice 2 bistream、DashScope streaming_call）能根本解决，但需要 GPU/云。
5. **CPU 资源竞争**：评测是独占 4 核；真实部署时 ASR、VAD、快脑（若本地）在同一台机器上抢 CPU，Matcha 用 2 线程留有余量，Kokoro/Melo 这类会明显变差。同一进程内多会话共享一个模型、按段串行（单线程执行器），并发会话多时首包会排队。
6. **英文**：本地模型中英混读 CER 8–19%，人名/品牌（Taylor Swift、Costco）最差；纯中文模型只能用近音字读英文。若业务英文多，优先 edge/云端或 GPU 大模型。
7. edge-tts 使用的是 Edge 浏览器朗读的非公开接口，随时可能变更或限流；在 TLS 中间人代理后需要 `ca_file`（默认读 `$SSL_CERT_FILE`）。

## 9. 接口变更请求

1. **（已提交，`core:` 单独 commit f5c2950）引擎时钟在插件 `start()` 之后才开始。** 原来 `_t0` 在加载模型之前设定，真实 TTS 加载 2 s 会整体平移时间线，导致打断停止延迟和轮次窗口被误判（并发评测时 barge_in 场景失败）。改动是一行、向后兼容，mock 基线结果不变。
2. （建议）`bench tts` 写 wav 时使用 `chunk.sample_rate`，而不是固定 16 kHz；目前本方向所有插件都输出 16 kHz，所以不受影响。
3. （建议）`pyproject.toml` 的 `testpaths` 加上 `experiments`，或合并时把 `experiments/tts/test_tts.py` 移到 `tests/`；目前需要 `python -m pytest -q experiments/tts` 单独跑。
4. （建议）根目录 `.gitignore` 的 `results/` 会忽略规范要求入库的 `experiments/<kind>/results/`，本分支用 `git add -f` 提交了结果 JSON；合并时建议改成只忽略根目录的 `/results/`。
5. （可选，未做）`TTSChunk` 增加可选的 `start_s/end_s` 或 `segment_end` 字段，便于输出端在被打断时按实际播放的样本数精确截断，而不是只能按"整块播完"计入。目前用字级 chunk 已能把误差控制在约半个字，所以不急。

## 10. 待补测

- CosyVoice 2/3 在 GPU 上的首包与 RTF（`cosyvoice_server.py --stream` 的分块流式）；双向流式输入（把快脑的文本流直接送给 CosyVoice）的收益。
- DashScope CosyVoice 云端（需 `DASHSCOPE_API_KEY`）：首包、RTF、词时间戳对齐。
- OpenAI `gpt-4o-mini-tts` 等兼容服务（需 Key）。
- Fish-Speech/OpenAudio、IndexTTS-2、F5-TTS 的兼容服务封装与 GPU 实测。
- 主观听测（小规模 MOS/AB），用来校准 UTMOS 在中文上的可信度。

