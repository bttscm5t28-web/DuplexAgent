# 主机实测指南（Mac M5 / A100 / OpenRouter）

五个方向已合并到主分支，各方向的首选方案组成了 `configs/combo/` 下的四套组合配置。云端容器（4 核 CPU，没有声卡和 GPU）只能做模拟测试，真人对话、回声和 GPU 方案要在主机上测。

## 1. 组合配置

| 配置 | 适用 | 输入/输出 | ASR | 轮次/打断 | 快脑 | 慢脑（hybrid 内层） | TTS |
|---|---|---|---|---|---|---|---|
| `cpu.toml` | 任意机器（基准） | 模拟用户 | SenseVoice + VAD 伪流式 | fusion（Silero + 文本规则 + Qwen2.5-0.5B） | 规则路由 + Qwen2.5-1.5B | Qwen3-4B 工具 agent（llama.cpp server） | Matcha zh+en |
| `mac_m5.toml` | Mac M5（Metal） | 浏览器 | 同上 | 同上 | **Qwen3-4B 门控**，全部层放 GPU | 同上，server 加 `--n_gpu_layers -1` | 同上 |
| `a100.toml` | Linux + A100 | 浏览器（SSH 隧道） | 同上 | 同上 | Qwen3-4B 门控，CUDA | **vLLM 上的 Qwen3-32B** | **CosyVoice 2**（GPU） |
| `openrouter.toml` | 有 OpenRouter Key | 模拟用户（可改成浏览器） | 本地 | 本地 | OpenRouter 小模型 | OpenRouter 强模型 | 本地 |

配置支持 `extends`：组合配置只写与 `cpu.toml` 不同的部分。任何键都可以在命令行用 `--set` 覆盖，例如 `--set input.impl=browser --set output.impl=browser`。

## 2. 安装

```bash
git clone https://github.com/bttscm5t28-web/DuplexAgent && cd DuplexAgent
git checkout claude/dual-brain-voice-assistant-rnvomt
scripts/setup_host.sh mac      # 或 cpu / a100
source .venv/bin/activate
```

脚本会做这些事：
- 建 `.venv`，安装依赖。llama.cpp 按 profile 编译（Metal / CUDA / CPU）。
- 下载 Qwen2.5-1.5B、Qwen3-4B 的 GGUF 权重，以及 Matcha TTS 模型。
- 生成 TTS 合成的测试场景音频，最后跑一遍 pytest。

SenseVoice 和 Qwen2.5-0.5B（轮次判断用）首次运行时会自动下载。

## 3. 第一步：用场景复现基准（无需说话）

先确认主机上的数字比云端 4 核 CPU 好，再进入真人测试。

```bash
# 慢脑内层服务（Mac 加 --n_gpu_layers -1；A100 用 a100.toml 里的 vllm 命令）
python -m llama_cpp.server --model models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf --n_ctx 8192 \
    --n_gpu_layers -1 --port 8082 --model_alias qwen3-4b &

python -m duplexagent eval --serial -c configs/combo/mac_m5.toml scenarios/asr/*.toml       --out experiments/combo/results/mac_base.json
python -m duplexagent eval --serial -c configs/combo/mac_m5.toml scenarios/turn/wav/*.toml  --out experiments/combo/results/mac_turn.json
```

评测时有 `--scenario` 就会自动改用模拟输入输出，所以同一份 `mac_m5.toml` 既能跑评测，也能真人对话。

**云端 4 核 CPU 的参照数字**（`configs/combo/cpu.toml`，五个模块同时争抢 CPU，结果在 `experiments/combo/results/`）：见第 7 节。

## 4. 第二步：真人对话

```bash
python -m duplexagent run -c configs/combo/mac_m5.toml \
    --set input.record_path='"results/live/s1_mic.wav"' --timeline results/live/s1.jsonl
# 打开 http://localhost:8765（Chrome），点「开始」，直接对着扬声器说话；关掉页面即结束本次会话
python scripts/live_summary.py results/live/s1.jsonl
```

- **浏览器输入输出**（`browser`）：浏览器的 WebRTC 音频栈负责回声消除、降噪和自动增益，可以外放。这是测全双工的推荐方式。
- **A100**：`ssh -L 8765:localhost:8765 <host>`，然后在本机浏览器打开 `http://localhost:8765`（麦克风权限要求 localhost 或 https）。
- **本地声卡**（`--set input.impl=mic --set output.impl=speaker`）：没有回声消除，必须戴耳机。
- `--timeline` 记录全部事件，`record_path` 保存麦克风音频（与时间线对齐）。两者配合用来标注“这里应该打断 / 不该抢话”。

## 5. 测试清单

每项至少测 10 次。现象记在 `results/live/NOTES.md`（时间点、期望行为、实际行为）。

| # | 测什么 | 怎么测 | 通过标准 |
|---|---|---|---|
| 1 | 响应延迟 | 正常问答 20 轮 | `live_summary` 的 p50 < 0.8 s、p90 < 1.2 s |
| 2 | 句中停顿不抢话 | “我想……（停 1 秒）订一张去上海的票” | 不在停顿处回答 |
| 3 | 附和不打断 | 助手长回复时说“嗯”“对”“好的” | 助手继续说 |
| 4 | 打断 | 助手说话时说“停一下”“不对”“等等” | 0.5 s 内停下，并回应新问题 |
| 5 | **回声（外放）** | 不戴耳机、扬声器音量正常 | 助手不会被自己的声音打断 |
| 6 | 快慢脑 | “帮我算一下 1 到 1000 有多少个质数”，然后闲聊 | 先回执，闲聊正常，结果回来后自然插入 |
| 7 | 进度与取消 | 委派后问“好了吗”，再说“算了不用了” | 能报进度；取消后不再播报结果 |
| 8 | 补充条件 | 委派“规划周末去杭州”，随后补一句“预算两千” | 结果体现补充条件 |
| 9 | 环境噪声 | 开着空调、电视、键盘声 | 不误判开口，不误打断 |
| 10 | 中英混说 | “帮我查一下 iPhone 17 的 price” | 识别正确 |

**真人数据回收**：第 2–5、9 项各录 10–20 段（保存 `record_path` 的 wav），按 `scenarios/turn/*.toml` 的格式写成带 `wav = ...` 的场景，放进 `scenarios/real/`。之后每次改动都用这些真人场景回归。各方向报告都指出，目前的结论建立在合成语音上，真人数据是最需要补的。

## 6. GPU / API 待补测项（按优先级）

1. **快脑**：Qwen3-4B 门控在 Metal / CUDA 上的首 token 延迟。CPU 上约 1.9 s；GPU 上 prefill 只需几十毫秒，预期能进 800 ms 预算。
2. **慢脑**：A100 上 vLLM 跑 Qwen3-32B 的正确率和耗时：`python experiments/slow/bench.py -c configs/combo/a100.toml --name a100_qwen3_32b`。
3. **慢脑（API）**：OpenRouter 强模型跑同一个基准，记录成本。
4. **TTS**：CosyVoice 2 在 A100 上的首包和 RTF：`python experiments/tts/eval_tts.py`（参数见 `experiments/tts/REPORT.md` §6）。
5. **ASR**：FireRedASR2 / Fun-ASR-Nano 在 GPU 上能否让完整句延迟进入 0.6 s（配置里加 `provider = "cuda"`）。
6. **轮次**：TEN Turn Detection 7B 用 vLLM 服务作为 fusion 的第三路（适配器 `backend="openai"`）。

结果 JSON 放 `experiments/combo/results/` 或各方向的 `experiments/<kind>/results/`，然后 commit 推回仓库，我可以接着分析。

## 7. 云端 4 核 CPU 组合实测（参照）

见 `experiments/combo/results/`，由 `configs/combo/cpu.toml` 跑出。
