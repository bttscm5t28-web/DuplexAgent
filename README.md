# DuplexAgent

快慢脑结合的全双工语音对话引擎。每个环节（输入、ASR、轮次/打断、快脑、慢脑、TTS、融合策略、输出）都是可替换插件。

```bash
python -m duplexagent eval scenarios/*.toml                     # 全 mock 基线，无需模型即可跑通
python -m duplexagent run -c configs/combo/mac_m5.toml          # 真人对话：打开 http://localhost:8765
python -m duplexagent list                                      # 所有已注册插件
python -m pytest -q
```

- [调研与架构方案](docs/01-research-and-architecture.md)
- [骨架说明与五路实验规范](docs/02-skeleton-and-experiments.md)
- [主机实测指南（Mac M5 / A100 / OpenRouter）](docs/03-host-testing.md)
- 五路实验报告：[ASR](experiments/asr/REPORT.md) · [打断/轮次](experiments/turn/REPORT.md) · [快脑](experiments/fast/REPORT.md) · [慢脑](experiments/slow/REPORT.md) · [TTS](experiments/tts/REPORT.md)
- 组合配置：`configs/combo/`（cpu / mac_m5 / a100 / openrouter）
