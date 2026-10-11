# DuplexAgent

快慢脑结合的全双工语音对话引擎。每个环节（输入、ASR、轮次/打断、快脑、慢脑、TTS、融合策略、输出）都是可替换插件。

```bash
python -m duplexagent eval scenarios/*.toml       # 全 mock 基线，无需模型即可跑通
python -m duplexagent run -s scenarios/delegate.toml
python -m pytest -q
```

- [调研与架构方案](docs/01-research-and-architecture.md)
- [骨架说明与五路实验规范](docs/02-skeleton-and-experiments.md)
