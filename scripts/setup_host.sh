#!/usr/bin/env bash
# 在主机上准备运行环境和模型。用法：
#   scripts/setup_host.sh cpu      # 任意机器，纯 CPU（configs/combo/cpu.toml）
#   scripts/setup_host.sh mac      # Apple silicon，llama.cpp 走 Metal（configs/combo/mac_m5.toml）
#   scripts/setup_host.sh a100     # Linux + NVIDIA，llama.cpp 走 CUDA，另装 vLLM（configs/combo/a100.toml）
# 模型权重都放在 models/（不入库）。可重复执行，已下载的会跳过。
set -euo pipefail
profile="${1:-cpu}"
cd "$(dirname "$0")/.."

python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -U pip wheel

pip install -e ".[llm,live,dev]"
# ASR / 轮次 / TTS / 慢脑工具（轻量依赖，三种 profile 相同）
pip install numpy huggingface_hub sherpa-onnx onnxruntime soundfile soxr tokenizers pyarrow \
            anthropic mcp claude-agent-sdk
pip install --no-deps silero-vad            # 只取 Silero VAD 的 ONNX 权重

case "$profile" in
  cpu)  pip install "llama-cpp-python[server]" --prefer-binary \
            --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu ;;
  mac)  CMAKE_ARGS="-DGGML_METAL=on" pip install "llama-cpp-python[server]" --no-binary llama-cpp-python ;;
  a100) CMAKE_ARGS="-DGGML_CUDA=on" pip install "llama-cpp-python[server]" --no-binary llama-cpp-python
        pip install vllm ;;
  *) echo "unknown profile: $profile (cpu|mac|a100)"; exit 1 ;;
esac

# 快脑 / 慢脑的 GGUF 权重
python - <<'PY'
from huggingface_hub import hf_hub_download
for repo, f in [("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
                ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")]:
    print(hf_hub_download(repo, f, local_dir="models"))
PY
# TTS（Matcha zh+en）；ASR（SenseVoice）和轮次 LM（Qwen2.5-0.5B）首次运行时自动下载
python experiments/tts/download.py matcha-zh-en

if [ "$profile" = "a100" ]; then
  # CosyVoice 2：单独的 venv，避免和主环境的 torch 版本冲突
  [ -d models/third_party/CosyVoice ] || git clone --recursive https://github.com/FunAudioLLM/CosyVoice models/third_party/CosyVoice
  python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("FunAudioLLM/CosyVoice2-0.5B", local_dir="models/third_party/CosyVoice2-0.5B")
PY
  echo "CosyVoice 依赖请按 models/third_party/CosyVoice/README 装进 models/third_party/cv-venv"
fi

# 真实音频测试场景（TTS 合成，用于对照；真人录音见 docs/03-host-testing.md）
python experiments/asr/prepare_data.py --scenarios
pip install "piper-tts[zh]" && python experiments/turn/prepare_data.py || echo "跳过轮次场景音频（piper 安装失败）"

python -m pytest -q
echo "完成。下一步见 docs/03-host-testing.md"
