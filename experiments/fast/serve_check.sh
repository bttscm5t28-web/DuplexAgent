#!/usr/bin/env bash
# 用本地 llama.cpp OpenAI 兼容服务验证两条“云端/服务端”路径（无需 API Key）：
#   1) 骨架自带的 fast/openai_compat（JSON 工具调用）
#   2) fast/llamacpp backend="openai"（文本协议，走 /v1/chat/completions）
# 需要：pip install 'llama-cpp-python[server]'
set -u
cd "$(dirname "$0")/../.."
MODEL=${MODEL:-models/qwen2.5-3b-instruct-q4_k_m.gguf}
python -m llama_cpp.server --model "$MODEL" --chat_format chatml-function-calling --n_threads 4 --n_ctx 4096 --port 8089 > /tmp/llama_server.log 2>&1 &
PID=$!
trap 'kill $PID' EXIT
until curl -s localhost:8089/v1/models > /dev/null; do sleep 1; done
URL=http://localhost:8089/v1
if [ "${ONLY_NOSTREAM:-0}" = 1 ]; then
python experiments/fast/bench.py delegation --data bench_data/fast/delegation_mini.jsonl --set fast.impl=openai_compat --set fast.base_url="\"$URL\"" \
  --set fast.model='"local"' --set fast.api_key_env='"NO_KEY"' --set fast.stream=false --out experiments/fast/results/serve_openai_compat_nostream.json | cut -c1-600
exit 0
fi
echo "=== openai_compat (tool calls)"
python experiments/fast/bench.py delegation --data bench_data/fast/delegation_mini.jsonl --set fast.impl=openai_compat --set fast.base_url="\"$URL\"" \
  --set fast.model='"local"' --set fast.api_key_env='"NO_KEY"' --out experiments/fast/results/serve_openai_compat.json | cut -c1-600
echo "=== openai_compat stream=false"
python experiments/fast/bench.py delegation --data bench_data/fast/delegation_mini.jsonl --set fast.impl=openai_compat --set fast.base_url="\"$URL\"" \
  --set fast.model='"local"' --set fast.api_key_env='"NO_KEY"' --set fast.stream=false --out experiments/fast/results/serve_openai_compat_nostream.json | cut -c1-600
echo "=== llamacpp backend=openai (text protocol)"
python experiments/fast/bench.py delegation --data bench_data/fast/delegation_mini.jsonl -c configs/exp/fast/hybrid-qwen2.5-3b.toml --set fast.backend='"openai"' \
  --set fast.base_url="\"$URL\"" --set fast.model='"local"' --set fast.api_key_env='"NO_KEY"' --set fast.mode='"llm"' \
  --out experiments/fast/results/serve_llamacpp_openai.json | cut -c1-600
