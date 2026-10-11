#!/usr/bin/env bash
# 端到端：骨架 5 个场景 + 快脑 4 个场景，逐个候选串行跑（LLM 候选共享一个进程内模型）。
#   bash experiments/fast/e2e.sh router hybrid-gated-qwen3-4b-2507 ...
set -u
cd "$(dirname "$0")/../.."
for name in "$@"; do
  echo "=== $name $(date +%T)"
  python -m duplexagent eval --serial -c "configs/exp/fast/$name.toml" scenarios/*.toml experiments/fast/scenarios/*.toml \
    --out "experiments/fast/results/e2e_$name.json" 2>&1 | grep -v "^llama"
done
echo "=== E2E DONE $(date +%T)"
