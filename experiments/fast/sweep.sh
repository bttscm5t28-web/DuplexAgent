#!/usr/bin/env bash
# 依次跑快脑候选的离线基准（4 核 CPU 上串行，避免互相抢核）。
#   bash experiments/fast/sweep.sh router llm-qwen2.5-1.5b ...
set -u
cd "$(dirname "$0")/../.."
for name in "$@"; do
  echo "=== $name $(date +%T)"
  python experiments/fast/bench.py all --warm -c "configs/exp/fast/$name.toml" --out "experiments/fast/results/bench_$name.json" 2>&1 | cut -c1-600
done
echo "=== SWEEP DONE $(date +%T)"
