#!/usr/bin/env bash
# Evaluate TTS candidates one after another (sequential, so latencies are not distorted).
#   bash experiments/tts/run_all.sh [name ...]        # names = configs/exp/tts/<name>.toml
# Writes experiments/tts/results/<name>.json (eval_tts) and bench_<name>.json (`bench tts`).
set -u
cd "$(dirname "$0")/../.."
names=("$@")
[ ${#names[@]} -eq 0 ] && names=($(ls configs/exp/tts/*.toml | xargs -n1 basename | sed 's/.toml$//'))
for n in "${names[@]}"; do
  echo "=== $n $(date +%T)"
  python -m duplexagent bench tts -c configs/exp/tts/$n.toml --data bench_data/tts/sentences.jsonl \
    --out experiments/tts/results/bench_$n.json 2>/dev/null | tr -d '\n ' ; echo
  python experiments/tts/eval_tts.py -c configs/exp/tts/$n.toml --data bench_data/tts/sentences.jsonl \
    --cps ${CPS:-20} --wav-dir out/tts/$n --out experiments/tts/results/$n.json 2>/dev/null | tail -1
done
