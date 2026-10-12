#!/usr/bin/env bash
# End-to-end scenarios with each TTS candidate swapped in (serial, so latencies are comparable).
#   bash experiments/tts/e2e.sh [name ...]      # "mock" = configs/default.toml baseline
set -u
cd "$(dirname "$0")/../.."
names=("$@")
[ ${#names[@]} -eq 0 ] && names=(mock matcha-zh-en matcha-zh-baker piper-huayan kokoro edge)
for n in "${names[@]}"; do
  c=configs/exp/tts/$n.toml; [ "$n" = mock ] && c=configs/default.toml
  echo "=== $n"
  python -m duplexagent eval scenarios/*.toml -c $c --serial --out experiments/tts/results/e2e_$n.json 2>/dev/null
done
