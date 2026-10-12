#!/usr/bin/env bash
# Ablations on one backend (default matcha-zh-en): segmentation strategy and alignment method.
#   bash experiments/tts/sweep.sh [config-name]
set -u
cd "$(dirname "$0")/../.."
n=${1:-matcha-zh-en}
c=configs/exp/tts/$n.toml
R=experiments/tts/results
ev() { python experiments/tts/eval_tts.py -c $c --data bench_data/tts/sentences.jsonl "$@" 2>/dev/null | tail -1; }
# alignment: one chunk per segment / equal share per char / syllable weights / weights + pause anchors
for a in segment uniform weight anchor; do
  echo "== align=$a"; ev --no-mos --set tts.align=\"$a\" --out $R/sweep_${n}_align_$a.json
done
# segmentation: (first_min, first_max, min, max)
for s in "1 6 4 20" "2 8 6 30" "4 14 10 60" "8 24 20 80" "40 120 40 120"; do
  set -- $s
  echo "== seg=$1-$2-$3-$4"
  ev --cps ${CPS:-20} --set tts.first_min_chars=$1 --set tts.first_max_chars=$2 --set tts.min_chars=$3 --set tts.max_chars=$4 \
     --out $R/sweep_${n}_seg_$1-$2-$3-$4.json
done
