"""Run every turn candidate under every evaluation condition and write the leaderboard.

    python experiments/turn/run_suite.py                       # all conditions x all configs
    python experiments/turn/run_suite.py --cond wav_ideal --config adaptive --config fusion

Conditions (ASR is the experiment mock, so differences come from the turn detector):
  text_ideal   scenarios/turn/*.toml, noise-PCM "speech", mock ASR without lag (energy VAD forced)
  text_lagged  same, ASR partials +300 ms / finals +200 ms
  wav_ideal    scenarios/turn/wav/*.toml (Piper TTS speech, ESC-50 noises), ASR without lag
  wav_lagged   real audio + lagged ASR (the most realistic one)
  wav_vacuum   real audio + continuous vacuum-cleaner noise at 10 dB SNR + lagged ASR
  wav_washer   real audio + washing-machine noise at 10 dB SNR + lagged ASR
  (4-talker babble was dropped: every VAD, neural or not, rightly calls background speech "speech")
  base_wav     the 5 skeleton scenarios as real audio (gate: no regression)
Results: experiments/turn/results/suite_<cond>.json and experiments/turn/results/leaderboard.md
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

from duplexagent.config import load_config  # noqa: E402
from experiments.turn.evaluate import add_experiment_modules, aggregate, run_config  # noqa: E402

TURN = sorted(glob.glob("scenarios/turn/*.toml"))
WAV = [p for p in sorted(glob.glob("scenarios/turn/wav/*.toml")) if not os.path.basename(p).startswith("base_")]
BASE_WAV = sorted(glob.glob("scenarios/turn/wav/base_*.toml"))
LAG = ["asr.impl=lagged_mock", "asr.partial_lag_ms=300", "asr.final_lag_ms=200"]
IDEAL = ["asr.impl=lagged_mock", "asr.partial_lag_ms=0", "asr.final_lag_ms=0"]
NOISY = ["input.impl=noisy_sim"]

CONDITIONS = {
    "text_ideal": (TURN, ["asr.impl=mock", "turn.vad=\"energy\""]),
    "text_lagged": (TURN, LAG + ["turn.vad=\"energy\""]),
    "wav_ideal": (WAV, IDEAL),
    "wav_lagged": (WAV, LAG),
    "wav_vacuum": (WAV, LAG + NOISY + ['input.noise_wav="experiments/turn/data/bg/vacuum.wav"', "input.snr_db=10"]),
    "wav_washer": (WAV, LAG + NOISY + ['input.noise_wav="experiments/turn/data/bg/washer.wav"', "input.snr_db=10"]),
    "base_wav": (BASE_WAV, IDEAL),
}
CONFIGS = ["configs/default.toml"] + sorted(glob.glob("configs/exp/turn/*.toml"))


def cname(path: str) -> str:
    return "rule" if path.endswith("default.toml") else os.path.splitext(os.path.basename(path))[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", action="append", choices=sorted(CONDITIONS))
    ap.add_argument("--config", action="append", help="config names (e.g. adaptive, rule)")
    ap.add_argument("-j", "--concurrency", type=int, default=8)
    ap.add_argument("--skip", action="append", default=[], help="config names to skip")
    a = ap.parse_args()
    conds = a.cond or list(CONDITIONS)
    configs = [c for c in CONFIGS if (not a.config or cname(c) in a.config) and cname(c) not in a.skip]
    os.makedirs("experiments/turn/results", exist_ok=True)
    for cond in conds:
        paths, sets = CONDITIONS[cond]
        out = f"experiments/turn/results/suite_{cond}.json"
        res = json.load(open(out)) if os.path.exists(out) else {}
        for c in configs:
            name = cname(c)
            cfg = load_config(c, sets)
            if name == "rule" and "turn.vad=\"energy\"" in sets:
                cfg["turn"].pop("vad", None)  # the baseline has no vad option
            add_experiment_modules(cfg)
            t0 = time.time()
            try:
                runs = asyncio.run(run_config(cfg, paths, a.concurrency))
            except Exception as e:  # a candidate whose deps are missing must not stop the suite
                print(f"[{cond}] {name}: ERROR {type(e).__name__}: {e}", flush=True)
                continue
            agg = aggregate(runs)
            agg["wall_s"] = round(time.time() - t0, 1)
            for r in runs:
                r.pop("events", None)
            res[name] = {"config": c, "overrides": sets, "aggregate": agg, "scenarios": runs}
            print(f"[{cond}] {name:<16} score={agg['score']} pass={agg['scenarios_passed']}/{agg['scenarios']} cut_in={agg['cut_in_rate']} "
                  f"miss_barge={agg['missed_barge_rate']} false_barge={agg['false_barge_rate']} false_turn={agg['false_turn']} "
                  f"eot50={agg['eot_p50']} resp50={agg['resp_p50']} stop50={agg['stop_p50']}", flush=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
    write_leaderboard()


COLS = [("score", "综合分↓"), ("scenarios_passed", "场景通过"), ("cut_in_rate", "抢话率"), ("missed_barge_rate", "漏打断率"),
        ("false_barge_rate", "误打断率"), ("false_turn", "噪声/附和误成轮次"), ("missed_eot", "漏判说完"), ("eot_p50", "说完→TURN_END p50"),
        ("eot_p90", "p90"), ("resp_p50", "说完→出声 p50"), ("stop_p50", "打断停止 p50"), ("stop_p90", "p90")]


def write_leaderboard() -> None:
    lines = ["# 轮次/打断候选排行榜（自动生成：experiments/turn/run_suite.py）", ""]
    for cond in CONDITIONS:
        f = f"experiments/turn/results/suite_{cond}.json"
        if not os.path.exists(f):
            continue
        res = json.load(open(f))
        lines += [f"## {cond}", "", "| 候选 | " + " | ".join(h for _, h in COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
        for name, r in sorted(res.items(), key=lambda kv: kv[1]["aggregate"]["score"]):
            g = r["aggregate"]
            cells = [f"{g['scenarios_passed']}/{g['scenarios']}" if k == "scenarios_passed" else ("—" if g[k] is None else str(g[k])) for k, _ in COLS]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    with open("experiments/turn/results/leaderboard.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
