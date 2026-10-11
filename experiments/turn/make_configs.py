"""Regenerate configs/exp/turn/*.toml from configs/default.toml, replacing only the [turn] table."""

import os
import re

TURN = {
    "adaptive": ("能量 VAD + 中文文本规则 EOT（自适应静音阈值）+ 命令词/附和词/回声感知的打断判别。纯标准库。", '''impl = "adaptive"
eot = ["text"]
min_silence_ms = 250
max_silence_ms = 2000
base_silence_ms = 500
p_low = 0.15
p_high = 0.75
barge_in_ms = 300
barge_in_max_ms = 1300  # 能量 VAD 分不清噪声：没有文字时要更久才打断'''),
    "silero": ("Silero VAD（MIT）替换能量 VAD，其余同基线：固定 500 ms 静音，无文本 EOT。", '''impl = "silero"
vad_threshold = 0.5
base_silence_ms = 500
barge_in_ms = 300
barge_in_max_ms = 700'''),
    "ten_vad": ("TEN VAD（Apache-2.0 附加条款）替换能量 VAD，其余同基线：固定 500 ms 静音，无文本 EOT。", '''impl = "ten_vad"
vad_threshold = 0.5
base_silence_ms = 500
barge_in_ms = 300
barge_in_max_ms = 700'''),
    "adaptive_silero": ("Silero VAD + adaptive 的文本规则 EOT 与打断判别（纯 CPU，~0.1 ms/帧）。", '''impl = "adaptive"
vad = "silero"
vad_threshold = 0.5
eot = ["text"]
min_silence_ms = 250
max_silence_ms = 2000
base_silence_ms = 500
p_low = 0.15
p_high = 0.75
barge_in_ms = 300
barge_in_max_ms = 700'''),
    "adaptive_ten": ("TEN VAD + adaptive 的文本规则 EOT 与打断判别。", '''impl = "adaptive"
vad = "ten"
vad_threshold = 0.5
eot = ["text"]
min_silence_ms = 250
max_silence_ms = 2000
base_silence_ms = 500
p_low = 0.15
p_high = 0.75
barge_in_ms = 300
barge_in_max_ms = 1300  # TEN VAD 对咳嗽/敲击也给高分'''),
    "smart_turn": ("Silero VAD + Pipecat Smart Turn v3.2（BSD-2，音频 EOT，CPU ~55 ms/次）+ adaptive 的打断判别。", '''impl = "smart_turn"
vad = "silero"
eot = ["smart_turn"]
min_silence_ms = 200
max_silence_ms = 1500
base_silence_ms = 500
p_low = 0.2
p_high = 0.7
audio_probe_ms = 100
hangover_ms = 60
barge_in_ms = 300
barge_in_max_ms = 700'''),
    "lm_eot": ("Silero VAD + LiveKit 式文本 EOT：Qwen2.5-0.5B-Instruct（Apache-2.0）在用户话后接 <|im_end|> 的概率，log 校准。", '''impl = "lm_eot"
vad = "silero"
eot = ["lm"]
min_silence_ms = 200
max_silence_ms = 2000
base_silence_ms = 500
p_low = 0.15
p_high = 0.75
hangover_ms = 60
barge_in_ms = 300
barge_in_max_ms = 700'''),
    "fusion": ("Silero VAD + 文本规则 + Qwen2.5-0.5B LM EOT 融合（加权平均），adaptive 的打断判别。", '''impl = "fusion"
vad = "silero"
eot = ["text", "lm"]
eot_weights = { text = 1.0, lm = 1.0 }
min_silence_ms = 200
max_silence_ms = 2000
base_silence_ms = 500
p_low = 0.15
p_high = 0.75
hangover_ms = 60
barge_in_ms = 300
barge_in_max_ms = 700'''),
    "fusion_st": ("Silero VAD + 文本规则 + LM + Smart Turn 三路融合。", '''impl = "fusion"
vad = "silero"
eot = ["text", "lm", "smart_turn"]
eot_weights = { text = 1.0, lm = 1.0, smart_turn = 0.5 }
min_silence_ms = 200
max_silence_ms = 2000
base_silence_ms = 500
p_low = 0.15
p_high = 0.75
hangover_ms = 60
barge_in_ms = 300
barge_in_max_ms = 700'''),
    "ten_turn": ("Silero VAD + TEN Turn Detection（Qwen2.5-7B 微调，Apache-2.0 附加条款）Q4_K_M GGUF / llama.cpp CPU。", '''impl = "ten_turn"
vad = "silero"
eot = ["ten_turn"]
min_silence_ms = 200
max_silence_ms = 2000
base_silence_ms = 500
p_low = 0.15
p_high = 0.75
hangover_ms = 60
barge_in_ms = 300
barge_in_max_ms = 700'''),
}


def main(only=None):
    base = open("configs/default.toml", encoding="utf-8").read()
    head, rest = base.split("[turn]\n", 1)
    tail = rest[rest.index("\n[") + 1 :]
    head = re.sub(r"^#.*\n", "", head, flags=re.M).lstrip("\n")
    os.makedirs("configs/exp/turn", exist_ok=True)
    for name, (desc, turn) in TURN.items():
        if only and name not in only:
            continue
        txt = f"# 候选：{name} —— {desc}\n# 由 experiments/turn/make_configs.py 基于 configs/default.toml 生成，只改 [turn]。\n\n"
        txt += head + "[turn]\n" + turn + "\n\n" + tail
        open(f"configs/exp/turn/{name}.toml", "w", encoding="utf-8").write(txt)
        print("wrote", name)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:] or None)
