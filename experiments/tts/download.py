"""Download TTS / evaluation model files into models/ (git-ignored).

    python experiments/tts/download.py matcha-zh-en melo-zh-en     # sherpa presets
    python experiments/tts/download.py all                          # every sherpa preset
    python experiments/tts/download.py eval                         # SenseVoice ASR + UTMOS (for eval_tts.py)
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from duplexagent.plugins.tts.sherpa import PRESETS, RELEASE  # noqa: E402

EVAL = {
    "models/asr": [f"{RELEASE}/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2"],
    "models/eval": ["https://huggingface.co/TigreGotico/utmos-onnx/resolve/main/utmos22_strong.onnx"],
}


def fetch(url: str, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    name = url.rsplit("/", 1)[-1]
    target = os.path.join(dest, name.removesuffix(".tar.bz2"))
    if os.path.exists(target):
        print("have", target)
        return
    print("get ", url)
    tmp = os.path.join(dest, name + ".part")
    subprocess.run(["curl", "-sSfL", "--retry", "4", "-C", "-", "-o", tmp, url], check=True)
    if name.endswith(".tar.bz2"):
        subprocess.run(["tar", "xjf", tmp, "-C", dest], check=True)
        os.remove(tmp)
    else:
        os.replace(tmp, target)


def main(names: list[str]) -> None:
    if not names:
        print(__doc__)
        print("presets:", ", ".join(sorted(PRESETS)))
        return
    for n in names:
        if n == "eval":
            for dest, urls in EVAL.items():
                for u in urls:
                    fetch(u, dest)
            continue
        for p in sorted(PRESETS) if n == "all" else [n]:
            for f in PRESETS[p]["files"]:
                fetch(f"{RELEASE}/{f}", "models/tts")


if __name__ == "__main__":
    main(sys.argv[1:])
