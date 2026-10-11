"""Build real-audio versions of the turn scenarios (wav files are not committed).

    pip install -r experiments/turn/requirements.txt
    python experiments/turn/prepare_data.py            # -> experiments/turn/data/wav/*.wav
                                                       #    scenarios/turn/wav/*.toml (wav = ... fields)

Speech: Piper TTS (rhasspy/piper-voices, zh_CN huayan / xiao_ya / chaowen; piper-tts is GPL-3.0 and
only used here, offline, to make test data). A turn split by thinking pauses ("我想……订一张机票")
is synthesized as ONE sentence and cut at the quietest point near the word boundary, so the first
piece keeps its mid-sentence (non-final) prosody instead of the falling tone of an isolated "我想".
Noise: ESC-50 clips (CC BY-NC 3.0, via the DynamicSuperb parquet mirrors on Hugging Face):
coughing / keyboard typing / door or wood knock; the loudest window of the needed length is used.

Timing: every utterance keeps its scripted offset, shifted by how much longer/shorter the real
audio of the utterances before it turned out to be, so the agent is still talking when a scripted
barge-in or backchannel starts.
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import sys
import tomllib
import wave

import numpy as np

sys.path.insert(0, os.getcwd())

VOICES = ["huayan/medium/zh_CN-huayan-medium", "xiao_ya/medium/zh_CN-xiao_ya-medium", "chaowen/medium/zh_CN-chaowen-medium"]
ESC = {
    "human": ("DynamicSuperb/EnvironmentalSoundClassification_ESC50-HumanAndNonSpeechSounds",
              "data/test-00000-of-00001-f26a742cb0c463e9.parquet"),
    "domestic": ("DynamicSuperb/EnvironmentalSoundClassification_ESC50-InteriorAndDomesticSounds",
                 "data/test-00000-of-00001-f39a5a53c36d3ecd.parquet"),
}
SOUND_LABELS = {"cough": ("human", "coughing"), "keyboard": ("domestic", "keyboard typing"), "knock": ("domestic", "door or wood knock"),
                "sneeze": ("human", "sneezing"), "laugh": ("human", "laughing"), "breath": ("human", "breathing")}
SR = 16000
SPEECH_RMS = 0.08
NOISE_RMS = 0.06


def resample(x: np.ndarray, sr: int, to: int = SR) -> np.ndarray:
    if sr == to:
        return x
    n = int(round(len(x) * to / sr))
    # windowed-sinc would be better; linear interpolation after a crude low-pass is enough for VAD/EOT tests
    if sr > to:
        k = int(np.ceil(sr / to))
        x = np.convolve(x, np.ones(k) / k, mode="same")
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def frame_rms(x: np.ndarray, win: int = 160) -> np.ndarray:
    n = len(x) // win
    return np.sqrt((x[: n * win].reshape(n, win) ** 2).mean(1) + 1e-12)


def trim(x: np.ndarray, thr: float = 0.03, pad: int = 160) -> np.ndarray:
    r = frame_rms(x)
    on = np.nonzero(r > thr * max(1e-6, r.max()))[0]
    if not len(on):
        return x
    a, b = max(0, on[0] * 160 - pad), min(len(x), (on[-1] + 1) * 160 + pad)
    return x[a:b]


def set_rms(x: np.ndarray, target: float) -> np.ndarray:
    r = frame_rms(x)
    act = r[r > 0.2 * r.max()]
    cur = float(np.sqrt((act**2).mean())) if len(act) else 1e-6
    y = x * (target / cur)
    peak = np.abs(y).max()
    return y * (0.95 / peak) if peak > 0.95 else y


def write_wav(path: str, x: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


class TTS:
    def __init__(self, models_dir: str, length_scale: float) -> None:
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice, SynthesisConfig

        self.voices = []
        for v in VOICES:
            onnx = hf_hub_download("rhasspy/piper-voices", f"zh/zh_CN/{v}.onnx", local_dir=os.path.join(models_dir, "piper"))
            hf_hub_download("rhasspy/piper-voices", f"zh/zh_CN/{v}.onnx.json", local_dir=os.path.join(models_dir, "piper"))
            self.voices.append(PiperVoice.load(onnx, download_dir=os.path.join(models_dir, "piper")))
        self.cfg = SynthesisConfig(length_scale=length_scale)

    def __call__(self, text: str, voice: int) -> np.ndarray:
        v = self.voices[voice % len(self.voices)]
        pcm = b"".join(c.audio_int16_bytes for c in v.synthesize(text, syn_config=self.cfg))
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64) / 32768.0
        return trim(resample(x, v.config.sample_rate))


class Noises:
    def __init__(self, models_dir: str) -> None:
        self.dir = models_dir
        self.tables: dict[str, list] = {}

    def clips(self, label: str) -> list[np.ndarray]:
        part, name = SOUND_LABELS[label]
        if part not in self.tables:
            import pyarrow.parquet as pq
            from huggingface_hub import hf_hub_download

            repo, f = ESC[part]
            t = pq.read_table(hf_hub_download(repo, f, repo_type="dataset", local_dir=os.path.join(self.dir, "esc50")))
            self.tables[part] = list(zip(t.column("label").to_pylist(), t.column("audio").to_pylist()))
        out = []
        for lab, audio in self.tables[part]:
            if lab == name:
                with wave.open(io.BytesIO(audio["bytes"])) as w:
                    sr, ch = w.getframerate(), w.getnchannels()
                    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
                if ch > 1:
                    x = x.reshape(-1, ch).mean(1)
                out.append(resample(x, sr))
        return out

    def __call__(self, label: str, seconds: float, k: int) -> np.ndarray:
        clips = self.clips(label)
        x = clips[k % len(clips)]
        n = int(seconds * SR)
        if len(x) < n:
            x = np.tile(x, n // len(x) + 1)
        # loudest window of the requested length (ESC-50 clips are mostly silence around the event)
        e = np.convolve(x**2, np.ones(n), mode="valid")
        i = int(e.argmax())
        y = x[i : i + n].copy()
        fade = min(160, n // 4)
        y[:fade] *= np.linspace(0, 1, fade)
        y[-fade:] *= np.linspace(1, 0, fade)
        return set_rms(y, NOISE_RMS)


def split_at_boundaries(full: np.ndarray, pieces: list[str], tts: TTS, voice: int) -> list[np.ndarray]:
    """Cut a whole-sentence rendering into pieces at the quietest 10 ms frame near each boundary."""
    if len(pieces) == 1:
        return [full]
    total_chars = sum(len(p) for p in pieces)
    r = frame_rms(full)
    cuts, acc = [], 0
    for p in pieces[:-1]:
        acc += len(p)
        # estimate from the duration of the prefix rendered alone, fall back to char ratio
        est = len(tts("".join(pieces)[:acc], voice)) / 160 if acc else 0
        est = 0.5 * est + 0.5 * len(r) * acc / total_chars
        lo, hi = int(est * 0.85), min(len(r) - 1, int(est * 1.15) + 1)
        j = lo + int(np.argmin(r[lo:hi])) if hi > lo else int(est)
        cuts.append(j * 160)
    bounds = [0] + cuts + [len(full)]
    out = []
    for a, b in zip(bounds, bounds[1:]):
        seg = full[a:b].copy()
        f = min(80, len(seg) // 4)  # 5 ms fades against clicks
        seg[:f] *= np.linspace(0, 1, f)
        seg[-f:] *= np.linspace(1, 0, f)
        out.append(seg)
    return out


def build(scenario_path: str, out_toml: str, wav_dir: str, tts: TTS, noises: Noises, voice: int) -> str:
    with open(scenario_path, "rb") as f:
        raw = tomllib.load(f)
    name = raw["name"]
    turns = sorted(raw.get("turns", []), key=lambda u: u["at"])
    cps = 5.0

    # group turn + following conts so they are rendered as one sentence
    groups: list[list[int]] = []
    for i, u in enumerate(turns):
        if u.get("kind") == "cont" and groups:
            groups[-1].append(i)
        else:
            groups.append([i])
    audio: dict[int, np.ndarray] = {}
    for g in groups:
        kind = turns[g[0]].get("kind", "turn")
        if kind == "noise":
            u = turns[g[0]]
            audio[g[0]] = noises(u.get("sound", "cough"), u.get("duration", 0.4), voice)
            continue
        texts = [turns[i]["text"] for i in g]
        full = set_rms(tts("".join(texts), voice), SPEECH_RMS)
        for i, seg in zip(g, split_at_boundaries(full, texts, tts, voice)):
            audio[i] = seg

    shift = 0.0
    lines = [f"# 由 experiments/turn/prepare_data.py 从 {scenario_path} 生成（真实语音版，wav 不入库）。"]
    lines += [ln for ln in open(scenario_path, encoding="utf-8").read().splitlines() if ln.startswith("#")]
    body = []
    end = 0.0
    for i, u in enumerate(turns):
        x = audio[i]
        path = os.path.join(wav_dir, f"{name}_{i}.wav")
        write_wav(path, x)
        new_dur = len(x) / SR
        old_dur = u.get("duration") or max(0.2, len(u["text"]) / u.get("cps", cps))
        at = round(u["at"] + shift, 3)
        if at < end + 0.05 and u.get("kind") != "cont":  # never overlap the previous utterance
            shift += end + 0.05 - at
            at = round(end + 0.05, 3)
        shift += new_dur - old_dur
        end = at + new_dur
        body += ["[[turns]]", f"at = {at}", f"text = {_q(u['text'])}"]
        if u.get("kind", "turn") != "turn":
            body.append(f'kind = "{u["kind"]}"')
        body.append(f"wav = {_q(path)}")
        body.append("")
    duration = max(float(raw.get("duration", end + 5)) + max(0.0, shift), end + 4.0)
    lines += [f'name = "{name}"', f"duration = {round(duration, 2)}", ""] + body
    if raw.get("expect"):
        lines.append("[expect]")
        for k, v in raw["expect"].items():
            lines.append(f"{k} = {_toml_val(v)}")
    os.makedirs(os.path.dirname(out_toml), exist_ok=True)
    with open(out_toml, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_toml


def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _toml_val(v) -> str:
    if isinstance(v, str):
        return _q(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return "[" + ", ".join(_toml_val(x) for x in v) + "]"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=["scenarios/turn/*.toml", "scenarios/*.toml"])
    ap.add_argument("--out", default="scenarios/turn/wav")
    ap.add_argument("--wav-dir", default="experiments/turn/data/wav")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--length-scale", type=float, default=0.9, help="Piper speaking rate (<1 = faster)")
    a = ap.parse_args()

    tts = TTS(a.models_dir, a.length_scale)
    noises = Noises(a.models_dir)
    paths = sorted({p for s in a.scenarios for p in glob.glob(s)})
    for k, p in enumerate(paths):
        base = os.path.splitext(os.path.basename(p))[0]
        prefix = "" if os.path.dirname(p).endswith("turn") else "base_"
        out = build(p, os.path.join(a.out, prefix + base + ".toml"), a.wav_dir, tts, noises, voice=k)
        print("wrote", out)


if __name__ == "__main__":
    main()
