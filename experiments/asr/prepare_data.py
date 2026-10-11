"""下载并生成 ASR 评测数据（wav 不入库，jsonl 入库）。

    python experiments/asr/prepare_data.py                 # 全部数据集
    python experiments/asr/prepare_data.py --sets aishell ascend
    python experiments/asr/prepare_data.py --scenarios     # 生成 scenarios/asr/*.toml 用的 TTS 音频

数据来源（均来自 Hugging Face）：
  aishell   AISHELL-1 test（carlot/AIShell 的 parquet 转存，原始许可 Apache-2.0）  朗读、安静
  ascend    CAiRE/ASCEND test 中 language=mixed 的句子（CC BY-SA 4.0）           中英混说、口语
  wenet     WenetSpeech test_net（TwinkStart/WenetSpeech 转存；WenetSpeech 仅限非商业研究用途）  网络视频、口语
  noisy     从 aishell 子集派生：白噪声 / 多人 babble 噪声，SNR 10 dB 与 0 dB

输出：bench_data/asr/<set>.jsonl（每行 {"wav", "text", "id", "dur"}），
      wav 写到 bench_data/asr/wav/<set>/（已被 bench_data/asr/.gitignore 忽略）。
所有 wav 为 16 kHz 单声道 int16，峰值归一化到 0.7，首尾静音各留 ~300 ms（裁得太紧会切掉句尾轻声），
这样 `bench asr --realtime` 的「说完」时刻接近真实语音结束。

依赖：numpy pyarrow soundfile huggingface_hub（见 requirements.txt）。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "bench_data", "asr")
CACHE = os.path.join(ROOT, "models", "data")
SR = 16000

SOURCES = {
    "aishell": ("carlot/AIShell", "data/test-00000-of-00003-25094929c3ce9a28.parquet"),
    "ascend": ("CAiRE/ASCEND", "main/test-00000-of-00001.parquet"),
    "wenet": ("TwinkStart/WenetSpeech", "data/test_net-00000-of-00002.parquet"),
}


def fetch(name: str) -> str:
    from huggingface_hub import hf_hub_download

    repo, fn = SOURCES[name]
    return hf_hub_download(repo, fn, repo_type="dataset", cache_dir=CACHE)


def decode(raw: bytes):
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:  # linear resampling is enough for ASR test audio
        n = int(len(x) * SR / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype("float32")
    return x


def trim(x, lead: float = 0.3, tail: float = 0.3, thresh_db: float = -40.0):
    """Cut leading/trailing silence: voiced = within 40 dB of the loudest frame and 10 dB above the noise floor."""
    import numpy as np

    f = 320
    n = len(x) // f
    if n == 0:
        return x
    e = 10 * np.log10((x[: n * f].reshape(n, f) ** 2).mean(axis=1) + 1e-10)
    voiced = np.where(e > max(e.max() + thresh_db, np.percentile(e, 10) + 10))[0]
    if len(voiced) == 0:
        return x
    a = max(0, voiced[0] * f - int(lead * SR))
    b = min(len(x), (voiced[-1] + 1) * f + int(tail * SR))
    return x[a:b]


def write_wav(path: str, x) -> None:
    import numpy as np

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)


def read_wav(path: str):
    import numpy as np

    with wave.open(path, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype("float32") / 32768


def rel(p: str) -> str:
    return os.path.relpath(p, ROOT)


def dump(name: str, rows: list[dict]) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dur = sum(r["dur"] for r in rows)
    print(f"{name}: {len(rows)} utts, {dur / 60:.1f} min -> bench_data/asr/{name}.jsonl")


def iter_parquet(path: str, columns: list[str]):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=64, columns=columns):
        yield from batch.to_pylist()


def build(name: str, rows, n: int, text_key: str, keep, seed: int = 0) -> list[dict]:
    """Reservoir-sample n rows that pass `keep`, decode, trim and write them."""
    rnd = random.Random(seed)
    pool: list[dict] = []
    seen = 0
    for r in rows:
        if not keep(r):
            continue
        seen += 1
        if len(pool) < n:
            pool.append(r)
        elif (j := rnd.randrange(seen)) < n:
            pool[j] = r
    out = []
    for i, r in enumerate(sorted(pool, key=lambda r: r["audio"]["path"])):
        x = trim(decode(r["audio"]["bytes"]))
        x *= 0.7 / max(1e-6, float(abs(x).max()))  # peak-normalize (like a mic with AGC); AISHELL is recorded at ~-35 dBFS
        uid = os.path.splitext(os.path.basename(r["audio"]["path"]))[0] or f"{i:04d}"
        p = os.path.join(OUT, "wav", name, f"{uid}.wav")
        write_wav(p, x)
        out.append({"id": uid, "wav": rel(p), "text": " ".join(r[text_key].split()), "dur": round(len(x) / SR, 3)})
    return out


def prep_aishell(n: int) -> None:
    rows = build("aishell", iter_parquet(fetch("aishell"), ["audio", "transcription"]), n, "transcription", lambda r: True)
    for r in rows:  # AISHELL transcripts are word-segmented with spaces; drop them
        r["text"] = r["text"].replace(" ", "")
    dump("aishell", rows)
    dump("aishell_rt", rows[::4])  # realtime subset (realtime runs cost wall-clock = audio length)


def prep_ascend(n: int) -> None:
    def keep(r):
        return r["language"] == "mixed" and 1.5 <= r["duration"] <= 10.0 and len(r["transcription"]) >= 6

    rows = build("ascend", iter_parquet(fetch("ascend"), ["audio", "transcription", "language", "duration"]), n, "transcription", keep)
    dump("ascend", rows)


def prep_wenet(n: int) -> None:
    def keep(r):
        return 1.5 <= r["end_time"] - r["begin_time"] <= 10.0 and len(r["text"]) >= 5

    rows = build("wenet", iter_parquet(fetch("wenet"), ["audio", "text", "begin_time", "end_time"]), n, "text", keep)
    dump("wenet", rows)


def prep_noisy(n: int) -> None:
    import numpy as np

    src = [json.loads(line) for line in open(os.path.join(OUT, "aishell.jsonl"), encoding="utf-8")]
    rnd = np.random.default_rng(0)
    base = src[1::4][:n]
    others = [read_wav(os.path.join(ROOT, r["wav"])) for r in src[::4][:40]]

    def babble(m: int):
        out = np.zeros(m, "float32")
        for k in rnd.choice(len(others), 4, replace=False):
            o = np.tile(others[k], m // len(others[k]) + 2)
            off = rnd.integers(0, len(others[k]))
            out += o[off : off + m] / (np.sqrt((o**2).mean()) + 1e-9)
        return out

    for kind in ("white", "babble"):
        for snr in (10, 0):
            name = f"noisy_{kind}{snr}"
            rows = []
            for r in base:
                x = read_wav(os.path.join(ROOT, r["wav"]))
                nz = rnd.standard_normal(len(x)).astype("float32") if kind == "white" else babble(len(x))
                # SNR measured over the voiced part of the clean signal
                ps = (x[np.abs(x) > 0.02 * np.abs(x).max()] ** 2).mean()
                nz *= np.sqrt(ps / (10 ** (snr / 10)) / ((nz**2).mean() + 1e-12))
                y = x + nz
                y /= max(1.0, np.abs(y).max() / 0.95)
                p = os.path.join(OUT, "wav", name, f"{r['id']}.wav")
                write_wav(p, y)
                rows.append({**r, "wav": rel(p)})
            dump(name, rows)


# ---- scenario audio ------------------------------------------------------------------

TTS_REPO = "csukuangfj/vits-zh-hf-fanchen-C"  # 中文多说话人 VITS（sherpa-onnx 格式）


def tts_engine():
    import sherpa_onnx
    from huggingface_hub import snapshot_download

    d = snapshot_download(TTS_REPO, local_dir=os.path.join(ROOT, "models", "tts", TTS_REPO.split("/")[1]))
    files = os.listdir(d)
    onnx = next(f for f in files if f.endswith(".onnx"))
    rule_fsts = ",".join(os.path.join(d, f) for f in ("phone.fst", "date.fst", "number.fst") if f in files)
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=os.path.join(d, onnx), lexicon=os.path.join(d, "lexicon.txt"), tokens=os.path.join(d, "tokens.txt"), dict_dir=os.path.join(d, "dict")),
            num_threads=2,
        ),
        rule_fsts=rule_fsts,
    )
    return sherpa_onnx.OfflineTts(cfg)


def prep_scenarios(speaker: int = 2, speed: float = 1.0) -> None:
    """Synthesize every turn of scenarios/*.toml and write scenarios/asr/<name>.toml with `wav = ...`."""
    import tomllib

    import numpy as np

    tts = tts_engine()
    sdir = os.path.join(ROOT, "scenarios")
    odir = os.path.join(sdir, "asr")
    for fn in sorted(os.listdir(sdir)):
        if not fn.endswith(".toml"):
            continue
        with open(os.path.join(sdir, fn), "rb") as f:
            sc = tomllib.load(f)
        lines = [f"# {sc.get('name', fn)}：与 scenarios/{fn} 相同，但每句话带 TTS 合成的 wav（由 experiments/asr/prepare_data.py --scenarios 生成）",
                 f'name = "asr_{sc.get("name", fn[:-5])}"', f"duration = {sc['duration']}", ""]
        for k, t in enumerate(sc["turns"]):
            audio = tts.generate(t["text"], sid=speaker, speed=speed)
            x = np.asarray(audio.samples, "float32")
            if audio.sample_rate != SR:
                m = int(len(x) * SR / audio.sample_rate)
                x = np.interp(np.linspace(0, len(x) - 1, m), np.arange(len(x)), x).astype("float32")
            x = trim(x, lead=0.0, tail=0.0)
            x *= 0.5 / max(1e-6, np.abs(x).max())
            p = os.path.join(odir, "wav", f"{fn[:-5]}_{k}.wav")
            write_wav(p, x)
            lines += ["[[turns]]", f"at = {t['at']}", f"text = {json.dumps(t['text'], ensure_ascii=False)}", f'wav = "{rel(p)}"']
            if "kind" in t:
                lines.append(f'kind = "{t["kind"]}"')
            lines.append("")
            print(f"{fn}: {t['text']} -> {rel(p)} ({len(x) / SR:.2f}s)")
        lines.append("[expect]")
        for k, v in sc.get("expect", {}).items():
            lines.append(f"{k} = {json.dumps(v, ensure_ascii=False)}")
        with open(os.path.join(odir, fn), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=["aishell", "ascend", "wenet", "noisy"])
    ap.add_argument("--n", type=int, default=200, help="utterances per set (noisy: n/4)")
    ap.add_argument("--scenarios", action="store_true", help="only synthesize scenario audio")
    a = ap.parse_args()
    if a.scenarios:
        prep_scenarios()
        return
    for s in a.sets:
        {"aishell": lambda: prep_aishell(a.n), "ascend": lambda: prep_ascend(a.n // 2), "wenet": lambda: prep_wenet(a.n // 2),
         "noisy": lambda: prep_noisy(a.n // 4)}[s]()


if __name__ == "__main__":
    sys.exit(main())
