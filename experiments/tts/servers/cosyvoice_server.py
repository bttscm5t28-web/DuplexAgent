"""Minimal OpenAI-compatible `/v1/audio/speech` server for CosyVoice 2/3 (out-of-process TTS).

Runs in CosyVoice's own environment (torch, transformers, ...), so the engine process only needs
the stdlib `openai_speech` plugin:

    # once: git clone --recursive https://github.com/FunAudioLLM/CosyVoice models/third_party/CosyVoice
    #       + its requirements in a separate venv, + model dir (e.g. FunAudioLLM/CosyVoice2-0.5B)
    models/third_party/cv-venv/bin/python experiments/tts/servers/cosyvoice_server.py \
        --repo models/third_party/CosyVoice --model models/third_party/CosyVoice2-0.5B --port 8091

    python -m duplexagent eval scenarios/*.toml -c configs/exp/tts/cosyvoice2-http.toml

Request body: {"input": str, "voice": ignored, "response_format": "pcm", "speed": float}.
Response: raw s16le mono PCM at the model rate (24 kHz), streamed as it is generated
(header X-Sample-Rate). Zero-shot voice = the repo's asset/zero_shot_prompt.wav unless
--prompt-wav/--prompt-text are given.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="models/third_party/CosyVoice")
    ap.add_argument("--model", default="models/third_party/CosyVoice2-0.5B")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--prompt-wav")
    ap.add_argument("--prompt-text", default="希望你以后能够做的比我还好呦。")
    ap.add_argument("--stream", action="store_true", help="use CosyVoice's chunked streaming inference")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    repo = os.path.abspath(a.repo)
    sys.path[:0] = [repo, os.path.join(repo, "third_party", "Matcha-TTS")]
    import numpy as np
    import torch

    torch.set_num_threads(a.threads)
    from cosyvoice.cli.cosyvoice import AutoModel

    t = time.time()
    model = AutoModel(model_dir=os.path.abspath(a.model))
    prompt_wav = a.prompt_wav or os.path.join(repo, "asset", "zero_shot_prompt.wav")
    assert model.add_zero_shot_spk(a.prompt_text, prompt_wav, "default") is True
    print(f"loaded in {time.time() - t:.1f}s, sample_rate={model.sample_rate}", flush=True)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_POST(self) -> None:
            if not self.path.rstrip("/").endswith("/audio/speech"):
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            text = body.get("input", "")
            self.send_response(200)
            self.send_header("Content-Type", "audio/pcm")
            self.send_header("X-Sample-Rate", str(model.sample_rate))
            self.end_headers()
            t0 = time.time()
            first = None
            with lock:
                for out in model.inference_zero_shot(text, "", "", zero_shot_spk_id="default", stream=a.stream, speed=float(body.get("speed", 1.0))):
                    x = out["tts_speech"].squeeze(0).numpy()
                    first = first or time.time() - t0
                    try:
                        self.wfile.write((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
                        self.wfile.flush()
                    except BrokenPipeError:  # client cancelled
                        break
            print(f"{time.time() - t0:.2f}s (first {first and round(first, 2)}s) {text}", flush=True)

        def log_message(self, *args) -> None:
            pass

    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
