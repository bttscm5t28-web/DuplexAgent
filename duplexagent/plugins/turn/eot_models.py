"""End-of-turn scorers for the `hybrid` turn detectors. Each returns P(user finished) in [0, 1].

  TextRuleEOT   Chinese heuristics (lexicon.text_eot_prob), stdlib, ~0 ms
  SmartTurnEOT  Pipecat Smart Turn v3 (BSD-2, audio, Whisper-tiny encoder, ONNX); numpy log-mel port
                of WhisperFeatureExtractor so `transformers` is not needed
  LMEOT         LiveKit-style text EOT: P(<|im_end|>) after the user's words under a small chat LM.
                Default weights: Qwen2.5-0.5B-Instruct ONNX (Apache-2.0). NOT LiveKit's own turn-detector
                weights: their license only allows use inside LiveKit Agents.
  TenTurnEOT    TEN Turn Detection (Qwen2.5-7B fine-tune, Apache-2.0 + conditions): finished / unfinished /
                wait via llama.cpp GGUF, or any OpenAI-compatible server (vLLM) hosting it.

Models are cached per process (several engine instances share one ONNX session).
Weights are downloaded from Hugging Face into `models/` on first use.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Any

from duplexagent.plugins.turn.lexicon import text_eot_prob

_lock = threading.Lock()
_cache: dict[str, Any] = {}
MODELS_DIR = os.environ.get("DUPLEX_MODELS_DIR", "models")


def _cached(key: str, factory):
    with _lock:
        if key not in _cache:
            _cache[key] = factory()
        return _cache[key]


def hf_file(repo: str, filename: str, revision: str | None = None) -> str:
    from huggingface_hub import hf_hub_download

    local = os.path.join(MODELS_DIR, repo.replace("/", "__"))
    return hf_hub_download(repo, filename, revision=revision, local_dir=local)


# ---- text rules --------------------------------------------------------------------------------


class TextRuleEOT:
    name = "text"
    needs = "text"
    blocking = False

    def __call__(self, text: str, audio: bytes | None = None, history: list | None = None) -> float:
        return text_eot_prob(text)


# ---- Smart Turn v3 (audio) ---------------------------------------------------------------------


def _mel_filters(np, sr: int = 16000, n_fft: int = 400, n_mels: int = 80):
    """Slaney-style mel filterbank, identical to librosa / transformers' Whisper defaults."""

    def hz_to_mel(f):
        f = np.asanyarray(f, dtype=np.float64)
        lin = f / (200.0 / 3)
        log = 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0)
        return np.where(f >= 1000.0, log, lin)

    def mel_to_hz(m):
        m = np.asanyarray(m, dtype=np.float64)
        lin = m * (200.0 / 3)
        log = 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0))
        return np.where(m >= 15.0, log, lin)

    fftfreqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights


def whisper_log_mel(np, audio, n_fft: int = 400, hop: int = 160, n_mels: int = 80):
    """numpy port of WhisperFeatureExtractor._np_extract_fbank_features (do_normalize applied by caller)."""
    window = np.hanning(n_fft + 1)[:-1]
    x = np.pad(audio.astype(np.float64), n_fft // 2, mode="reflect")
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * window, n=n_fft)) ** 2  # (frames, bins)
    spec = spec[:-1].T  # drop last frame like the reference
    mel = _cached(f"mel:{n_fft}:{n_mels}", lambda: _mel_filters(np, 16000, n_fft, n_mels)) @ spec
    log_spec = np.log10(np.clip(mel, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


class SmartTurnEOT:
    name = "smart_turn"
    needs = "audio"
    blocking = True  # run in a worker thread

    def __init__(self, model_path: str | None = None, filename: str = "smart-turn-v3.2-cpu.onnx", threads: int = 1):
        import numpy as np

        from duplexagent.plugins.turn.vad import _ort_session

        self.np = np
        path = model_path or hf_file("pipecat-ai/smart-turn-v3", filename)
        self.sess = _ort_session(path, threads)
        self.input_name = self.sess.get_inputs()[0].name

    def features(self, audio: bytes):
        np = self.np
        x = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        n = 8 * 16000
        x = x[-n:]
        if len(x) < n:
            x = np.pad(x, (n - len(x), 0))
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        return whisper_log_mel(np, x)[None]

    def __call__(self, text: str, audio: bytes | None = None, history: list | None = None) -> float:
        if not audio:
            return 0.5
        out = self.sess.run(None, {self.input_name: self.features(audio)})
        return float(out[0].reshape(-1)[0])


# ---- LiveKit-style LM end-of-utterance ---------------------------------------------------------


class LMEOT:
    """P(end of turn) = probability mass the chat LM puts on "<|im_end|>" (plus sentence-final
    punctuation, which Chinese ASR output usually lacks) right after the user's words."""

    name = "lm"
    needs = "text"
    blocking = True

    def __init__(
        self,
        repo: str = "onnx-community/Qwen2.5-0.5B-Instruct",
        filename: str = "onnx/model_quantized.onnx",
        threads: int = 2,
        max_ctx_tokens: int = 128,
        punct_tokens: tuple[str, ...] = ("。", "？", "！", "?", "!", "."),
        temperature: float = 1.0,
        system: str | None = "",
        calib: str = "log",
        log_center: float = -1.7,  # log10 P(end) mapped to 0.5 (raw P is tiny for unpunctuated statements)
        log_slope: float = 2.5,
    ):
        import numpy as np
        from tokenizers import Tokenizer

        from duplexagent.plugins.turn.vad import _ort_session

        self.np = np
        self.tok = _cached(f"tok:{repo}", lambda: Tokenizer.from_file(hf_file(repo, "tokenizer.json")))
        for f in ("config.json",):
            hf_file(repo, f)
        model_path = hf_file(repo, filename)
        _has_external(repo, filename)  # download a sidecar weights file next to it, if the export has one
        self.sess = _ort_session(model_path, threads)
        self.inputs = {i.name: i for i in self.sess.get_inputs()}
        self.kv_names = [n for n in self.inputs if n.startswith("past_key_values")]
        import json

        with open(hf_file(repo, "config.json")) as f:
            cfg = json.load(f)
        self.n_kv = cfg["num_key_value_heads"]
        self.head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
        self.kv_dtype = np.float16 if "float16" in self.inputs[self.kv_names[0]].type else np.float32
        self.end_ids = [self.tok.token_to_id("<|im_end|>")]
        self.punct_ids = [i for i in (self._single_id(p) for p in punct_tokens) if i is not None]
        self.max_ctx_tokens = max_ctx_tokens
        self.temperature = temperature
        self.system = system
        self.calib, self.log_center, self.log_slope = calib, log_center, log_slope

    def _single_id(self, s: str):
        ids = self.tok.encode(s, add_special_tokens=False).ids
        return ids[0] if len(ids) == 1 else None

    def prompt(self, text: str, history: list | None) -> str:
        parts = []
        if self.system is not None:
            parts.append(f"<|im_start|>system\n{self.system}<|im_end|>\n")
        for role, content in (history or [])[-4:]:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n{text}")
        return "".join(parts)

    def __call__(self, text: str, audio: bytes | None = None, history: list | None = None) -> float:
        np = self.np
        if not text.strip():
            return 0.5
        ids = self.tok.encode(self.prompt(text.strip(), history), add_special_tokens=False).ids
        ids = ids[-self.max_ctx_tokens :]
        n = len(ids)
        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, n), dtype=np.int64),
        }
        if "position_ids" in self.inputs:
            feed["position_ids"] = np.arange(n, dtype=np.int64)[None]
        for k in self.kv_names:
            feed[k] = np.zeros((1, self.n_kv, 0, self.head_dim), dtype=self.kv_dtype)
        logits = self.sess.run(["logits"], feed)[0][0, -1].astype(np.float64) / self.temperature
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        p = float(probs[self.end_ids].sum() + probs[self.punct_ids].sum())
        if self.calib != "log":
            return p
        z = self.log_slope * (math.log10(max(p, 1e-9)) - self.log_center)
        return 1.0 / (1.0 + math.exp(-z))


def _has_external(repo: str, filename: str) -> bool:
    """Some ONNX exports keep weights in a sidecar `<file>_data`; fetch it when present."""
    try:
        from huggingface_hub import list_repo_files

        files = list_repo_files(repo)
    except Exception:
        return False
    side = filename + "_data"
    if side in files:
        hf_file(repo, side)
        return True
    return False


# ---- TEN Turn Detection ------------------------------------------------------------------------


class TenTurnEOT:
    """TEN Turn Detection: one generated token in {finished, unfinished, wait}.

    backend = "llama_cpp": GGUF via llama-cpp-python (CPU; Q4_K_M ≈ 4.7 GB, ~0.5–2 s per call on 4 cores)
    backend = "openai":    OpenAI-compatible server (e.g. `vllm serve TEN-framework/TEN_Turn_Detection`)
    """

    name = "ten_turn"
    needs = "text"
    blocking = True

    def __init__(
        self,
        backend: str = "llama_cpp",
        gguf_repo: str = "sheldonrobinson/TEN_Turn_Detection-Q4_K_M-GGUF",
        gguf_file: str | None = None,
        threads: int = 4,
        base_url: str = "http://localhost:8000/v1",
        model: str = "TEN-framework/TEN_Turn_Detection",
        system: str = "",
    ):
        self.backend = backend
        self.system = system
        if backend == "llama_cpp":
            self.llm = _cached(f"ten:{gguf_repo}:{gguf_file}", lambda: self._load_llama(gguf_repo, gguf_file, threads))
        elif backend == "openai":
            self.base_url, self.model = base_url, model
        else:
            raise ValueError(f"unknown ten_turn backend {backend!r}")

    @staticmethod
    def _load_llama(repo: str, filename: str | None, threads: int):
        from llama_cpp import Llama

        if filename is None:
            from huggingface_hub import list_repo_files

            filename = next(f for f in list_repo_files(repo) if f.endswith(".gguf"))
        return Llama(model_path=hf_file(repo, filename), n_ctx=512, n_threads=threads, verbose=False, logits_all=False)

    def classify(self, text: str) -> dict[str, float]:
        """Probabilities of the three labels as the first generated token."""
        if self.backend == "llama_cpp":
            import numpy as np

            llm = self.llm
            prompt = (f"<|im_start|>system\n{self.system}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n"
                      "<|im_start|>assistant\n")
            with _lock:  # one llama context: calls must not interleave
                toks = llm.tokenize(prompt.encode(), add_bos=False, special=True)
                llm.reset()
                llm.eval(toks)
                # llama-cpp-python only mirrors logits into .scores with logits_all; read the last row directly
                n_vocab = llm.n_vocab()
                logits = np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(n_vocab,)).astype(np.float64)
            if not hasattr(self, "_label_ids"):
                self._label_ids = {lab: llm.tokenize(lab.encode(), add_bos=False, special=False)[0]
                                   for lab in ("finished", "unfinished", "wait")}
            logits -= logits.max()
            p = np.exp(logits)
            p /= p.sum()
            return {lab: float(p[i]) for lab, i in self._label_ids.items()}
        import json
        import urllib.request

        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": text}]
        body = json.dumps({"model": self.model, "messages": msgs, "max_tokens": 1, "temperature": 0,
                           "logprobs": True, "top_logprobs": 10}).encode()
        req = urllib.request.Request(self.base_url.rstrip("/") + "/chat/completions", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            choice = json.load(r)["choices"][0]
        tops = {}
        for t in ((choice.get("logprobs") or {}).get("content") or [{}])[0].get("top_logprobs", []):
            tops[t["token"].strip().lower()] = math.exp(t["logprob"])
        return tops or {choice["message"]["content"].strip().lower(): 1.0}

    def __call__(self, text: str, audio: bytes | None = None, history: list | None = None) -> float:
        if not text.strip():
            return 0.5
        tops = self.classify(text.strip())
        # "finished" and "wait" (= "shut up") both end the user's turn; tokens may be split ("fin", "un")
        fin = sum(v for k, v in tops.items() if k.startswith(("fin", "wa")))
        unf = sum(v for k, v in tops.items() if k.startswith("un"))
        return fin / (fin + unf) if fin + unf > 0 else 0.5


SCORERS = {"text": TextRuleEOT, "smart_turn": SmartTurnEOT, "lm": LMEOT, "ten_turn": TenTurnEOT}


def make_scorer(name: str, cfg: dict | None = None):
    cls = SCORERS.get(name)
    if cls is None:
        raise ValueError(f"unknown eot scorer {name!r}; expected one of {sorted(SCORERS)}")
    cfg = dict(cfg or {})
    return cls(**cfg)
