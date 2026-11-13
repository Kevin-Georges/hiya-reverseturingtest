"""Audio constants and small helpers. Everything internal is 16 kHz mono float32."""
from __future__ import annotations

import io
import time
import wave

import numpy as np

SR = 16000
FRAME_MS = 20
FRAME = SR * FRAME_MS // 1000  # 320 samples per frame
FRAME_S = FRAME_MS / 1000.0


class Clock:
    """Wall clock with an optional speed-up, so tests with fake backends run faster than real time.

    Real backends (Whisper, an LLM, TTS) always run at speed 1.0, because their compute time
    is part of what we are measuring.
    """

    def __init__(self, speed: float = 1.0):
        self.speed = speed

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds / self.speed)

    def perf(self) -> float:
        return time.perf_counter()


def silence(n: int = FRAME) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def to_frames(pcm: np.ndarray) -> list[np.ndarray]:
    pcm = np.asarray(pcm, dtype=np.float32)
    pad = (-len(pcm)) % FRAME
    if pad:
        pcm = np.concatenate([pcm, silence(pad)])
    return [pcm[i : i + FRAME].copy() for i in range(0, len(pcm), FRAME)]


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear-interpolation resampler. Good enough for speech going into VAD and Whisper."""
    if sr_in == sr_out or len(x) == 0:
        return x.astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    t_in = np.arange(len(x)) / sr_in
    t_out = np.arange(n_out) / sr_out
    return np.interp(t_out, t_in, x).astype(np.float32)


def trim_silence(x: np.ndarray, thresh: float = 0.01, pad_s: float = 0.03) -> np.ndarray:
    """Strip leading and trailing silence so utterance timing reflects actual speech."""
    if len(x) == 0:
        return x
    idx = np.nonzero(np.abs(x) > thresh)[0]
    if len(idx) == 0:
        return x[:0]
    pad = int(pad_s * SR)
    return x[max(0, idx[0] - pad) : min(len(x), idx[-1] + pad)]


def wav_bytes_to_pcm(data: bytes) -> np.ndarray:
    """Decode a PCM WAV (any rate, 8/16/32-bit, mono or stereo) to 16 kHz mono float32."""
    with wave.open(io.BytesIO(data), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width {width}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return resample(x, sr, SR)


def write_wav(path: str, x: np.ndarray, sr: int = SR) -> None:
    """Write float32 audio; x may be (n,) mono or (n, channels)."""
    x = np.asarray(x, dtype=np.float32)
    ch = 1 if x.ndim == 1 else x.shape[1]
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
