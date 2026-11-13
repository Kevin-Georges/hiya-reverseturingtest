"""Real sound devices: a live human caller (mic + headphones) and a monitor to listen in on bot calls."""
from __future__ import annotations

import collections
import threading

import numpy as np

from .audio import FRAME, SR
from .line import Endpoint


class HumanEndpoint(Endpoint):
    """The caller side is a person: the line takes frames from the mic and plays the screener in their headphones.

    Use headphones. With speakers the mic hears the screener and the screener hears itself.
    """

    def __init__(self, name: str = "caller", input_device=None, output_device=None):
        super().__init__(name)
        import sounddevice as sd

        self._mic: collections.deque = collections.deque(maxlen=10)  # bounded: drop old audio rather than lag
        self._spk: collections.deque = collections.deque(maxlen=25)
        self._lk = threading.Lock()
        self.stream = sd.Stream(
            samplerate=SR,
            blocksize=FRAME,
            channels=1,
            dtype="float32",
            device=(input_device, output_device),
            callback=self._cb,
        )
        self.stream.start()

    def _cb(self, indata, outdata, frames, time_info, status):
        with self._lk:
            self._mic.append(indata[:, 0].copy())
            out = self._spk.popleft() if self._spk else np.zeros(frames, dtype=np.float32)
        outdata[:, 0] = out[:frames] if len(out) >= frames else np.pad(out, (0, frames - len(out)))

    def pull_frame(self, t: float):
        with self._lk:
            return self._mic.popleft() if self._mic else None

    def push_frame(self, frame, t: float) -> None:
        with self._lk:
            self._spk.append(frame)

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()


class Monitor:
    """Plays the mix of both sides of a simulated call on the speakers (handy for the demo video)."""

    def __init__(self, device=None):
        import sounddevice as sd

        self._q: collections.deque = collections.deque(maxlen=50)
        self._lk = threading.Lock()
        self.stream = sd.OutputStream(samplerate=SR, blocksize=FRAME, channels=1, dtype="float32", device=device, callback=self._cb)
        self.stream.start()

    def __call__(self, frame: np.ndarray) -> None:
        with self._lk:
            self._q.append(frame)

    def _cb(self, outdata, frames, time_info, status):
        with self._lk:
            out = self._q.popleft() if self._q else np.zeros(frames, dtype=np.float32)
        outdata[:, 0] = out[:frames]

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()


def play_blocking(pcm: np.ndarray, device=None) -> None:
    import sounddevice as sd

    sd.play(pcm, SR, device=device)
    sd.wait()


def check_echo() -> int:
    """Play speech through the output while recording the mic, and report whether the mic hears it.

    python -m rtt.audio_io      (wear your headphones as you would on a call, and stay quiet)
    """
    import sounddevice as sd

    from .backends import Pyttsx3TTS

    i, o = sd.default.device
    print(f"output: {sd.query_devices(o)['name']}\ninput:  {sd.query_devices(i)['name']}")
    print("Stay quiet. Playing a test sentence in 1 second...", flush=True)
    speech = Pyttsx3TTS(voice_index=1).synth(
        "This is an audio check. If you can hear this, stay quiet until it finishes. One, two, three, four, five.")
    gap = np.zeros(int(1.5 * SR), dtype=np.float32)
    played = np.concatenate([gap, speech, gap])
    rec = sd.playrec(played, SR, channels=1, dtype="float32")
    sd.wait()
    rec = rec[:, 0]

    fr = FRAME
    n = len(played) // fr
    env = lambda x: np.sqrt((x[: n * fr].reshape(n, fr) ** 2).mean(1)) + 1e-9
    ep, er = env(played), env(rec)
    talking = ep > 0.02
    quiet = np.r_[np.ones(int(1.2 * SR / fr), bool), np.zeros(n - int(1.2 * SR / fr), bool)]  # before playback starts
    ratio_db = 20 * np.log10(np.median(er[talking]) / np.median(er[quiet]))
    corr, lag = max((np.corrcoef(ep[: n - L], er[L:])[0, 1], L) for L in range(0, int(0.8 * SR / fr)))
    print(f"\nmic level while it played vs before: {ratio_db:+.1f} dB   envelope match {corr:.2f} at {lag * fr / SR * 1000:.0f} ms delay")
    if corr > 0.4 and ratio_db > 6:
        print("ECHO: the mic hears your output. Calls will be flagged and skipped for training.\n"
              "Check the output above is your headphones, that they are in your ears, and lower the volume.")
        return 1
    print("OK: the mic does not pick up your output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(check_echo())
