"""Voice activity detection and a streaming segmenter that turns frames into speech segments."""
from __future__ import annotations

import collections
import queue
from dataclasses import dataclass

import numpy as np

from .audio import FRAME_S, SR


class VAD:
    """WebRTC VAD with an energy gate in front of it (falls back to energy only if not installed)."""

    def __init__(self, aggressiveness: int = 2, energy_gate: float = 0.004):
        """aggressiveness 0-3 (higher ignores more noise); a frame quieter than energy_gate is never speech."""
        self.energy_gate = energy_gate
        try:
            import webrtcvad

            self._vad = webrtcvad.Vad(aggressiveness)
        except ImportError:  # pragma: no cover
            self._vad = None

    def is_speech(self, frame: np.ndarray) -> bool:
        """Is this 20 ms frame speech? Too quiet: no. Otherwise ask WebRTC VAD, or use a loudness threshold if it
        isn't installed.
        """
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        if rms < self.energy_gate:
            return False
        if self._vad is None:
            return rms > 0.02
        pcm = (np.clip(frame, -1, 1) * 32767).astype("<i2").tobytes()
        return self._vad.is_speech(pcm, SR)


@dataclass
class Segment:
    """One stretch of speech: start, end and its audio."""
    start: float  # line time of first voiced frame
    end: float  # line time just after last voiced frame
    audio: np.ndarray


class Listener:
    """Consumes an endpoint's incoming frames, keeps a full voiced/unvoiced timeline,
    and emits ('start', t) and ('end', Segment) events."""

    def __init__(self, endpoint, vad: VAD, start_ms: int = 60, end_silence_ms: int = 800, preroll_ms: int = 200):
        """start_ms of speech starts a segment, end_silence_ms of silence ends it, and preroll_ms of audio before the
        start is kept.
        """
        self.ep = endpoint
        self.vad = vad
        self.start_frames = max(1, int(start_ms / 1000 / FRAME_S))
        self.end_frames = max(1, int(end_silence_ms / 1000 / FRAME_S))
        self._pre: collections.deque = collections.deque(maxlen=max(self.start_frames, int(preroll_ms / 1000 / FRAME_S)))
        self.timeline: list[tuple[float, bool]] = []
        self.segments: list[Segment] = []
        self.latest_t = 0.0
        self.reset()

    def reset(self) -> None:
        """Forget any speech in progress (used by half-duplex bots that ignore what they hear while talking)."""
        self.in_speech = False
        self.seg_start: float | None = None
        self.last_voiced: float | None = None
        self._buf: list[np.ndarray] = []
        self._run = 0
        self._silence = 0
        self._pre.clear()

    def speech_so_far(self) -> float:
        """How long the current speech has lasted, in seconds (0 if nobody is talking)."""
        if not self.in_speech or self.seg_start is None or self.last_voiced is None:
            return 0.0
        return self.last_voiced + FRAME_S - self.seg_start

    def step(self, timeout: float = 0.05) -> list:
        """Process every frame that has arrived; return ('start', t) and ('end', Segment) events.

        How: a two-state machine. Not in speech: keep a short buffer of recent frames and count voiced frames in a
        row; after start_ms of them, start a segment (backdated to its first voiced frame, buffered frames included).
        In speech: collect frames; after end_silence_ms of silence, emit the segment with trailing silence trimmed.
        Every frame's speech yes/no is also added to `timeline` for scoring later.
        """
        events: list = []
        try:
            items = [self.ep.incoming.get(timeout=timeout)]
        except queue.Empty:
            return events
        while True:
            try:
                items.append(self.ep.incoming.get_nowait())
            except queue.Empty:
                break
        for t, frame in items:
            self.latest_t = t
            v = self.vad.is_speech(frame)
            self.timeline.append((t, v))
            if not self.in_speech:
                self._pre.append(frame)
                self._run = self._run + 1 if v else 0
                if self._run >= self.start_frames:
                    self.in_speech = True
                    self.seg_start = t - (self.start_frames - 1) * FRAME_S
                    self.last_voiced = t
                    self._silence = 0
                    self._buf = list(self._pre)
                    self._pre.clear()
                    events.append(("start", self.seg_start))
            else:
                self._buf.append(frame)
                if v:
                    self.last_voiced = t
                    self._silence = 0
                else:
                    self._silence += 1
                    if self._silence >= self.end_frames:
                        keep = len(self._buf) - self._silence + 2  # trim trailing silence, keep 40 ms
                        seg = Segment(self.seg_start, self.last_voiced + FRAME_S, np.concatenate(self._buf[:keep]))
                        events.append(("end", seg))
                        self.segments.append(seg)
                        self.in_speech = False
                        self.seg_start = None
                        self._buf = []
                        self._run = 0
        return events


def voiced_arrays(timeline: list[tuple[float, bool]]) -> tuple[np.ndarray, np.ndarray]:
    """Split a timeline into numpy arrays of frame times and speech flags."""
    if not timeline:
        return np.zeros(0), np.zeros(0, dtype=bool)
    t = np.array([x[0] for x in timeline])
    v = np.array([x[1] for x in timeline], dtype=bool)
    return t, v


def runs(t: np.ndarray, v: np.ndarray, value: bool, min_s: float = 0.0) -> list[tuple[float, float]]:
    """Contiguous runs where v == value, as (start, end) in seconds, at least min_s long."""
    out = []
    start = None
    for ti, vi in zip(t, v):
        if vi == value and start is None:
            start = ti
        elif vi != value and start is not None:
            if ti - start >= min_s:
                out.append((start, ti))
            start = None
    if start is not None and len(t) and t[-1] + FRAME_S - start >= min_s:
        out.append((start, t[-1] + FRAME_S))
    return out
