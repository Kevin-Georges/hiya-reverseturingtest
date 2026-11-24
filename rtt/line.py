"""A simulated full-duplex phone line.

Two endpoints exchange 20 ms audio frames in real time, with a configurable one-way delay.
The line is the only channel between the parties: the screener never sees the bot's text,
only the audio it sends, exactly as on a real call.
"""
from __future__ import annotations

import collections
import queue
import threading
from dataclasses import dataclass, field

import numpy as np

from .audio import FRAME_S, Clock, silence, to_frames


@dataclass
class Utterance:
    """Something an endpoint plays onto the line. Times are line time in seconds."""

    text: str
    start_t: float | None = None
    end_t: float | None = None
    cut: bool = False  # stopped early (barge-in)
    done: threading.Event = field(default_factory=threading.Event)


class Endpoint:
    """One side of a call. The line pulls one outgoing frame and pushes one incoming frame per tick."""

    def __init__(self, name: str):
        """Empty outgoing queue of (utterance, frame, is_last), and an incoming queue that the line fills."""
        self.name = name
        self._out: collections.deque = collections.deque()
        self._lock = threading.Lock()
        self._current: Utterance | None = None
        self._last_pull_t = 0.0
        self.incoming: queue.Queue = queue.Queue()
        self.line: "PhoneLine | None" = None
        self.utterances: list[Utterance] = []

    # ---- called by the line thread -------------------------------------------------------
    def pull_frame(self, t: float) -> np.ndarray | None:
        """Line thread, every tick: hand over the next outgoing frame (None if we're silent).

        How: also stamps an utterance's real start time on its first frame, and its end time on its last frame
        (marking it done), so the screener knows exactly when its question finished on the line.
        """
        with self._lock:
            self._last_pull_t = t
            if not self._out:
                return None
            utt, frame, is_last = self._out.popleft()
            if utt.start_t is None:
                utt.start_t = t
                self._current = utt
            if is_last:
                utt.end_t = t + FRAME_S
                utt.done.set()
                self._current = None
            return frame

    def push_frame(self, frame: np.ndarray, t: float) -> None:
        """Line thread, every tick: deliver one incoming frame with its line time."""
        self.incoming.put((t, frame))

    # ---- called by the agent --------------------------------------------------------------
    def play(self, pcm: np.ndarray, text: str = "") -> Utterance:
        """Queue audio to send, frame by frame. Returns at once, with an Utterance whose times fill in as it plays."""
        utt = Utterance(text=text)
        frames = to_frames(pcm) or [silence()]
        with self._lock:
            for i, f in enumerate(frames):
                self._out.append((utt, f, i == len(frames) - 1))
            self.utterances.append(utt)
        return utt

    def stop_playback(self) -> Utterance | None:
        """Barge-in: drop everything still queued. Returns the utterance that was cut, if any.

        How: under the lock, note the utterance that is mid-play, clear the queue, mark it cut with its real end
        time, and mark any queued-but-unplayed utterances done too, so nothing waiting on them hangs.
        """
        with self._lock:
            cut = self._current
            pending = {id(u): u for u, _, _ in self._out}
            self._out.clear()
            if cut is not None:
                cut.end_t = self._last_pull_t + FRAME_S
                cut.cut = True
                cut.done.set()
                self._current = None
            for u in pending.values():
                if not u.done.is_set():
                    u.cut = True
                    u.done.set()
            return cut

    def is_playing(self) -> bool:
        """True while frames are still queued to send."""
        with self._lock:
            return bool(self._out)

    def now(self) -> float:
        """Current line time in seconds."""
        return self.line.now() if self.line else 0.0

    @property
    def clock(self) -> Clock:
        """The line's clock (sped up in fake runs)."""
        return self.line.clock if self.line else Clock()


class PhoneLine:
    """Runs the 50 Hz tick that moves audio between two endpoints and records both directions."""

    def __init__(
        self,
        a: Endpoint,
        b: Endpoint,
        one_way_latency_ms: float = 60.0,
        clock: Clock | None = None,
        monitor=None,
    ):
        """Connect two endpoints. The one-way delay becomes a queue of silent frames in each direction (60 ms = 3
        frames).
        """
        self.a, self.b = a, b
        a.line = b.line = self
        self.clock = clock or Clock()
        self.delay_frames = max(0, int(round(one_way_latency_ms / 1000.0 / FRAME_S)))
        self.monitor = monitor  # optional callable(frame_mix) for listening in
        self._a_to_b = collections.deque([silence()] * self.delay_frames)
        self._b_to_a = collections.deque([silence()] * self.delay_frames)
        self._idx = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="phone-line", daemon=True)
        self.rec_a: list[np.ndarray] = []  # audio as sent by a
        self.rec_b: list[np.ndarray] = []  # audio as sent by b

    @property
    def one_way_latency_s(self) -> float:
        """The actual one-way delay, rounded to whole frames."""
        return self.delay_frames * FRAME_S

    def now(self) -> float:
        """Line time: ticks so far x 20 ms."""
        return self._idx * FRAME_S

    def start(self) -> None:
        """Start the tick thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop the tick thread and wait for it (up to 2 s)."""
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        """The line itself: every 20 ms, move one frame each way.

        How: pull a frame from each side (silence if none), record both, push each into its delay queue and deliver
        the frame coming out of the other end. Each tick is scheduled from the start time (t0 + n x 20 ms) rather
        than 'sleep 20 ms', so a late tick is caught up instead of the line drifting.
        """
        t0 = self.clock.perf()
        tick = FRAME_S / self.clock.speed
        while not self._stop.is_set():
            t = self._idx * FRAME_S
            fa = self.a.pull_frame(t)
            fb = self.b.pull_frame(t)
            fa = silence() if fa is None else fa
            fb = silence() if fb is None else fb
            self.rec_a.append(fa)
            self.rec_b.append(fb)
            self._a_to_b.append(fa)
            self._b_to_a.append(fb)
            self.b.push_frame(self._a_to_b.popleft(), t)
            self.a.push_frame(self._b_to_a.popleft(), t)
            if self.monitor is not None:
                self.monitor(np.clip(fa + fb, -1, 1))
            self._idx += 1
            wait = t0 + self._idx * tick - self.clock.perf()
            if wait > 0:
                self._stop.wait(wait)

    def recording(self) -> np.ndarray:
        """Stereo array: left = endpoint a, right = endpoint b."""
        n = min(len(self.rec_a), len(self.rec_b))
        if n == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.stack([np.concatenate(self.rec_a[:n]), np.concatenate(self.rec_b[:n])], axis=1)
