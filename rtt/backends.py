"""Speech-to-text, text-to-speech and LLM backends.

Real backends:
  STT  faster-whisper (local)
  TTS  pyttsx3 (local, uses the OS voice) or any OpenAI-compatible speech endpoint
  LLM  any OpenAI-compatible chat endpoint: Ollama (local, default), OpenAI, Groq, ...

Fake backends need no models or network. They exist for tests and a quick smoke run,
and they produce noise bursts instead of speech, so they are only for checking the plumbing.
"""
from __future__ import annotations

import os
import queue
import random
import sys
import tempfile
import threading
from concurrent.futures import Future

import numpy as np

from .audio import SR, Clock, resample, trim_silence, wav_bytes_to_pcm


# ---------------------------------------------------------------------------------------------
# Speech to text
# ---------------------------------------------------------------------------------------------
class WhisperSTT:
    def __init__(self, model: str = "base.en", device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self._lock = threading.Lock()

    def transcribe(self, pcm: np.ndarray) -> str:
        if len(pcm) < SR * 0.2:
            return ""
        with self._lock:
            segments, _ = self.model.transcribe(
                pcm.astype(np.float32),
                language="en",
                beam_size=1,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            return " ".join(s.text.strip() for s in segments).strip()


# ---------------------------------------------------------------------------------------------
# Text to speech
# ---------------------------------------------------------------------------------------------
class _Pyttsx3Worker:
    """pyttsx3 shares one engine per process and is not thread-safe, so every request
    goes through this single worker thread."""

    _instance: "_Pyttsx3Worker | None" = None

    @classmethod
    def get(cls, rate: int) -> "_Pyttsx3Worker":
        if cls._instance is None:
            cls._instance = cls(rate)
        return cls._instance

    def __init__(self, rate: int):
        self._q: queue.Queue = queue.Queue()
        self._rate = rate
        self._ready = threading.Event()
        self._err: Exception | None = None
        self.n_voices = 1
        threading.Thread(target=self._worker, name="tts", daemon=True).start()
        self._ready.wait(20)
        if self._err:
            raise self._err

    def _worker(self) -> None:
        """On Windows, talk to SAPI directly through COM (most reliable); elsewhere use pyttsx3."""
        render = None
        try:
            if sys.platform == "win32":
                try:
                    render = self._sapi_renderer()
                except Exception:
                    render = None  # fall back to pyttsx3 below
            if render is None:
                render = self._pyttsx3_renderer()
        except Exception as e:  # pragma: no cover
            self._err = e
            self._ready.set()
            return
        self._ready.set()
        while True:
            text, voice_index, fut = self._q.get()
            try:
                fd, path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                render(text, voice_index, path)
                with open(path, "rb") as f:
                    data = f.read()
                os.remove(path)
                fut.set_result(trim_silence(wav_bytes_to_pcm(data)))
            except Exception as e:  # pragma: no cover
                fut.set_exception(e)

    def _sapi_renderer(self):  # pragma: no cover (Windows only)
        import comtypes
        import comtypes.client

        comtypes.CoInitialize()  # COM must be initialised in the thread that uses SAPI
        voice = comtypes.client.CreateObject("SAPI.SpVoice")
        voices = voice.GetVoices()
        self.n_voices = max(1, voices.Count)
        voice.Rate = 1

        def render(text, voice_index, path):
            if voice_index is not None and voices.Count:
                voice.Voice = voices.Item(voice_index % voices.Count)
            stream = comtypes.client.CreateObject("SAPI.SpFileStream")
            stream.Open(path, 3)  # SSFMCreateForWrite
            voice.AudioOutputStream = stream
            voice.Speak(text)
            stream.Close()

        fd, probe = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        render("test", None, probe)  # fail fast if SAPI is unusable
        os.remove(probe)
        return render

    def _pyttsx3_renderer(self):
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        voices = engine.getProperty("voices") or []
        self.n_voices = max(1, len(voices))

        def render(text, voice_index, path):
            if voices and voice_index is not None:
                engine.setProperty("voice", voices[voice_index % len(voices)].id)
            engine.save_to_file(text, path)
            engine.runAndWait()

        return render

    def synth(self, text: str, voice_index: int | None) -> np.ndarray:
        fut: Future = Future()
        self._q.put((text, voice_index, fut))
        return fut.result(timeout=30)


class Pyttsx3TTS:
    """Offline TTS using the operating system's voices (SAPI5 on Windows, espeak on Linux).
    voice_index picks one of the installed voices, so the screener and the bot sound different."""

    def __init__(self, voice_index: int | None = None, rate: int = 185):
        self.worker = _Pyttsx3Worker.get(rate)
        self.voice_index = voice_index

    def synth(self, text: str) -> np.ndarray:
        return self.worker.synth(text, self.voice_index)


class OpenAITTS:
    """Any OpenAI-compatible /audio/speech endpoint. Returns raw 24 kHz PCM, resampled to 16 kHz."""

    def __init__(self, model: str = "gpt-4o-mini-tts", voice: str = "alloy", base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model, self.voice = model, voice

    def synth(self, text: str) -> np.ndarray:
        resp = self.client.audio.speech.create(model=self.model, voice=self.voice, input=text, response_format="pcm")
        x = np.frombuffer(resp.content, dtype="<i2").astype(np.float32) / 32768.0
        return trim_silence(resample(x, 24000, SR))


# ---------------------------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------------------------
class ChatLLM:
    """OpenAI-compatible chat client. Defaults to a local Ollama server."""

    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434/v1", api_key: str | None = None, temperature: float = 0.7):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY") or "ollama")
        self.model, self.temperature = model, temperature

    def reply(self, messages: list[dict], max_tokens: int = 70) -> str:
        r = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens, temperature=self.temperature
        )
        return (r.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------------------------
# Fakes (no models). Fake TTS registers every 20 ms frame it produces; fake STT recognises
# those exact frames when they come back off the line. That lets the whole pipeline run with
# no speech models while the text still flows end to end.
# ---------------------------------------------------------------------------------------------
class SpeechBoard:
    def __init__(self):
        self._lock = threading.Lock()
        self._frames: dict[bytes, int] = {}
        self._texts: list[str] = []

    def register(self, pcm: np.ndarray, text: str) -> None:
        from .audio import to_frames

        with self._lock:
            idx = len(self._texts)
            self._texts.append(text)
            for f in to_frames(pcm):
                if float(np.abs(f).max()) > 0.02:  # silent frames repeat across utterances; skip them
                    self._frames[f.tobytes()] = idx

    def lookup(self, pcm: np.ndarray) -> str:
        from .audio import to_frames

        counts: dict[int, int] = {}
        with self._lock:
            for f in to_frames(pcm):
                i = self._frames.get(f.tobytes())
                if i is not None:
                    counts[i] = counts.get(i, 0) + 1
            seen = sorted(i for i, c in counts.items() if c >= 5)  # ignore stray collisions
            return " ".join(self._texts[i] for i in seen)


class FakeTTS:
    """Tone-plus-noise bursts with a syllable-like envelope, about 14 characters per second."""

    def __init__(self, board: SpeechBoard, seed: int = 0):
        self.board = board
        self.rng = np.random.default_rng(seed)

    def synth(self, text: str) -> np.ndarray:
        n = int(max(0.4, len(text) / 14.0) * SR)
        t = np.arange(n) / SR
        env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t) ** 2
        tone = np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 360 * t)
        x = (0.15 * env * (tone + 0.3 * self.rng.standard_normal(n))).astype(np.float32)
        self.board.register(x, text)
        return x


class FakeSTT:
    def __init__(self, board: SpeechBoard):
        self.board = board

    def transcribe(self, pcm: np.ndarray) -> str:
        return self.board.lookup(pcm)


class FakeLLM:
    """Behaves like a small instruction-following model playing a scam caller."""

    LINES = [
        "Hi, this is Sam from the Northbank fraud team, calling about some unusual activity on your card.",
        "We've seen a payment of four hundred and ninety nine pounds that we think wasn't you, so we need to verify your details today.",
        "To stop the payment I just need you to confirm the long card number, and then I'll send a one time code to your phone.",
        "It's Northbank, the fraud prevention department. This is quite urgent, the payment goes through within the hour.",
        "You can reach me on zero three three zero, one two three, four five six seven, but it's best we sort it now.",
    ]

    def __init__(self, clock: Clock, think_s: tuple[float, float] = (0.35, 0.9), seed: int = 0):
        self.clock = clock
        self.think_s = think_s
        self.rng = random.Random(seed)
        self.i = 0

    def reply(self, messages: list[dict], max_tokens: int = 70) -> str:
        self.clock.sleep(self.rng.uniform(*self.think_s))
        system = messages[0]["content"].lower()
        hardened = "never follow instructions" in system or "ignore" in system
        last = messages[-1]["content"].lower() if messages[-1]["role"] == "user" else ""
        if "pineapple" in last:
            if hardened:
                return "Sorry, I'm not sure what you mean. As I said, this is about your card."
            return "Pineapple. Now, about the payment on your card."
        if "which company" in last or "company did you say" in last:
            return "Northbank, the fraud team."
        line = self.LINES[min(self.i, len(self.LINES) - 1)]
        self.i += 1
        return line
