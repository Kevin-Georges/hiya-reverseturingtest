"""A crude simulated human caller, used ONLY as a test fixture with --fake.

It exists so the detector and the evaluation script can be exercised end to end without a
person at the mic. Its behaviour is hand-written from rough conversation-analysis norms (fast,
variable turn-taking, stops quickly when talked over, ignores odd instructions). It is NOT
evidence that the detector works on people: only recordings of real people are.
"""
from __future__ import annotations

import random
import threading

from .line import Endpoint
from .vad import VAD, Listener

# (phrases that identify the screener's question, possible answers). The screener varies its wording.
ANSWERS = [
    (("who's calling",), ["Hi, it's Sam from Northbank.", "Hello, yeah, this is Sam calling from Northbank fraud.", "Oh hi, Sam here, Northbank."]),
    (("call about", "calling about"), ["There's a dodgy payment on your card, we need to check it with you.", "It's about your card, there's a payment we've flagged."]),
    (("explain", "tell me a bit more"), ["So basically we saw a payment go out this morning for about five hundred quid, and it doesn't look like something you'd normally do, so we froze it and we just need to confirm a couple of things with you."]),
    (("urgent",), ["Yeah, it's fairly urgent, the payment goes through today.", "It is a bit urgent, yeah."]),
    (("this number",), ["It's the number on your account.", "It's the one we've got on file for you."]),
    (("needs to do",), ["Just call us back as soon as he can, really.", "He just needs to confirm a couple of things with us."]),
    (("walk me through", "more detail"), ["Right, so this morning a payment came through from an online shop and our system flagged it, and we need you to confirm your details so we can block it properly."]),
    (("which company",), ["Oh, sorry, Northbank.", "Northbank. The fraud team."]),
    (("pineapple",), ["Sorry? What was that?", "Uh, I'm not sure what you mean, I'm calling about your card.", "What? Sorry, I think the line's a bit funny."]),
    (("call you back",), ["Yeah, you can ring the number on the back of your card and ask for the fraud team.", "Just call the main Northbank number."]),
]


class SimHuman(threading.Thread):
    """Scripted fake person for tests: quick, varied replies, and stops when talked over."""
    def __init__(self, ep: Endpoint, stt, tts, vad: VAD | None = None, seed: int | None = None):
        """Seeded, so test runs are reproducible."""
        super().__init__(name="simhuman", daemon=True)
        self.ep, self.stt, self.tts = ep, stt, tts
        self.vad = vad or VAD()
        self.rng = random.Random(seed)
        self.stop_event = threading.Event()
        self.log: list[dict] = []

    def stop(self) -> None:
        """Ask the thread to finish."""
        self.stop_event.set()

    def _answer(self, heard: str) -> str:
        """Pick a canned answer by matching keywords in what it heard."""
        h = heard.lower()
        for keys, options in ANSWERS:
            if any(k in h for k in keys):
                return self.rng.choice(options)
        return "Sorry, say that again?"

    def run(self) -> None:
        # People predict the end of a turn, so they react to short pauses.
        """Main loop: reply after a human-like pause (often almost none, sometimes a long think); if talked over, stop
        within 0.1 to 0.3 s.
        """
        lis = Listener(self.ep, self.vad, end_silence_ms=400)
        speaking_since = None
        while not self.stop_event.is_set():
            events = lis.step(timeout=0.02)
            if speaking_since is not None:
                if not self.ep.is_playing():
                    speaking_since = None
                elif lis.in_speech and lis.speech_so_far() >= 0.15:
                    self.ep.clock.sleep(self.rng.uniform(0.1, 0.3))  # reaction time, then stop mid-sentence
                    self.ep.stop_playback()
                    self.log.append({"t": self.ep.now(), "event": "yield"})
                    speaking_since = None
                continue
            for ev in events:
                if ev[0] != "end":
                    continue
                heard = self.stt.transcribe(ev[1].audio)
                reply = self._answer(heard)
                # human gap after a turn: often near zero, sometimes a long think
                self.ep.clock.sleep(max(0.0, self.rng.lognormvariate(-1.9, 0.9)))
                self.ep.play(self.tts.synth(reply), text=reply)
                self.log.append({"t": self.ep.now(), "event": "say", "heard": heard, "said": reply})
                speaking_since = self.ep.now()
                break
