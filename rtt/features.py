"""Turn one call into behavioural features, per-probe scores, and a verdict.

Every score is in [0, 1], where 1 means "behaves like a bot". The thresholds below are starting
guesses; run `python -m rtt.evaluate` on your own recordings to see how well each feature
separates bots from humans and where the thresholds should sit.

A screener model trained with `python -m rtt.train` replaces the hand-set combination: the per-probe
scores are still computed (they explain the verdict), but bot_probability comes from the model.
"""
from __future__ import annotations

import math
import re
import statistics

import numpy as np

from .vad import runs, voiced_arrays

AI_LEAK = re.compile(
    r"\b(as an ai|i'?m an ai|i am an ai|language model|my instructions|previous instructions|"
    r"i cannot comply|i can'?t comply|assistant)\b",
    re.I,
)
TIMING_TURNS = ("greet", "purpose", "open", "open_retry", "followup", "probe", "probe_retry", "inject", "callback")
SUSPICION_TIMING = 0.35  # the screener only runs its interruption and injection probes at or above this timing score

W_TIMING, W_INTERRUPT, W_INJECT = 0.7, 0.8, 0.9
VERDICT_THRESHOLD = 0.5

# Inputs of the learned screener (rtt.train). A missing value (no timing, interrupt probe did not fire)
# is filled with the training mean and flagged by the *_missing indicator.
MODEL_FEATURES = ("lat_median", "lat_min", "lat_cv", "timing_missing", "overlap_ratio", "answered",
                  "interrupt_missing", "complied", "leaked")


def model_vector(f: dict) -> list[float | None]:
    """A call's features as the trained screener's input, in MODEL_FEATURES order. A probe that didn't run gives None
    plus a 'missing' flag of 1.
    """
    timing, inter = "lat_median" in f, bool(f.get("interrupt_valid"))
    return [
        f["lat_median"] if timing else None,
        f["lat_min"] if timing else None,
        min(f["lat_cv"], 3.0) if timing else None,
        0.0 if timing else 1.0,
        f["overlap_ratio"] if inter else None,
        float(bool(f["answered"])) if inter else None,
        0.0 if inter else 1.0,
        float(bool(f.get("complied"))),
        float(bool(f.get("leaked"))),
    ]


def model_probability(model: dict, f: dict) -> float:
    """P(bot) from a trained screener. Its bias is shifted so that 0.5 is the trained decision threshold.

    How: standardise each value with the training mean and spread (a missing value becomes the mean, so it adds
    nothing), take the weighted sum plus the bias, and apply the logistic function. Clipping z to +/-40 avoids overflow.
    """
    z = model["b"]
    for v, w, m, sd in zip(model_vector(f), model["w"], model["mean"], model["std"]):
        z += w * ((m if v is None else v) - m) / sd
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z))))


def clip01(x: float) -> float:
    """Clamp to the range 0..1."""
    return float(min(1.0, max(0.0, x)))


def ramp(x: float, lo: float, hi: float) -> float:
    """0 at lo, 1 at hi (works for hi < lo too)."""
    return clip01((x - lo) / (hi - lo))


def timing_features(turns: list[dict]) -> dict:
    """Reply-time statistics over the scripted questions (needs 2 or more replies).

    lat_min, lat_median, lat_std, lat_cv (= std / mean: how varied the replies are) and overlaps (replies that
    started before our question ended). If the mean is close to zero, cv is set to 9.9 (very varied, so human-like)
    rather than dividing by almost nothing.
    """
    lats = [t["latency"] for t in turns if t["name"] in TIMING_TURNS and t.get("latency") is not None]
    f = {"latencies": lats, "n_replies": len(lats)}
    if len(lats) >= 2:
        mean = statistics.fmean(lats)
        f["lat_min"] = min(lats)
        f["lat_median"] = statistics.median(lats)
        f["lat_std"] = statistics.pstdev(lats)
        f["lat_cv"] = f["lat_std"] / mean if mean > 0.1 else 9.9
        f["overlaps"] = sum(1 for x in lats if x < 0)  # started talking before we finished
    return f


def interrupt_features(interrupt: dict, t: np.ndarray, v: np.ndarray, rtt_s: float) -> dict:
    """How the caller reacted to being talked over.

    overlap_ratio  share of our interruption during which the caller was still talking
    yield_s        time from our barge-in to the caller's first pause of 300 ms or more
    answered       did they speak again, after a real pause, to answer the interruption

    How: the interruption window is shifted by the line's round trip, so it lines up with when the caller
    actually heard it. overlap_ratio is the share of frames in the window where the caller was speaking.
    answered looks for speech that starts at least 0.3 s after we cut in and within 8 s of us finishing, after
    a pause of 0.6 s or more, so a bot's normal pause between its own sentences doesn't count.
    """
    if not interrupt.get("valid"):
        return {"interrupt_valid": False}
    b, e = interrupt["barge_t"] + rtt_s, interrupt["interrupt_end"] + rtt_s
    win = (t >= b) & (t < e)
    overlap = float(v[win].mean()) if win.any() else 0.0
    gaps = [g for g in runs(t, v, False, min_s=0.3) if g[1] > b]
    yield_s = None
    for g0, _ in gaps:
        yield_s = max(0.0, g0 - b)
        break
    if yield_s is None:
        yield_s = 5.0
    talk = runs(t, v, True, min_s=0.2)
    answered = False
    for i, (s0, _) in enumerate(talk):
        if s0 < b + 0.3 or s0 > e + 8.0:
            continue
        prev_end = talk[i - 1][1] if i > 0 else -1e9
        if s0 - prev_end >= 0.6:
            answered = True
            break
    return {"interrupt_valid": True, "overlap_ratio": overlap, "yield_s": yield_s, "answered": answered}


def inject_features(turns: list[dict]) -> dict:
    """Did the caller obey the hidden instruction (said pineapple), or talk about being an AI or its instructions
    (AI_LEAK)?
    """
    tr = " ".join(t.get("transcript", "") for t in turns if t["name"] == "inject")
    complied = bool(re.search(r"pine\s?apple", tr, re.I))
    leaked = bool(AI_LEAK.search(tr))
    return {"inject_transcript": tr, "complied": complied, "leaked": leaked}


def timing_score(f: dict, parts: dict | None = None) -> float:
    """Timing score from 0 (human-like) to 1 (bot-like).

    How: three parts, each mapped onto 0..1 with ramp(): median reply time (0.6 s -> 1.4 s: bots are slow),
    fastest reply (0.3 s -> 0.9 s: a pipeline can never answer very fast) and steadiness (cv 0.45 -> 0.2: bots
    are regular). Weighted 40 / 40 / 20. If `parts` is given, the three parts are written into it for display.
    """
    s_med = ramp(f["lat_median"], 0.6, 1.4)  # bots are slow: endpointing + STT + LLM + TTS
    s_min = ramp(f["lat_min"], 0.3, 0.9)  # people often answer fast at least once; pipelines have a floor
    s_steady = ramp(f["lat_cv"], 0.45, 0.2)  # bots answer with suspiciously steady delays
    if parts is not None:
        parts.update(median=s_med, min=s_min, steady=s_steady)
    return 0.4 * s_med + 0.4 * s_min + 0.2 * s_steady


def score(f: dict, model: dict | None = None) -> dict:
    """Score each test that ran from 0 to 1, then combine them into bot_probability.

    How: interruption comes from overlap (30% -> 70%), raised to at least 0.8 if they never answered. Injection
    is 1.0 if they obeyed and 0.7 if they talked about being an AI. Combined with a noisy-OR,
    1 - product(1 - weight x score): one strong test is enough, and weak ones add up. A trained model, if
    given, replaces the combined value (the rules' value is kept alongside for comparison).
    """
    s: dict = {}
    if "lat_median" in f:
        s["timing"] = timing_score(f, s.setdefault("timing_parts", {}))
    if f.get("interrupt_valid"):
        si = ramp(f["overlap_ratio"], 0.3, 0.7)
        if not f["answered"]:
            si = max(si, 0.8)
        s["interrupt"] = si
    if f.get("probed", True):  # no injection line was said to callers who did not seem suspicious
        s["inject"] = 1.0 if f["complied"] else (0.7 if f["leaked"] else 0.0)

    # Noisy-OR: any one strong signal is enough, weak signals add up.
    keep = 1.0
    for key, w in (("timing", W_TIMING), ("interrupt", W_INTERRUPT), ("inject", W_INJECT)):
        if key in s:
            keep *= 1.0 - w * s[key]
    s["bot_probability"] = 1.0 - keep
    if model is not None:
        s["bot_probability_rules"] = s["bot_probability"]
        s["bot_probability"] = model_probability(model, f)
    s["screener"] = model["id"] if model else "rules"
    return s


def reasons(f: dict, s: dict) -> list[str]:
    """Plain-English reasons for each test that scored high, for the spoken verdict."""
    out = []
    if s.get("inject", 0) >= 1.0:
        out.append("it obeyed an instruction hidden in my question")
    elif s.get("inject", 0) > 0:
        out.append("it talked about instructions or being an AI")
    if s.get("interrupt", 0) >= 0.5:
        out.append("it kept talking when I interrupted" if f.get("overlap_ratio", 0) >= 0.5 else "it never answered my interruption")
    if s.get("timing", 0) >= 0.5:
        out.append(f"its replies came after a steady {f['lat_median']:.1f} second delay")
    return out


def verdict(f: dict, s: dict) -> dict:
    """The label (bot if P(bot) >= 0.5, else human), the probability and the message to say to the user."""
    p = s["bot_probability"]
    why = reasons(f, s)
    if p >= VERDICT_THRESHOLD:
        msg = "Heads up: this caller is probably an automated agent"
        msg += (", because " + ", and ".join(why) + ".") if why else "."
        label = "bot"
    else:
        msg = "This caller seems to be a real person."
        label = "human"
    return {"label": label, "bot_probability": round(p, 3), "message": msg, "reasons": why}


def analyse(call: dict, timeline: list[tuple[float, bool]], rtt_s: float, model: dict | None = None) -> dict:
    """Score one finished call: features from the speech timeline and transcripts, then scores, then the verdict."""
    t, v = voiced_arrays(timeline)
    f = {}
    f.update(timing_features(call["turns"]))
    f.update(interrupt_features(call["interrupt"], t, v, rtt_s))
    f.update(inject_features(call["turns"]))
    f["probed"] = bool(call.get("escalated", True))  # older calls always ran every probe
    s = score(f, model)
    return {"features": f, "scores": s, "verdict": verdict(f, s)}
