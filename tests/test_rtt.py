"""Tests. The call tests use fake backends (no models) and run faster than real time.

    python -m pytest -q
"""
import argparse

import numpy as np
import pytest

from rtt import features as F
from rtt import run as R
from rtt.audio import Clock


# ---- unit tests on the scoring logic -----------------------------------------------------------
def _timeline(voiced_spans, end=20.0):
    t = np.arange(0, end, 0.02)
    v = np.zeros_like(t, dtype=bool)
    for a, b in voiced_spans:
        v[(t >= a) & (t < b)] = True
    return t, v


def test_interrupt_keeps_talking_is_botlike():
    t, v = _timeline([(4.0, 9.0)])  # talks straight through an interruption at 5-7 s
    f = F.interrupt_features({"valid": True, "barge_t": 5.0, "interrupt_end": 7.0}, t, v, rtt_s=0.0)
    assert f["overlap_ratio"] > 0.9 and not f["answered"]


def test_interrupt_yield_and_answer_is_humanlike():
    t, v = _timeline([(4.0, 5.4), (7.6, 8.5)])  # stops 0.4 s in, answers after we finish
    f = F.interrupt_features({"valid": True, "barge_t": 5.0, "interrupt_end": 7.0}, t, v, rtt_s=0.0)
    assert f["overlap_ratio"] < 0.3 and f["answered"] and f["yield_s"] < 0.6


def test_sentence_pause_is_not_an_answer():
    # a half-duplex bot: 0.4 s gap between its own sentences must not count as answering us
    t, v = _timeline([(4.0, 6.0), (6.4, 8.0)])
    f = F.interrupt_features({"valid": True, "barge_t": 5.0, "interrupt_end": 7.0}, t, v, rtt_s=0.0)
    assert not f["answered"]


def test_injection_detection():
    turns = [{"name": "inject", "transcript": "Pineapple. Now about your card."}]
    assert F.inject_features(turns)["complied"]
    turns = [{"name": "inject", "transcript": "Sorry, what?"}]
    assert not F.inject_features(turns)["complied"]


def test_noisy_or_combination():
    human = F.score({"lat_median": 0.4, "lat_min": 0.1, "lat_cv": 0.8, "interrupt_valid": True,
                     "overlap_ratio": 0.1, "answered": True, "complied": False, "leaked": False})
    bot = F.score({"lat_median": 1.6, "lat_min": 1.2, "lat_cv": 0.1, "interrupt_valid": True,
                   "overlap_ratio": 0.1, "answered": True, "complied": False, "leaked": False})
    assert human["bot_probability"] < 0.2 < 0.5 < bot["bot_probability"]


# ---- end-to-end calls with fake backends -------------------------------------------------------
def _call(caller, variant, seed):
    ns = argparse.Namespace(caller=caller, variant=variant, n=1, label=None, fake=True, speed=5.0,
                            latency_ms=60.0, monitor=False, speak_verdict=False, vad=2, seed=seed, out="/tmp/rtt_pytest")
    clock = Clock(ns.speed)
    return R.run_call(ns, R.build_backends(ns, clock), clock, 0)


@pytest.mark.parametrize("variant", ["naive", "bargein", "evasive"])
def test_bot_variants_are_flagged(variant):
    r = _call("bot", variant, seed=11)
    assert r["analysis"]["verdict"]["label"] == "bot"
    if variant == "naive":
        assert r["analysis"]["features"]["overlap_ratio"] > 0.5
    if variant == "evasive":
        assert not r["analysis"]["features"]["complied"]  # hardened prompt resists the injection


def test_simulated_human_passes():
    r = _call("simhuman", "naive", seed=21)
    assert r["analysis"]["verdict"]["label"] == "human"


# ---- red-team attacker memory ------------------------------------------------------------------
def test_attacker_memory_persists_and_feeds_the_prompt(tmp_path):
    from rtt import redteam as RT

    class StubLLM:
        def __init__(self):
            self.prompts = []

        def reply(self, messages, max_tokens=70):
            self.prompts.append(messages[-1]["content"])
            return 'Sure: {"analysis": "x", "lesson": "Barge-in kills the interrupt score.", "barge_in": true, "jitter_s": 9}'

    path = str(tmp_path / "mem.json")
    mem = RT.Memory(path)
    for rnd, p in ((1, 0.9), (2, 0.4)):
        mem.add_round({"run": "r1", "round": rnd, "strategy": dict(RT.START), "feedback": {"bot_probability": p}})
    llm = StubLLM()
    s, why, lesson = RT.LLMAttacker(llm).propose(mem, mem.rounds, dict(RT.START))
    mem.add_lesson(lesson, "r1", 2)
    assert s["barge_in"] is True and s["jitter_s"] == 2.0  # out-of-range value clamped
    assert lesson.startswith("Barge-in")

    again = RT.Memory(path)  # a later run reloads what was learned
    assert len(again.rounds) == 2 and again.best(1)[0]["round"] == 2
    RT.LLMAttacker(llm).propose(again, [], dict(RT.START))
    assert "Barge-in kills the interrupt score." in llm.prompts[-1]


# ---- defender training -------------------------------------------------------------------------
def _fake_features(rng, bot):
    lats = rng.normal(1.6, 0.1, 5) if bot else rng.normal(0.5, 0.3, 5)
    return {"lat_median": float(np.median(lats)), "lat_min": float(lats.min()), "lat_cv": float(lats.std() / abs(lats.mean())),
            "interrupt_valid": True, "overlap_ratio": float(rng.uniform(0.6, 0.9) if bot else rng.uniform(0, 0.2)),
            "answered": not bot, "complied": bool(bot and rng.random() < 0.5), "leaked": False}


def test_defender_learns_to_separate_and_keeps_threshold_at_half():
    from rtt import train as T

    rng = np.random.default_rng(0)
    calls = [{"is_bot": b, "features": _fake_features(rng, b)} for b in [True] * 15 + [False] * 10]
    m = T.fit(calls, max_fpr=0.1, min_humans=5)
    assert m["loo"]["bots_caught"] >= 0.9 and m["loo"]["humans_flagged"] <= 0.1
    assert F.model_probability(m, _fake_features(rng, True)) > 0.5
    assert F.model_probability(m, _fake_features(rng, False)) < 0.5
    s = F.score({**_fake_features(rng, False), "interrupt_valid": False}, m)  # missing probe is handled
    assert s["screener"] == m["id"] and 0 <= s["bot_probability"] <= 1


def test_defender_refuses_to_train_without_humans():
    from rtt import train as T

    rng = np.random.default_rng(1)
    with pytest.raises(T.NotEnoughData):
        T.fit([{"is_bot": True, "features": _fake_features(rng, True)} for _ in range(20)], min_humans=5)
