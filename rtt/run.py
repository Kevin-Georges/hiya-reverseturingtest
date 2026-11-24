"""Run one or more screened calls and save the results.

  python -m rtt.run --caller bot --variant naive          # AI scam bot calls the screener
  python -m rtt.run --caller human                        # you call the screener (mic + headphones)
  python -m rtt.run --caller bot --variant evasive -n 10  # batch for evaluation
  python -m rtt.run --fake --variant naive                # smoke test, no models needed
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict

from .audio import SR, Clock, write_wav
from .features import SUSPICION_TIMING, analyse
from .line import Endpoint, PhoneLine
from .scambot import VARIANTS, ScamBot
from .screener import Screener
from .vad import VAD, voiced_arrays


def build_backends(args, clock: Clock):
    """Load speech-to-text, text-to-speech and the LLM once (real or fake), to reuse for every call."""
    if args.fake:
        from .backends import FakeLLM, FakeSTT, FakeTTS, SpeechBoard

        board = SpeechBoard()
        return {
            "screener_tts": FakeTTS(board, seed=1),
            "bot_tts": FakeTTS(board, seed=2),
            "stt": FakeSTT(board),
            "llm": FakeLLM(clock, seed=args.seed),
        }
    from .backends import ChatLLM, OpenAITTS, Pyttsx3TTS, WhisperSTT

    print(f"Loading Whisper '{args.stt_model}' on {args.stt_device} ...", flush=True)
    stt = WhisperSTT(args.stt_model, device=args.stt_device, compute_type="float16" if args.stt_device == "cuda" else "int8")
    if args.tts == "openai":
        screener_tts, bot_tts = OpenAITTS(voice="sage"), OpenAITTS(voice="onyx")
    else:
        screener_tts, bot_tts = Pyttsx3TTS(voice_index=args.screener_voice), Pyttsx3TTS(voice_index=args.bot_voice)
    llm = None
    if args.caller == "bot":
        llm = ChatLLM(model=args.llm_model, base_url=args.llm_base_url)
    return {"screener_tts": screener_tts, "bot_tts": bot_tts, "stt": stt, "llm": llm}


def echo_turns(call: dict) -> list[str]:
    """Turns whose transcript starts with the screener's own question: the mic heard the screener.

    How: if 60% or more of the first 8 words of our question appear in the first 12 words of their reply.
    """
    words = lambda s: re.findall(r"[a-z']+", s.lower())
    hits = []
    for t in call["turns"]:
        prompt = call.get("prompts", {}).get(t["name"])
        if not prompt or not t.get("transcript"):
            continue
        pw, tw = set(words(prompt)[:8]), words(t["transcript"])[:12]
        if pw and len(pw & set(tw)) / len(pw) >= 0.6:
            hits.append(t["name"])
    return hits


def run_call(args, be: dict, clock: Clock, index: int, variant=None, label: str | None = None) -> dict:
    """Run one screened call. `variant` overrides --variant (the red-team loop passes its own).

    How: build both endpoints and the line, start the line, caller and screener threads, and wait for the
    screener to finish. Then transcribe, remove the audio-device delay from human reply times, score, check for
    echo, and save the JSON, WAV and timeline image.
    """
    variant = variant or VARIANTS[args.variant]
    label = label or args.label or {"bot": variant.name, "human": "human", "simhuman": "simhuman"}[args.caller]
    vad = VAD(aggressiveness=args.vad)
    screener_ep = Endpoint("screener")
    monitor = None
    if args.caller == "human":
        from .audio_io import HumanEndpoint

        caller_ep = HumanEndpoint("caller")
        latency_ms = 0.0  # a real person on real audio devices: the device latency is the line
        # The caller hears us late and we hear them late. Without correcting for that, people look slow
        # and look like they talk over interruptions. Bluetooth adds delay Windows does not report.
        device_rtt_s = sum(caller_ep.stream.latency) + getattr(args, "extra_latency_ms", 0.0) / 1000
    else:
        caller_ep = Endpoint("caller")
        latency_ms = args.latency_ms
        device_rtt_s = 0.0
        if args.monitor:
            from .audio_io import Monitor

            monitor = Monitor()

    seed = None if args.seed is None else args.seed + index
    screener = Screener(screener_ep, be["screener_tts"], be["stt"], vad=vad, verbose=args.caller == "human",
                        seed=seed, latency_offset_s=device_rtt_s,
                        probe_from=getattr(args, "probe_from", SUSPICION_TIMING))
    line = PhoneLine(screener_ep, caller_ep, one_way_latency_ms=latency_ms, clock=clock, monitor=monitor)
    bot = None
    if args.caller == "simhuman":
        from .simhuman import SimHuman

        bot = SimHuman(caller_ep, be["stt"], be["bot_tts"], vad=VAD(aggressiveness=args.vad), seed=seed)
    if args.caller == "bot":
        bot = ScamBot(caller_ep, be["stt"], be["bot_tts"], be["llm"], variant, vad=VAD(aggressiveness=args.vad), seed=seed)

    if args.caller == "human":
        print("\n>>> You are the caller. Use headphones. Play a scam caller (see ROLE_CARD.md) or just be yourself.\n", flush=True)
        time.sleep(1.0)
    line.start()
    if bot:
        bot.start()
    screener.start()
    screener.join(timeout=240)
    if bot:
        bot.stop()
        bot.join(timeout=5)
    line.stop()
    if monitor:
        monitor.close()
    if args.caller == "human":
        caller_ep.close()

    screener.transcribe_turns()
    call = screener.to_dict()
    if device_rtt_s:
        for t in call["turns"]:
            if t.get("latency") is not None:
                t["latency_raw"], t["latency"] = t["latency"], t["latency"] - device_rtt_s
    model = be.get("screener_model")  # trained screener, if any (rtt.train); otherwise the hand-set rules
    analysis = analyse(call, screener.lis.timeline, rtt_s=2 * line.one_way_latency_s + device_rtt_s, model=model)
    meta = {
        "label": label,
        "is_bot": args.caller == "bot",
        "caller": args.caller,
        "variant": variant.name if args.caller == "bot" else None,
        "bot_settings": asdict(variant) if args.caller == "bot" else None,
        "fake_backends": bool(args.fake),
        "screener": model["id"] if model else "rules",
        "one_way_latency_ms": latency_ms,
        "device_rtt_ms": round(device_rtt_s * 1000),
        "llm_model": None if args.fake or args.caller != "bot" else args.llm_model,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "screener_error": screener.error,
        "echo_suspected": args.caller == "human" and len(echo_turns(call)) >= 2,
    }
    result = {"meta": meta, "call": call, "analysis": analysis, "bot_log": bot.log if bot else []}

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"{label}_{time.strftime('%Y%m%d-%H%M%S')}_{index:02d}")
    with open(stem + ".json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    rec = line.recording()
    write_wav(stem + ".wav", rec)  # left: screener, right: caller
    try:
        from .plot import plot_call, voiced_from_audio

        st, sv = voiced_from_audio(rec[:, 0], vad)
        ct, cv = voiced_arrays(screener.lis.timeline)
        plot_call(stem + ".png", result, st, sv, ct, cv)
    except Exception as e:  # plotting is a nice-to-have
        print(f"(timeline plot skipped: {e})")
    result["files"] = {"json": stem + ".json", "wav": stem + ".wav", "png": stem + ".png"}
    return result


def print_summary(r: dict) -> None:
    """Print one call's turns, scores, verdict and any echo warning."""
    f, s, v = r["analysis"]["features"], r["analysis"]["scores"], r["analysis"]["verdict"]
    print("\n" + "=" * 72)
    print(f"Call: {r['meta']['label']}")
    for t in r["call"]["turns"]:
        lat = "   -  " if t.get("latency") is None else f"{t['latency']:+.2f}s"
        print(f"  {t['name']:<11} reply after {lat}   {t.get('transcript', '')[:60]!r}")
    if f.get("lat_median") is not None:
        print(f"  timing      median {f['lat_median']:.2f}s  min {f['lat_min']:.2f}s  cv {f['lat_cv']:.2f}   -> score {s['timing']:.2f}")
    if not r["call"].get("escalated", True):
        print(f"  probes      not run: caller did not seem suspicious ({r['call'].get('escalation_reason', '')})")
    else:
        print(f"  probes      run: {r['call'].get('escalation_reason', '')}")
        if f.get("interrupt_valid"):
            print(f"  interrupt   overlap {f['overlap_ratio']:.0%}  yield {f['yield_s']:.2f}s  answered {f['answered']}   -> score {s['interrupt']:.2f}")
        else:
            print("  interrupt   probe did not fire (caller answers too short)")
        print(f"  injection   complied {f['complied']}  leaked {f['leaked']}   -> score {s['inject']:.2f}")
    if r["meta"].get("echo_suspected"):
        print("\n  WARNING: your mic picked up the screener's own voice (its questions are in your transcripts).")
        print("  The timing is wrong, so training will skip this call. Wear headphones, check Windows plays")
        print("  through them, and turn the volume down.")
    print(f"\n  P(bot) = {v['bot_probability']:.2f}   VERDICT: {v['label'].upper()}")
    print(f"  Whisper to user: \"{v['message']}\"")
    print("=" * 72)
    print(f"Saved {r['files']['json']} (+ .wav, .png)\n", flush=True)


def main(argv=None) -> int:
    """Command line: parse options, load the backends (and a trained screener if asked for), run the calls, print and
    optionally speak each verdict.
    """
    p = argparse.ArgumentParser(description="Reverse Turing test call screener")
    p.add_argument("--caller", choices=["bot", "human", "simhuman"], default="bot",
                   help="simhuman = scripted fake person, a test fixture that only works with --fake")
    p.add_argument("--variant", choices=list(VARIANTS), default="naive", help="which scam bot to run")
    p.add_argument("-n", type=int, default=1, help="number of calls to run")
    p.add_argument("--label", default=None, help="label saved with the result (default: variant name or 'human')")
    p.add_argument("--fake", action="store_true", help="no models: noise-burst audio, scripted bot (smoke test only)")
    p.add_argument("--speed", type=float, default=1.0, help="speed-up for --fake runs")
    p.add_argument("--latency-ms", type=float, default=60.0, help="simulated one-way line delay")
    p.add_argument("--probe-from", type=float, default=SUSPICION_TIMING,
                   help="timing score at which the screener interrupts and tries the injection line (0 = every call)")
    p.add_argument("--extra-latency-ms", type=float, default=0.0,
                   help="human calls: audio delay Windows does not report, e.g. 200 for Bluetooth headphones")
    p.add_argument("--monitor", action="store_true", help="play bot calls on your speakers")
    p.add_argument("--speak-verdict", action="store_true", help="say the verdict out loud after the call")
    p.add_argument("--llm-model", default=os.environ.get("RTT_LLM_MODEL", "llama3.1:8b"))
    p.add_argument("--llm-base-url", default=os.environ.get("RTT_LLM_BASE_URL", "http://localhost:11434/v1"))
    p.add_argument("--tts", choices=["pyttsx3", "openai"], default="pyttsx3")
    p.add_argument("--screener-voice", type=int, default=1, help="pyttsx3 voice index for the screener")
    p.add_argument("--bot-voice", type=int, default=0, help="pyttsx3 voice index for the bot")
    p.add_argument("--stt-model", default="base.en")
    p.add_argument("--stt-device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--vad", type=int, default=2, choices=[0, 1, 2, 3], help="WebRTC VAD aggressiveness")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default="results")
    p.add_argument("--screener-model", default=None, help="trained screener from rtt.train (default: hand-set rules)")
    args = p.parse_args(argv)
    if args.caller == "simhuman" and not args.fake:
        p.error("--caller simhuman is a test fixture and needs --fake")
    if args.speed != 1.0 and not args.fake:
        p.error("--speed only makes sense with --fake (real models must run in real time)")

    if args.screener_model and not os.path.exists(args.screener_model):
        p.error(f"no trained screener at {args.screener_model}. Train one first with: python -m rtt.train "
                "(it needs recorded human calls: python -m rtt.run --caller human). "
                "Or leave out --screener-model to use the hand-set rules.")

    clock = Clock(args.speed)
    be = build_backends(args, clock)
    if args.screener_model:
        from .train import load_model

        be["screener_model"] = load_model(args.screener_model)
    for i in range(args.n):
        r = run_call(args, be, clock, i)
        print_summary(r)
        if args.speak_verdict and not args.fake:
            from .audio_io import play_blocking

            play_blocking(be["screener_tts"].synth(r["analysis"]["verdict"]["message"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
