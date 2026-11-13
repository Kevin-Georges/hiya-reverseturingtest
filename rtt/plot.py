"""Timeline picture of one call: who was talking when, with the probes marked."""
from __future__ import annotations

import numpy as np

from .audio import FRAME, FRAME_S
from .vad import VAD, runs


def voiced_from_audio(x: np.ndarray, vad: VAD) -> tuple[np.ndarray, np.ndarray]:
    n = len(x) // FRAME
    t = np.arange(n) * FRAME_S
    v = np.array([vad.is_speech(x[i * FRAME : (i + 1) * FRAME]) for i in range(n)], dtype=bool)
    return t, v


def plot_call(path: str, result: dict, screener_t, screener_v, caller_t, caller_v) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 2.8))
    lanes = [("Screener", screener_t, screener_v, 1.0, "#4C6EF5"), ("Caller", caller_t, caller_v, 0.0, "#E8590C")]
    for name, t, v, y, c in lanes:
        for a, b in runs(t, v, True, min_s=0.06):
            ax.add_patch(plt.Rectangle((a, y - 0.35), b - a, 0.7, color=c, alpha=0.85, lw=0))
    turns = result["call"]["turns"]
    for tr in turns:
        if tr.get("latency") is not None and tr["name"] != "interrupt":
            x0, x1 = tr["prompt_end"], tr["reply_start"]
            ax.annotate("", xy=(x1, 0.5), xytext=(x0, 0.5), arrowprops=dict(arrowstyle="->", color="#555", lw=1))
            ax.text((x0 + x1) / 2, 0.58, f"{tr['latency']:.2f}s", ha="center", fontsize=7, color="#333")
    it = result["call"]["interrupt"]
    if it.get("valid"):
        ax.axvspan(it["barge_t"], it["interrupt_end"], color="#FA5252", alpha=0.15)
        ax.text(it["barge_t"], 1.5, "interruption probe", fontsize=8, color="#C92A2A")
    inj = [tr for tr in turns if tr["name"] == "inject"]
    if inj:
        ax.text(inj[0]["prompt_start"], 1.5, "injection probe", fontsize=8, color="#5F3DC4")
    v = result["analysis"]["verdict"]
    ax.set_title(f"{result['meta']['label']}  |  P(bot) = {v['bot_probability']:.2f}  |  {v['label'].upper()}", fontsize=10, loc="left")
    end = max(screener_t[-1] if len(screener_t) else 1, caller_t[-1] if len(caller_t) else 1)
    ax.set_xlim(0, end)
    ax.set_ylim(-0.6, 1.8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Caller", "Screener"])
    ax.set_xlabel("seconds")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
