"""Aggregate saved calls: how well does each probe separate bots from people?

  python -m rtt.evaluate results/            # real calls only
  python -m rtt.evaluate results/ --include-fake

Writes results/report.md and results/report.png.
AUC = probability that a random bot call scores higher than a random human call
(1.0 = perfect separation, 0.5 = coin flip).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from .features import VERDICT_THRESHOLD

# (name, getter, higher_means_bot)
FEATURES = [
    ("median reply delay (s)", lambda r: r["features"].get("lat_median"), True),
    ("fastest reply (s)", lambda r: r["features"].get("lat_min"), True),
    ("delay variability (CV)", lambda r: r["features"].get("lat_cv"), False),
    ("talked over interruption (%)", lambda r: r["features"].get("overlap_ratio"), True),
    ("answered interruption", lambda r: _b(r["features"].get("answered")), False),
    ("obeyed injection", lambda r: _b(r["features"].get("complied")), True),
    ("timing score", lambda r: r["scores"].get("timing"), True),
    ("interrupt score", lambda r: r["scores"].get("interrupt"), True),
    ("injection score", lambda r: r["scores"].get("inject"), True),
    ("combined P(bot)", lambda r: r["scores"].get("bot_probability"), True),
]
PROBES = [("timing", "timing"), ("interruption", "interrupt"), ("injection", "inject"), ("combined", "bot_probability")]


def _b(x):
    return None if x is None else float(bool(x))


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Mann-Whitney AUC with ties counted as half."""
    if not pos or not neg:
        return None
    p, n = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def load(paths: list[str], include_fake: bool) -> list[dict]:
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p]
    out = []
    for f in files:
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if "meta" not in r or "analysis" not in r:
            continue
        if r["meta"].get("fake_backends") and not include_fake:
            continue
        a = r["analysis"]
        out.append({"label": r["meta"]["label"], "is_bot": r["meta"]["is_bot"], "features": a["features"], "scores": a["scores"], "file": f})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["results"])
    ap.add_argument("--include-fake", action="store_true", help="include --fake runs (plumbing check, not evidence)")
    ap.add_argument("--out", default=None, help="where to write report.md / report.png (default: first path)")
    args = ap.parse_args(argv)
    rows = load(args.paths, args.include_fake)
    if not rows:
        print("No results found. Run some calls first (python -m rtt.run ...).")
        return 1
    out_dir = args.out or (args.paths[0] if os.path.isdir(args.paths[0]) else os.path.dirname(args.paths[0]))
    bots = [r for r in rows if r["is_bot"]]
    humans = [r for r in rows if not r["is_bot"]]
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[r["label"]].append(r)

    lines = ["# Reverse Turing test: evaluation", ""]
    if args.include_fake:
        lines += ["> Includes `--fake` runs (noise-burst audio, scripted callers). These check the plumbing; they are not evidence.", ""]
    lines += [f"Calls: {len(rows)} ({len(bots)} bot, {len(humans)} human). Groups: " + ", ".join(f"{k} ({len(v)})" for k, v in sorted(groups.items())), ""]

    # Per-feature separation
    lines += ["## How well each feature separates bots from humans", "", "| Feature | " + " | ".join(f"{k} mean" for k in sorted(groups)) + " | AUC (all bots vs humans) |", "|---|" + "---|" * len(groups) + "---|"]
    for name, get, higher_bot in FEATURES:
        means = []
        for k in sorted(groups):
            vals = [get(r) for r in groups[k] if get(r) is not None]
            means.append(f"{np.mean(vals):.2f}" if vals else "-")
        pos = [get(r) for r in bots if get(r) is not None]
        neg = [get(r) for r in humans if get(r) is not None]
        if not higher_bot:
            pos, neg = [-x for x in pos], [-x for x in neg]
        a = auc(pos, neg)
        lines.append(f"| {name} | " + " | ".join(means) + f" | {'-' if a is None else f'{a:.2f}'} |")

    # Detection rate per probe, per group, at the verdict threshold
    lines += ["", f"## Flag rate per probe (score >= {VERDICT_THRESHOLD})", "", "For bot groups this is the detection rate; for humans it is the false-alarm rate.", "",
              "| Group | " + " | ".join(p for p, _ in PROBES) + " |", "|---|" + "---|" * len(PROBES)]
    for k in sorted(groups):
        cells = []
        for _, key in PROBES:
            vals = [r["scores"].get(key) for r in groups[k] if r["scores"].get(key) is not None]
            cells.append(f"{np.mean([v >= VERDICT_THRESHOLD for v in vals]):.0%} ({len(vals)})" if vals else "-")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")

    report = os.path.join(out_dir, "report.md")
    with open(report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {report}")

    try:
        _plot(groups, os.path.join(out_dir, "report.png"))
        print(f"Wrote {os.path.join(out_dir, 'report.png')}")
    except Exception as e:
        print(f"(plot skipped: {e})")
    return 0


def _plot(groups: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [FEATURES[0], FEATURES[1], FEATURES[3], FEATURES[9]]
    order = sorted(groups, key=lambda k: (not groups[k][0]["is_bot"], k))
    colors = {k: ("#E8590C" if groups[k][0]["is_bot"] else "#2B8A3E") for k in order}
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.6))
    rng = np.random.default_rng(0)
    for ax, (name, get, _) in zip(axes, panels):
        for i, k in enumerate(order):
            vals = [get(r) for r in groups[k] if get(r) is not None]
            if not vals:
                continue
            x = i + rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(x, vals, s=22, color=colors[k], alpha=0.75, edgecolor="none")
            ax.hlines(np.median(vals), i - 0.28, i + 0.28, color="#222", lw=1.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=20)
        ax.set_title(name, fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[-1].axhline(VERDICT_THRESHOLD, color="#C92A2A", ls="--", lw=1)
    fig.suptitle("Bots (orange) vs humans (green); bar = median", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
