"""Train the screener (the defender) on labelled calls.

Fits a logistic regression on the raw behavioural features (rtt.features.MODEL_FEATURES) of every saved
call: bots are labelled by the run that made them, humans are calls made with `--caller human`.
The decision threshold is set so that at most --max-fpr of human calls get flagged, then folded into
the bias so that P(bot) >= 0.5 still means "bot" everywhere else in the code.

  python -m rtt.train                        # fit on results/, save results/screener_model.json
  python -m rtt.run --screener-model results/screener_model.json
  python -m rtt.redteam --defender-learns    # retrain after every red-team round

Without human calls there is nothing to tell bots apart from, so training refuses to run.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np

from .features import MODEL_FEATURES, model_vector

MIN_BOTS = 3


class NotEnoughData(Exception):
    pass


def load_calls(roots, fake: bool = False) -> list[dict]:
    """Every saved call under `roots`. Real and fake (--fake) calls are never mixed."""
    seen, calls = set(), []
    for root in [roots] if isinstance(roots, str) else roots:
        for path in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
            key = os.path.abspath(path)
            if key in seen:
                continue
            seen.add(key)
            try:
                with open(path) as f:
                    d = json.load(f)
            except (OSError, ValueError):
                continue
            if not isinstance(d, dict) or "meta" not in d or "analysis" not in d:
                continue  # red-team logs, memory, models, reports
            if bool(d["meta"].get("fake_backends")) != fake:
                continue
            if d["meta"].get("echo_suspected"):
                continue  # the mic heard the screener: timing is meaningless
            calls.append({"path": path, "is_bot": bool(d["meta"]["is_bot"]), "features": d["analysis"]["features"]})
    return calls


def _fit_beta(Z: np.ndarray, y: np.ndarray, sw: np.ndarray, l2: float) -> np.ndarray:
    """Weighted L2 logistic regression by Newton's method. Returns [w..., b]."""
    X = np.hstack([Z, np.ones((len(Z), 1))])
    R = l2 * np.eye(X.shape[1])
    R[-1, -1] = 0.0  # don't shrink the bias
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -40, 40)))
        g = X.T @ (sw * (p - y)) + R @ beta
        H = (X * (sw * p * (1 - p))[:, None]).T @ X + R + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def fit(calls: list[dict], max_fpr: float = 0.05, l2: float = 1.0, min_humans: int = 5, fake: bool = False) -> dict:
    y = np.array([c["is_bot"] for c in calls], dtype=float)
    n_bot, n_human = int(y.sum()), int(len(y) - y.sum())
    if n_human < min_humans:
        raise NotEnoughData(f"need at least {min_humans} human calls, have {n_human}"
                            + ("" if fake else " (record some with: python -m rtt.run --caller human)"))
    if n_bot < MIN_BOTS:
        raise NotEnoughData(f"need at least {MIN_BOTS} bot calls, have {n_bot}")

    raw = np.array([[np.nan if v is None else v for v in model_vector(c["features"])] for c in calls], dtype=float)
    present = ~np.isnan(raw)
    mean = np.where(present, raw, 0.0).sum(0) / np.maximum(present.sum(0), 1)  # mean of the values we have
    filled = np.where(np.isnan(raw), mean, raw)
    std = filled.std(0)
    std[std < 1e-6] = 1.0
    Z = (filled - mean) / std
    # Balance the classes: the red team produces far more bot calls than you will record human ones.
    sw = np.where(y == 1, len(y) / (2 * n_bot), len(y) / (2 * n_human))

    # Leave-one-out: each call is scored by a model that never saw it.
    loo = np.empty(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        beta = _fit_beta(Z[keep], y[keep], sw[keep], l2)
        loo[i] = _sigmoid(Z[i] @ beta[:-1] + beta[-1])
    humans, bots = np.sort(loo[y == 0])[::-1], loo[y == 1]
    allowed = int(math.floor(max_fpr * n_human))  # humans we accept flagging
    thr = 0.5 if allowed >= n_human else max(0.5, float(humans[allowed]) + 1e-6)

    beta = _fit_beta(Z, y, sw, l2)
    shift = math.log(thr / (1 - thr))  # move the threshold to 0.5
    return {
        "id": "defender-" + time.strftime("%Y%m%d-%H%M%S"),
        "features": list(MODEL_FEATURES),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "w": beta[:-1].tolist(),
        "b": float(beta[-1] - shift),
        "threshold": thr,
        "max_fpr": max_fpr,
        "l2": l2,
        "n_human": n_human,
        "n_bot": n_bot,
        "fake": fake,
        "loo": {
            "bots_caught": float(np.mean(bots >= thr)),
            "humans_flagged": float(np.mean(loo[y == 0] >= thr)),
            "auc": _auc(bots, loo[y == 0]),
        },
        "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def describe(m: dict) -> str:
    l = m["loo"]
    return (f"{m['id']} on {m['n_human']} humans + {m['n_bot']} bots: catches {l['bots_caught']:.0%} of bots, "
            f"flags {l['humans_flagged']:.0%} of humans (leave-one-out), AUC {l['auc']:.2f}")


def save_model(m: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(m, f, indent=2)


def load_model(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train the screener on labelled calls")
    p.add_argument("data", nargs="*", default=["results"], help="folders of saved calls (searched recursively)")
    p.add_argument("--fake", action="store_true", help="train on --fake calls instead (plumbing test only)")
    p.add_argument("--max-fpr", type=float, default=0.05, help="highest share of human calls it may flag")
    p.add_argument("--l2", type=float, default=1.0, help="regularisation; raise it if you have few calls")
    p.add_argument("--min-humans", type=int, default=5)
    p.add_argument("--out", default=None, help="default: results/screener_model.json (screener_model_fake.json with --fake)")
    a = p.parse_args(argv)

    calls = load_calls(a.data, fake=a.fake)
    try:
        m = fit(calls, max_fpr=a.max_fpr, l2=a.l2, min_humans=a.min_humans, fake=a.fake)
    except NotEnoughData as e:
        p.error(str(e))
    out = a.out or os.path.join("results", "screener_model_fake.json" if a.fake else "screener_model.json")
    save_model(m, out)
    print(describe(m))
    print("\nweight per feature (standardised; + means pushes towards bot):")
    for name, w in sorted(zip(m["features"], m["w"]), key=lambda x: -abs(x[1])):
        print(f"  {name:<18}{w:+.2f}")
    print(f"\nSaved {out}. Use it with: python -m rtt.run --screener-model {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
