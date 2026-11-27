"""Post-processing of the per-frame logits that eval.py saves (cache/probs/<model>.npy), no model run needed.
Probabilities below = softmax of the logits (only here, for the comparison).

  vote      argmax per frame, most frequent label within 1 s (what run.py does)
  avg       mean of the probabilities over 1 s, argmax
  edge5     avg, but "other" only allowed in the first and last 5 s
  span-T    avg; the wash runs from the first to the last second-long stretch of steps; inside it, "other" only when its
            probability is above T (else the best step)
  hmm-D     Viterbi over 0-6: step changes follow the order seen in METC training labels (next step likely, skips and going
            back possible), a step lasts about D s; emissions = probabilities / class frequency

python post.py tcn-r10-s1-center
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

from eval import episode, scores
from run import segments

STEPS = 7


def avg(prob, fps, sec=1.0):
    k = max(1, round(sec * fps))
    c = np.cumsum(np.vstack([np.zeros((1, STEPS)), prob]), 0)
    i = np.arange(len(prob))
    lo, hi = np.maximum(0, i - k // 2), np.minimum(len(prob), i - k // 2 + k)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def edge5(prob, fps):
    sm = avg(prob, fps)
    lab = sm.argmax(1)
    e = round(5 * fps)
    mid = np.arange(len(lab))
    mid = (mid >= e) & (mid < len(lab) - e)
    lab[mid] = sm[mid, 1:].argmax(1) + 1
    return lab


def span(prob, fps, thr):
    sm = avg(prob, fps)
    lab = sm.argmax(1)
    runs = [(b, e) for k, b, e in segments(lab, fps) if k > 0 and e - b >= 1.0]
    if not runs:
        return lab
    a, b = round(runs[0][0] * fps), round(runs[-1][1] * fps)
    inside = np.zeros(len(lab), bool)
    inside[a:b] = True
    swap = inside & (lab == 0) & (sm[:, 0] <= thr)
    lab[swap] = sm[swap, 1:].argmax(1) + 1
    return lab


def transitions():
    """Step-to-step changes in METC training labels (subjects id % 5 != 0), as a 7x7 matrix of next-state shares."""
    T = np.ones((STEPS, STEPS))  # +1 everywhere: any change stays possible
    freq = np.zeros(STEPS)
    for f in sorted(glob.glob("data/handwash/metc/*/Annotations/*/*.json")):
        m = re.match(r".*_\d\d-\d\d-\d\d-(\d+)", Path(f).stem)
        if not m or int(m.group(1)) % 5 == 0:
            continue
        lab = np.int64([l["code"] for l in json.load(open(f))["labels"]])
        lab[lab >= STEPS] = 0
        freq += np.bincount(lab, minlength=STEPS)
        s = [k for k, b, e in segments(lab, 16)]
        for a, b in zip(s, s[1:]):
            T[a, b] += 1
    np.fill_diagonal(T, 0)
    return T / T.sum(1, keepdims=True), freq / freq.sum()


def hmm(prob, fps, dur, trans, prior):
    stay = 1 - 1 / (dur * fps)
    A = np.log(trans * (1 - stay) + np.eye(STEPS) * stay)
    E = np.log(np.clip(prob / prior, 1e-9, None))
    n = len(prob)
    back = np.zeros((n, STEPS), np.int64)
    v = np.log(np.full(STEPS, 1 / STEPS)) + E[0]
    for t in range(1, n):
        s = v[:, None] + A
        back[t] = s.argmax(0)
        v = s.max(0) + E[t]
    lab = np.zeros(n, np.int64)
    lab[-1] = v.argmax()
    for t in range(n - 1, 0, -1):
        lab[t - 1] = back[t, lab[t]]
    return lab


if __name__ == "__main__":
    rows = np.load(f"cache/probs/{sys.argv[1]}.npy", allow_pickle=True)
    for r in rows:
        if "logit" in r:
            e = np.exp(r["logit"] - r["logit"].max(1, keepdims=True))
            r["prob"] = e / e.sum(1, keepdims=True)
    trans, prior = transitions()
    methods = {"vote": lambda r: avg(np.eye(STEPS)[r["prob"].argmax(1)], r["fps"]).argmax(1), "avg": lambda r: avg(r["prob"], r["fps"]).argmax(1), "edge5": lambda r: edge5(r["prob"], r["fps"])}
    for t in (0.5, 0.7, 0.9, 1.0):
        methods[f"span-{t}"] = lambda r, t=t: span(r["prob"], r["fps"], t)
    for d in (1, 2, 4):
        methods[f"hmm-{d}"] = lambda r, d=d: hmm(r["prob"], r["fps"], d, trans, prior)
    print(f"{sys.argv[1]}: accuracy / macro strict | +-0.5 s;  per video (step done >= 1 s): missing steps caught, false alarms, incomplete caught, complete flagged")
    for name in ("METC val", "PSKUS"):
        print(name)
        for k, fn in methods.items():
            res = [(r["ref"], fn(r), r["fps"]) for r in rows if r["set"] == name]
            c, mis, al, nd, ic, inc, cf, comp = episode(res, 1.0, 1.0)
            print(f"  {k:9s} " + "{:.1f} / {:.1f} | {:.1f} / {:.1f}".format(*scores(res, None)).ljust(28) +
                  f"missing {c}/{mis}  alarms {al}/{nd}  incomplete {ic}/{inc}  complete flagged {cf}/{comp}")
