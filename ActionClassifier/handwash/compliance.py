"""Per video: compliant (all 6 WHO steps, each >= 1 s in the labels) or not, and how often the prediction agrees.
A video is predicted compliant when every step gets >= `thr` s of predicted frames. Positive = non-compliant (the alert).

python compliance.py cache/probs/tcn-r10-s1-center.npy   # rows saved by eval.py (or compliance_all.py)
"""
import sys

import numpy as np

from post import avg


def verdicts(rows, thr):
    out = []
    for r in rows:
        x = r["logit"] if "logit" in r else r["prob"]
        lab = avg(np.eye(7)[x.argmax(1)], r["fps"]).argmax(1)  # most frequent label within 1 s, as run.py
        done = lambda l, s: all((l == k).sum() / r["fps"] >= s for k in range(1, 7))
        out.append((not done(r["ref"], 1.0), not done(lab, thr)))  # (truly non-compliant, alert)
    return np.array(out)


def report(rows, name):
    print(f"{name}: {len(rows)} videos")
    for thr in (0.5, 1.0, 1.5, 2.0):
        v = verdicts(rows, thr)
        tp, fn = (v[:, 0] & v[:, 1]).sum(), (v[:, 0] & ~v[:, 1]).sum()
        fp, tn = (~v[:, 0] & v[:, 1]).sum(), (~v[:, 0] & ~v[:, 1]).sum()
        print(f"  step done at >= {thr} s: accuracy {(tp + tn) / len(v) * 100:5.1f}%  |  non-compliant caught {tp}/{tp + fn} (recall {tp / max(1, tp + fn) * 100:.0f}%), "
              f"alerts right {tp}/{tp + fp} (precision {tp / max(1, tp + fp) * 100:.0f}%)  |  compliant passed {tn}/{tn + fp}, false alarms {fp}")


if __name__ == "__main__":
    rows = np.load(sys.argv[1], allow_pickle=True)
    for name in dict.fromkeys(r["set"] for r in rows):
        report([r for r in rows if r["set"] == name], name)
