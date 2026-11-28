"""METC-trained models on the whole PSKUS dataset (zenodo 4537209, 11 hospital sites/periods, 30 fps): does it generalize?

Landmarks are cached per video (cache/kpt-pskus/<DataSet>/<video>.npz, hands engine on every frame), so a rerun only does the
videos downloaded since. Then per model: frame accuracy / macro, and per video compliant or not by steps.

Also the WHO time rule: washing (first to last frame with a hand) under 40 s, and steps-or-time combined.

python pskus_eval.py [--models weight/tcn-r10-s1-center ...]   # every DataSet*/ found in data/handwash/pskus
"""
import argparse
import csv
import glob
import re
from pathlib import Path

import numpy as np
import torch

import eval as ev
from eval import label
from run import WHO_SECONDS, Classifier, classify, hand_span, landmarks, model_window

ROOT = Path("data/handwash/pskus")


def cached():
    out = []
    log = Path("log/pskus-download.log")
    ready = set(log.read_text().split()) if log.exists() else set()  # "HH:MM DataSetN K videos" once unzipped
    for d in sorted(ROOT.glob("DataSet*/")):
        if d.name != "DataSet7" and d.name not in ready:  # still downloading / unzipping
            continue
        ann = {}
        for f in sorted(glob.glob(str(d / "Annotations/*/*.json"))):
            ann.setdefault(Path(f).stem, f)
        for v in sorted(d.glob("Videos/*.mp4")):
            if v.stem not in ann or (d.name == "DataSet5" and v.stem.endswith("camera105")):  # its camera105 (the id is reused in other DataSets, fine there): red/green swapped, hands enter at the edge, MediaPipe finds almost none
                continue
            c = Path(f"cache/kpt-pskus/{d.name}/{v.stem}.npz")
            if not c.exists():
                _, _, x, p, fps, H, W, _ = landmarks(v)
                c.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(c, x=x, p=p, fps=fps, H=H, W=W, ref=label(ann[v.stem])[:len(x)])
            out.append((d.name, c))
        print(d.name, "landmarks ready", flush=True)
    return out


def washing_time():
    """Washing seconds per video from PSKUS statistics.csv (its washing_duration and movement_1 columns are glued: "15.65.8")."""
    out = {}
    for line in list(csv.reader(open(ROOT / "statistics.csv")))[1:]:
        m = re.fullmatch(r"(\d+\.\d)(\d+\.\d)", line[7])
        out[line[0][:-4]] = float(m.group(1) if m else line[7])
    return out


def verdict(ref, lab, fps, need, thr):
    ok = lambda l, s: all((l == k).sum() / fps >= s for k in range(1, 7))
    return not ok(ref, need), not ok(lab, thr)  # (truly non-compliant, alert)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["weight/tcn-r10-s1-center-v2"])
    a = ap.parse_args()
    torch.cuda.set_stream(torch.cuda.Stream())
    vids = cached()
    wt = washing_time()
    for m in a.models:
        clf, (rate, sec, anchor, interp) = Classifier(m), model_window(m)
        per = {}
        for ds, c in vids:
            d = np.load(c)
            logit, lab = classify(clf, d["x"], d["p"], float(d["fps"]), int(d["H"]), int(d["W"]), rate, sec, 1.0, anchor, interp)
            per.setdefault(ds, []).append((d["ref"], lab[:len(d["ref"])], float(d["fps"]), hand_span(d["p"], float(d["fps"])), wt.get(c.stem)))
        print(f"\n{Path(m).name} on PSKUS. Frames: accuracy / macro within +-0.5 s | +-1 s of a label. Videos, positive = non-compliant: compliant = every step >= 1 s (2.5 s) in the labels, "
              f"alert when a step has < 0.5 s (1.25 s) predicted; FPR = compliant washes alerted, recall = non-compliant caught")
        allrows = []
        for ds in sorted(per, key=lambda s: int(s[7:])) + ["all"]:
            rows = allrows if ds == "all" else per[ds]
            if ds != "all":
                allrows += rows
            fr = []
            for ev.TOL in (0.5, 1.0):
                fr.append("{2:4.1f} / {3:4.1f}".format(*ev.scores([(r, l, f) for r, l, f, _, _ in rows], None)))
            cells = []
            for need, thr in ((1.0, 0.5), (2.5, 1.25)):
                v = np.array([verdict(r, l, f, need, thr) for r, l, f, _, _ in rows])
                comp, bad = (~v[:, 0]).sum(), v[:, 0].sum()
                cells.append(f"{comp:3d} compliant: FPR {(v[:, 1] & ~v[:, 0]).sum() / max(1, comp) * 100:3.0f}%, recall {(v[:, 1] & v[:, 0]).sum() / max(1, bad) * 100:3.0f}%")
            # WHO time: washing >= 40 s (labels: statistics.csv), alert when the hand span is < 40 s; combined: steps (1 s / 0.5 s) or time
            t = [(w < WHO_SECONDS, h < WHO_SECONDS) + verdict(r, l, f, 1.0, 0.5) for r, l, f, h, w in rows if w is not None]
            bad_t, al_t, bad_s, al_s = (np.array(c, bool) for c in zip(*t))
            for name, bad, al in (("time < 40 s", bad_t, al_t), ("steps or time", bad_t | bad_s, al_t | al_s)):
                ok = ~bad
                cells.append(f"{name}: {ok.sum()} compliant, FPR {(al & ok).sum() / max(1, ok.sum()) * 100:3.0f}%, recall {(al & bad).sum() / max(1, bad.sum()) * 100:3.0f}%")
            print(f"  {ds:10s} {len(rows):4d} videos  frames +-0.5 s {fr[0]}, +-1 s {fr[1]}  |  " + "  |  ".join(cells), flush=True)
