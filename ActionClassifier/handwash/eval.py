"""Models on whole videos through run.py's pipeline: METC validation (42 videos, 16 fps) and 30 PSKUS DataSet7 videos (30 fps,
another hospital). Frames after 1 s smoothing; "strict" compares with the frame's own label, "+-0.5 s" counts a prediction right
when it matches any label within half a second (a step change is not a sharp line: two annotators also differ there).
Macro = mean over the 7 classes.

python eval.py weight/tcn-r10-s2 weight/tcn-r10-s2-center ...
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

from extract import videos
from run import Classifier, classify, landmarks, model_window

TOL = 0.5


def label(f):
    a = np.int64([l["code"] for l in json.load(open(f))["labels"]])
    return np.where(a >= 7, 0, a)


def sets():
    ann = {}
    for f in sorted(glob.glob("data/handwash/pskus/DataSet7/Annotations/*/*.json")):
        ann.setdefault(Path(f).stem, f)
    pskus = sorted(v for v in Path("data/handwash/pskus/DataSet7").glob("Videos/*.mp4") if v.stem in ann)[::6][:30]
    metc = [(v, a) for v, a in videos() if (m := re.match(r".*_\d\d-\d\d-\d\d-(\d+)", v.stem)) and int(m.group(1)) % 5 == 0]
    return {"METC val": [(v, label(a)) for v, a in metc], "PSKUS": [(v, label(ann[v.stem])) for v in pskus]}


def scores(rows, fps_of):
    ref = np.concatenate([r for r, _, _ in rows])
    lab = np.concatenate([l for _, l, _ in rows])
    ok_tol = np.concatenate([np.array([(l[i] == r[max(0, i - w):i + w + 1]).any() for i in range(len(r))])
                             for r, l, f in rows for w in [round(TOL * f)]])
    mac = lambda ok: np.mean([ok[ref == k].mean() for k in range(7) if (ref == k).any()]) * 100
    return (ref == lab).mean() * 100, mac(ref == lab), ok_tol.mean() * 100, mac(ok_tol)


def episode(rows, done=1.0, pred=1.0):
    """Per video, a step counts as done when its frames add up to `done` seconds (labels) or `pred` seconds (predictions).
    Missing steps: caught / missed / false alarms (a done step reported missing); videos: incomplete ones caught, complete ones flagged."""
    caught = missed = alarm = n_done = inc_caught = inc = comp_flag = comp = 0
    for ref, lab, f in rows:
        t = np.array([(ref == k).sum() / f >= done for k in range(1, 7)])
        p = np.array([(lab == k).sum() / f >= pred for k in range(1, 7)])
        caught += (~t & ~p).sum(); missed += (~t & p).sum(); alarm += (t & ~p).sum(); n_done += t.sum()
        if t.all():
            comp += 1; comp_flag += (~p).any()
        else:
            inc += 1; inc_caught += (~p).any()
    return caught, caught + missed, alarm, n_done, inc_caught, inc, comp_flag, comp


if __name__ == "__main__":
    torch.cuda.set_stream(torch.cuda.Stream())
    models = sys.argv[1:]
    clfs = {m: Classifier(m) for m in models}
    res, raw = {}, {}
    for name, vids in sets().items():
        for v, ref in vids:
            _, _, x, p, fps, H, W, _ = landmarks(v)
            ref = ref[:len(x)]
            for m in models:
                rate, sec, anchor, interp = model_window(m)
                prob, sm = classify(clfs[m], x, p, fps, H, W, rate, sec, 1.0, anchor, interp)
                res.setdefault((name, m), []).append((ref, sm.argmax(1).cpu().numpy()[:len(ref)], fps))
                raw.setdefault(m, []).append(dict(set=name, video=v.stem, fps=fps, ref=ref, prob=prob.cpu().numpy()[:len(ref)]))
        print(name, "done", flush=True)
    for m, rows in raw.items():  # per-frame probabilities (no smoothing) for post-processing: post.py
        Path("cache/probs").mkdir(parents=True, exist_ok=True)
        np.save(f"cache/probs/{Path(m).name}.npy", np.array(rows, dtype=object), allow_pickle=True)
    print("\naccuracy / macro, strict | +-0.5 s")
    print("model".ljust(28) + "METC val".ljust(32) + "PSKUS 30 fps")
    for m in models:
        cells = ["{:.1f} / {:.1f} | {:.1f} / {:.1f}".format(*scores(res[(n, m)], None)).ljust(32) for n in ("METC val", "PSKUS")]
        print(Path(m).name.ljust(28) + "".join(cells))
    print("\nwhole videos: a step is done with >= 1 s of it in the labels, reported done with >= `pred` s of it predicted")
    for n in ("METC val", "PSKUS"):
        for m in models:
            for pr in (0.5, 1.0, 2.0):
                c, mis, al, nd, ic, inc, cf, comp = episode(res[(n, m)], 1.0, pr)
                print(f"  {n:9s} {Path(m).name:24s} pred >= {pr:3.1f} s: missing steps caught {c}/{mis}, false alarms {al}/{nd} done steps; "
                      f"incomplete videos caught {ic}/{inc}, complete videos flagged {cf}/{comp}")
