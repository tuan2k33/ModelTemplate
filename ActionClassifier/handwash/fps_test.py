"""Classifiers at other frame rates: PSKUS videos (30 fps, another hospital, 320x240) re-encoded to 25 and 10 fps with ffmpeg.
Hands engine once per video and rate; every model then runs on the same landmarks with its windows resampled to the rate it
was trained at ("model") or on every frame of the video ("native"). Accuracy and macro (mean over the steps) after the 1 s vote.

python fps_test.py [--videos 30] [--models weight/tcn-r*]   # PSKUS DataSet7 in data/handwash/pskus (zenodo 4537209), videos in out/fps/
"""
import argparse
import glob
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from run import Classifier, classify, landmarks, model_window
from train import STEPS

ap = argparse.ArgumentParser()
ap.add_argument("--videos", type=int, default=30)
ap.add_argument("--fps", type=int, nargs="+", default=[30, 25, 10])
ap.add_argument("--models", nargs="+", default=["weight/tcn-r*"])
a = ap.parse_args()
torch.cuda.set_stream(torch.cuda.Stream())
src = Path("data/handwash/pskus/DataSet7")
ann = {}
for f in sorted(glob.glob(str(src / "Annotations/*/*.json"))):
    ann.setdefault(Path(f).stem, f)
vids = sorted(v for v in src.glob("Videos/*.mp4") if v.stem in ann)[::max(1, len(ann) // a.videos)][:a.videos]
models = sorted({m for g in a.models for m in glob.glob(g)}, key=lambda m: (-model_window(m)[0], model_window(m)[1]))
clfs = {m: Classifier(m) for m in models}
out = Path("out/fps")
out.mkdir(parents=True, exist_ok=True)
pred = {}  # (fps, model, mode) -> list of (ref, lab)
for v in vids:
    lab = np.int64([l["code"] for l in json.load(open(ann[v.stem]))["labels"]])
    lab[lab >= len(STEPS)] = 0
    for f in a.fps:
        p = out / f"{v.stem}_{f}.mp4"
        if f != 30 and not p.exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(v), "-vf", f"fps={f}", "-c:v", "libx264", "-crf", "18", str(p)], check=True)
        _, _, x, pr, fps, H, W, _ = landmarks(v if f == 30 else p)
        ref = lab[np.minimum(np.round(np.arange(len(x)) * 30 / f).astype(int), len(lab) - 1)]
        for m in models:
            rate, sec, anc, ip = model_window(m)
            for mode, r in (("model", rate), ("native", fps)):
                _, lab = classify(clfs[m], x, pr, fps, H, W, r, sec, 1.0, anc, ip)
                pred.setdefault((f, m, mode), []).append((ref, lab))
    print(v.stem, flush=True)


def score(rows):
    ref, lab = np.concatenate([r for r, _ in rows]), np.concatenate([l for _, l in rows])
    return (ref == lab).mean() * 100, np.mean([(lab[ref == k] == k).mean() for k in range(len(STEPS)) if (ref == k).any()]) * 100


print(f"\n{len(vids)} PSKUS videos, accuracy / macro, windows resampled to the model's rate | every video frame (native)")
print("model".ljust(16) + "".join(f"{f} fps".ljust(30) for f in a.fps))
for m in models:
    rate, sec, _, _ = model_window(m)
    cells = []
    for f in a.fps:
        s1, s2 = score(pred[(f, m, "model")]), score(pred[(f, m, "native")])
        cells.append(f"{s1[0]:.1f}/{s1[1]:.1f} | {s2[0]:.1f}/{s2[1]:.1f}".ljust(30))
    print(f"{rate} fps x {sec:g} s".ljust(16) + "".join(cells))
