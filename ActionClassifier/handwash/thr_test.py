"""Hand recall of the hands engine for palm / presence thresholds: METC (12 videos) and PSKUS DataSet7 (12 videos).
Recall = hands found / 2 while washing (both hands are in the sink). python thr_test.py"""
import glob
from pathlib import Path

import cv2
import numpy as np
import torch

from eval import label
from extract import videos
from hands import Engine, compile_engine, export

torch.cuda.set_stream(torch.cuda.Stream())
metc = [(v, None) for v, a in videos()[:12]]
ann = {}
for f in sorted(glob.glob("data/handwash/pskus/DataSet7/Annotations/*/*.json")):
    ann.setdefault(Path(f).stem, f)
pskus = [v for v in sorted(Path("data/handwash/pskus/DataSet7/Videos").glob("*.mp4")) if v.stem in ann][::17][:12]
import json
metc_lab = {v.stem: np.int64([l["code"] for l in json.load(open(a))["labels"]]) for v, a in videos()[:12]}
for det, pres in ((0.5, 0.5), (0.4, 0.5), (0.3, 0.5), (0.2, 0.5), (0.3, 0.4), (0.3, 0.3)):
    out = []
    for name, vids in (("METC", [v for v, _ in metc]), ("PSKUS", pskus)):
        n = np.zeros(3); steps = np.zeros(2)
        for v in vids:
            cap = cv2.VideoCapture(str(v))
            H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            p = Path(f"cache/mp/thr/hands_{H}x{W}_d{det}_p{pres}.engine")
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                export(p.with_suffix(".onnx"), H, W, det_thr=det, present_thr=pres)
                compile_engine(p.with_suffix(".onnx"), p)
            e = Engine(p)
            lab = metc_lab[v.stem] if name == "METC" else label(ann[v.stem])
            pr = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                pr.append(e(torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).cuda())[1].sum().item())
            pr = np.array(pr)[:len(lab)]
            m = lab[:len(pr)] > 0  # frames inside a step: both hands are there
            n += np.bincount(pr[m].astype(int), minlength=3)[:3]
        out.append(f"{name}: recall {(n[1] + 2 * n[2]) / (2 * n.sum()) * 100:4.1f}%  0/1/2 hands {n[0] / n.sum() * 100:3.0f}/{n[1] / n.sum() * 100:3.0f}/{n[2] / n.sum() * 100:3.0f}")
    print(f"palm >= {det}, presence >= {pres}:  " + "   ".join(out), flush=True)
