"""Hand recall of the hands engine for palm / presence thresholds and duplicate removal (python thr_test.py [det,pres,dedup ...]): METC (12 videos) and PSKUS DataSet7 (12 videos).
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
import sys
configs = [(0.5, 0.5, "iou"), (0.5, 0.5, "landmarks"), (0.3, 0.3, "landmarks"), (0.3, 0.3, "iou")] if len(sys.argv) < 2 else [tuple(c.split(",")) for c in sys.argv[1:]]
for det, pres, dedup in configs:
    det, pres = float(det), float(pres)
    out = []
    for name, vids in (("METC", [v for v, _ in metc]), ("PSKUS", pskus)):
        n = np.zeros(3); dup = 0
        for v in vids:
            cap = cv2.VideoCapture(str(v))
            H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            p = Path(f"cache/mp/thr/hands_{H}x{W}_d{det}_p{pres}_{dedup}.engine")
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                export(p.with_suffix(".onnx"), H, W, det_thr=det, present_thr=pres, dedup=dedup)
                compile_engine(p.with_suffix(".onnx"), p)
            e = Engine(p)
            lab = metc_lab[v.stem] if name == "METC" else label(ann[v.stem])
            pr = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                l, q = e(torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).cuda())
                pr.append(q.sum().item())
                if pr[-1] == 2:  # the same hand twice: all landmarks within a quarter hand
                    x = l[..., :2].cpu().numpy() * [W, H]
                    size = np.mean([np.linalg.norm(x[k, 9] - x[k, 0]) for k in range(2)])
                    dup += np.linalg.norm(x[0] - x[1], axis=1).mean() < 0.25 * size
            pr = np.array(pr)[:len(lab)]
            m = lab[:len(pr)] > 0  # frames inside a step: both hands are there
            n += np.bincount(pr[m].astype(int), minlength=3)[:3]
        out.append(f"{name}: recall {(n[1] + 2 * n[2]) / (2 * n.sum()) * 100:4.1f}%  0/1/2 hands {n[0] / n.sum() * 100:3.0f}/{n[1] / n.sum() * 100:3.0f}/{n[2] / n.sum() * 100:3.0f}, same hand twice {dup}")
    print(f"palm >= {det}, presence >= {pres}, dedup {dedup}:  " + "   ".join(out), flush=True)
