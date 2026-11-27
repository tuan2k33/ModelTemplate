"""Compare hands.py (eager, onnxruntime) with the MediaPipe CPU landmarks in cache/kpt-video-0.5. python check_hands.py [video index] [frames] [flag output]"""
import sys

import cv2
import numpy as np
import onnxruntime as ort
import torch

from extract import videos
from hands import Eager

idx, nf, flag = int(sys.argv[1]) if len(sys.argv) > 1 else 3, int(sys.argv[2]) if len(sys.argv) > 2 else 400, sys.argv[3] if len(sys.argv) > 3 else "Identity_1"
det = ort.InferenceSession("cache/mp/hand_detector.onnx")
lm = ort.InferenceSession("cache/mp/hand_landmarks_b2.onnx")
run_det = lambda x: [torch.from_numpy(o) for o in det.run(None, {"input_1": x.numpy()})]
def run_lm(x):
    o = dict(zip([n.name for n in lm.get_outputs()], lm.run(None, {"input_1": x.numpy()})))
    return torch.from_numpy(o["Identity"]), torch.from_numpy(o[flag])
v, a = videos()[idx]
ref = np.load(f"cache/kpt-video-0.5/{v.stem}.npz")
e = Eager(run_det, run_lm)
cap = cv2.VideoCapture(str(v))
cnt, ref_cnt, err = [], [], []
for i in range(nf):
    ok, f = cap.read()
    if not ok:
        break
    out, keep = e(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    out, keep = out.numpy(), keep.numpy()
    cnt.append(int(keep.sum())); ref_cnt.append(int(ref["present"][i].sum()))
    for k in np.where(keep > 0)[0]:  # nearest reference hand, error in pixels over the 21 points
        r = [ref["lm"][i, j] for j in range(2) if ref["present"][i, j]]
        if r:
            d = [np.abs((out[k, :, :2] - rr[:, :2]) * [640, 480]).mean() for rr in r]
            err.append(min(d))
cnt, ref_cnt = np.array(cnt), np.array(ref_cnt)
print(f"{v.stem}: {len(cnt)} frames; hands ours {cnt.mean():.2f} vs MediaPipe {ref_cnt.mean():.2f}; same count {np.mean(cnt == ref_cnt) * 100:.0f}%; "
      f"landmark error px median {np.median(err):.2f}, p90 {np.percentile(err, 90):.2f}")
