"""python det.py [--imgsz 960] [--epochs 50] : fine-tune YOLO26n as a text-line detector on SROIE (one class), mAP on its test split.
SROIE box files hold 8 polygon numbers + the text per line; the polygon becomes an axis-aligned box. Converted set: cache/yolo/ (images symlinked)."""
import argparse
from pathlib import Path

import cv2, numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--batch", type=int, default=8)
a = ap.parse_args()
src, dst = Path("data/sroie").resolve(), Path("cache/yolo").resolve()
for sp in ("train", "test"):
    (dst / "images" / sp).mkdir(parents=True, exist_ok=True); (dst / "labels" / sp).mkdir(parents=True, exist_ok=True)
    for img in sorted((src / sp / "img").glob("*.jpg")):
        h, w = cv2.imread(str(img)).shape[:2]
        rows = []
        for l in (src / sp / "box" / (img.stem + ".txt")).read_text(errors="ignore").splitlines():
            p = l.split(",", 8)
            if len(p) < 9: continue
            xy = np.array(p[:8], float).reshape(4, 2)
            x0, y0 = xy.min(0); x1, y1 = xy.max(0)
            rows.append(f"0 {(x0+x1)/2/w:.6f} {(y0+y1)/2/h:.6f} {(x1-x0)/w:.6f} {(y1-y0)/h:.6f}")
        (dst / "labels" / sp / (img.stem + ".txt")).write_text("\n".join(rows))
        link = dst / "images" / sp / img.name
        if not link.exists(): link.symlink_to(img.resolve())
(dst / "data.yaml").write_text(f"path: {dst}\ntrain: images/train\nval: images/test\nnames:\n  0: text\n")
YOLO("../Detector/cache/yolo26n.pt").train(data=str(dst / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=0, workers=0,
                                           project=str(Path("weight").resolve()), name="det", exist_ok=True)
