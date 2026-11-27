"""Hand landmarks of the METC hand-washing videos with MediaPipe HandLandmarker (CPU),
or the same networks as one TensorRT engine (--mode engine, hands.py, thresholds fixed at 0.5).

One cache/kpt-<mode>-<conf>/<video>.npz per video:
  lm      [F,2,21,3] float32  x, y in [0,1] of the frame, z relative depth; 2 slots kept apart by wrist position
  present [F,2] uint8          hand found in that slot
  label   [F] int8             WHO movement code (0 = other, 1-6 the steps, 7 faucet)

python extract.py [--mode video|image|engine] [--fps 20] [--conf 0.5] [--limit N] [--workers 6]
"""
import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

DATA = Path("data/handwash/metc")
TASK = "cache/hand_landmarker.task"
URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def videos():
    out, seen = [], set()
    for v in sorted(DATA.glob("Interface_number_*/Videos/*.mp4")):
        a = sorted(v.parent.parent.glob(f"Annotations/*/{v.stem}.json"))
        key = v.stat().st_size
        if not a or key in seen:  # the zips hold a few videos twice under two names
            continue
        seen.add(key)
        out.append((v, a[0]))
    return out


def mediapipe_hands(v, conf, mode):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opt = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=TASK), running_mode=getattr(vision.RunningMode, mode.upper()), num_hands=2,
        min_hand_detection_confidence=conf, min_hand_presence_confidence=conf, min_tracking_confidence=conf)
    cap = cv2.VideoCapture(str(v))
    fps = cap.get(cv2.CAP_PROP_FPS)
    with vision.HandLandmarker.create_from_options(opt) as det:
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            im = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            r = det.detect(im) if mode == "image" else det.detect_for_video(im, int(i * 1000 / fps))
            yield [np.float32([[p.x, p.y, p.z] for p in pts]) for pts in r.hand_landmarks]
            i += 1


def engine_hands(v):
    """hands.py engine (MediaPipe's networks + its glue as one TensorRT graph) on the GPU."""
    import torch
    from hands import Engine, compile_engine, export
    cap = cv2.VideoCapture(str(v))
    H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    p = Path(f"cache/mp/hands_{H}x{W}.engine")
    if not p.exists():
        export(p.with_suffix(".onnx"), H, W)
        compile_engine(p.with_suffix(".onnx"), p)
    e = Engine(p)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        lm, present = e(torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).cuda())
        lm, present = lm.cpu().numpy(), present.cpu().numpy()
        yield [lm[k] for k in range(2) if present[k] > 0]


class Slots:
    """Hands of one frame -> lm [2,21,3], present [2]. MediaPipe's left/right flips when seen from above,
    so the two slots follow the wrists instead (each hand goes to the slot whose last wrist is closer)."""

    def __init__(self):
        self.wrist = np.float32([[0.3, 0.5], [0.7, 0.5]])

    def __call__(self, hands):
        d = np.float32([[np.linalg.norm(h[0, :2] - w) for w in self.wrist] for h in hands])
        if len(hands) == 1:
            slots = [int(d[0].argmin())]
        elif len(hands) == 2:
            slots = [0, 1] if d[0, 0] + d[1, 1] <= d[0, 1] + d[1, 0] else [1, 0]
        else:
            slots = []
        x = np.zeros((2, 21, 3), np.float32)
        p = np.zeros(2, np.uint8)
        for k, h in zip(slots, hands):
            x[k], p[k], self.wrist[k] = h, 1, h[0, :2]
        return x, p


def extract(v, ann, conf, mode, out, fps=None):
    label = np.int8([l["code"] for l in json.load(open(ann))["labels"]])
    if fps:  # re-encode to another frame rate (ffmpeg repeats or drops frames), labels taken from the nearest original frame
        src = Path(f"cache/metc-{fps}fps/{v.name}")
        if not src.exists():
            src.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(v), "-vf", f"fps={fps}", "-c:v", "libx264", "-crf", "18", str(src)], check=True)
        n = int(cv2.VideoCapture(str(src)).get(cv2.CAP_PROP_FRAME_COUNT))
        label = label[np.minimum(np.round(np.arange(n) * cv2.VideoCapture(str(v)).get(cv2.CAP_PROP_FPS) / fps).astype(int), len(label) - 1)]
        v = src
    lm, present, slots = [], [], Slots()
    for hands in (engine_hands(v) if mode == "engine" else mediapipe_hands(v, conf, mode)):
        x, p = slots(hands)
        lm.append(x)
        present.append(p)
    n = min(len(lm), len(label))
    np.savez_compressed(out, lm=np.stack(lm)[:n], present=np.stack(present)[:n], label=label[:n])
    return v.name, n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--mode", default="video", choices=["video", "image", "engine"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fps", type=int, help="re-encode the videos to this frame rate first (cache/metc-<fps>fps/)")
    a = ap.parse_args()
    if not Path(TASK).exists():
        Path(TASK).parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(URL, TASK)
    vs = videos()[:a.limit]
    dst = Path(f"cache/kpt-{a.mode}-{a.conf}" + (f"-{a.fps}fps" if a.fps else ""))
    dst.mkdir(parents=True, exist_ok=True)
    todo = [(v, an, a.conf, a.mode, dst / f"{v.stem}.npz", a.fps) for v, an in vs if not (dst / f"{v.stem}.npz").exists()]
    print(f"{len(vs)} videos, {len(todo)} to do -> {dst}")
    with ProcessPoolExecutor(a.workers) as ex:
        for name, n in ex.map(extract, *zip(*todo)) if todo else []:
            print(name, n, flush=True)
