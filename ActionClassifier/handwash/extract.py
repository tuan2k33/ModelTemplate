"""Hand landmarks of the METC hand-washing videos: MediaPipe HandLandmarker (CPU) or YOLO26 hand pose (GPU).

One cache/kpt-<mode>-<conf>/<video>.npz per video:
  lm      [F,2,21,3] float32  x, y in [0,1] of the frame, z relative depth (yolo: keypoint confidence); 2 slots kept apart by wrist position
  present [F,2] uint8          hand found in that slot
  label   [F] int8             WHO movement code (0 = other, 1-6 the steps, 7 faucet)

python extract.py [--mode video|image|yolo] [--conf 0.5] [--limit N] [--workers 6]
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

DATA = Path("data/handwash/metc")
TASK = "cache/hand_landmarker.task"
YOLO_HAND = "cache/yolo26_hand_pose.pt"  # https://github.com/marceloeatworld/yolo26-training (Git LFS)
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


def yolo_hands(v, conf):
    """YOLO26s hand pose (marceloeatworld/yolo26-training) on the GPU; third channel is the keypoint confidence."""
    from ultralytics import YOLO
    for r in YOLO(YOLO_HAND).predict(str(v), stream=True, conf=conf, half=True, verbose=False, device=0):
        k = r.keypoints
        yield [np.concatenate([xy, c[:, None]], 1) for xy, c in zip(k.xyn.cpu().numpy()[:2], k.conf.cpu().numpy()[:2])]


def extract(v, ann, conf, mode, out):
    label = np.int8([l["code"] for l in json.load(open(ann))["labels"]])
    lm, present = [], []
    wrist = np.float32([[0.3, 0.5], [0.7, 0.5]])  # last wrist seen per slot
    for hands in (yolo_hands(v, conf) if mode == "yolo" else mediapipe_hands(v, conf, mode)):
        # left/right from the models flips when seen from above, so slots follow the wrists instead
        d = np.float32([[np.linalg.norm(h[0, :2] - w) for w in wrist] for h in hands])
        if len(hands) == 1:
            slots = [int(d[0].argmin())]
        elif len(hands) == 2:
            slots = [0, 1] if d[0, 0] + d[1, 1] <= d[0, 1] + d[1, 0] else [1, 0]
        else:
            slots = []
        x = np.zeros((2, 21, 3), np.float32)
        p = np.zeros(2, np.uint8)
        for k, h in zip(slots, hands):
            x[k], p[k], wrist[k] = h, 1, h[0, :2]
        lm.append(x)
        present.append(p)
    n = min(len(lm), len(label))
    np.savez_compressed(out, lm=np.stack(lm)[:n], present=np.stack(present)[:n], label=label[:n])
    return v.name, n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--mode", default="video", choices=["video", "image", "yolo"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if not Path(TASK).exists():
        Path(TASK).parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(URL, TASK)
    vs = videos()[:a.limit]
    dst = Path(f"cache/kpt-{a.mode}-{a.conf}")
    dst.mkdir(parents=True, exist_ok=True)
    todo = [(v, an, a.conf, a.mode, dst / f"{v.stem}.npz") for v, an in vs if not (dst / f"{v.stem}.npz").exists()]
    print(f"{len(vs)} videos, {len(todo)} to do -> {dst}")
    with ProcessPoolExecutor(a.workers) as ex:
        for name, n in ex.map(extract, *zip(*todo)) if todo else []:
            print(name, n, flush=True)
