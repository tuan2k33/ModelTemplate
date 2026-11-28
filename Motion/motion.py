"""Video in, motion video (white = differs from the background, H264) and the background at the last frame (image) out.
    python motion.py in.mp4 --mask out/motion.mp4 --background out/background.png [--every 5] [--form 5] [--cap 600] [--thr 25]
The mask is computed at 180x320 and scaled to the video size (nearest), the background is scaled bilinearly."""
import argparse, subprocess, time

import cv2, numpy as np, torch

from model import CACHE, VER, Engine, compile_engine, export

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--mask", required=True, help="motion video, H264")
ap.add_argument("--background", required=True, help="background image after the last frame (.png, .jpg)")
ap.add_argument("--every", type=int, default=5, help="update the background every Nth frame (the mask is computed on all of them)")
ap.add_argument("--form", type=float, default=5, help="seconds the background takes to re-form after most of the frame changes at once")
ap.add_argument("--cap", type=float, default=600, help="seconds: the most weight a background can gather, so the longest a still object stays an object")
ap.add_argument("--thr", type=float, default=25, help="a pixel matches when no channel differs by more than this (0..255)")
a = ap.parse_args()
torch.cuda.set_stream(torch.cuda.Stream())  # TensorRT syncs on the default stream

cap = cv2.VideoCapture(a.video)
fps, w0, h0 = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
form, top = (max(1, round(s * fps / a.every)) for s in (a.form, a.cap))
eng = CACHE / f"bg{VER}-{form}-{top}-{a.thr:g}.engine"  # same engine as run.py
if not eng.exists():
    export(eng.with_suffix(".onnx"), form, top, a.thr)
    compile_engine(eng.with_suffix(".onnx"), eng)
net = Engine(eng)

ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w0}x{h0}", "-r", str(fps), "-i", "-",
                      "-c:v", "libx264", "-pix_fmt", "yuv420p", a.mask], stdin=subprocess.PIPE)
i, t0 = 0, time.perf_counter()
while True:
    ok, fr = cap.read()
    if not ok:
        break
    m = net(torch.from_numpy(fr).cuda().flip(-1)[None], update=i % a.every == 0)[0].cpu().numpy()
    ff.stdin.write(cv2.resize(m * 255, (w0, h0), interpolation=cv2.INTER_NEAREST).tobytes())
    i += 1
ff.stdin.close(); ff.wait()
assert i, f"no frame read from {a.video}"
cv2.imwrite(a.background, cv2.resize(net.background().flip(-1).cpu().numpy(), (w0, h0)))  # state_out of the last frame, RGB -> BGR
print(f"{i} frames in {time.perf_counter() - t0:.1f} s -> {a.mask}, {a.background}")
