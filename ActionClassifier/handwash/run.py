"""Hand-washing step for every frame of a video, all on TensorRT engines.

video -> hands engine (hands.py, one frame per call: tracking needs the previous frame; results stay on the GPU)
      -> landmarks of the whole video [F,2,21,3] -> slots by wrist (extract.Slots, as in training)
      -> one window per frame (the seconds and frames per second the model was trained with, resampled from the video; --window native: every frame) -> TCN engine in batches
      -> step per frame, smoothed over --smooth seconds; summary of the steps; optional video with landmarks and step (H264)

python run.py video.mp4 [--out out.mp4] [--model weight/tcn] [--labels annotation.json]
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from extract import Slots
from hands import Engine, compile_engine, export, runtime
from train import STEPS, interpolate

FPS = 16  # the rate the classifier was trained at (METC)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
         (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


class Classifier:
    """TCN engine: landmarks [B,T,2,21,3], present [B,T,2] (cuda) -> logits [B,7]. Built from model.onnx on first use."""

    def __init__(self, folder, maxb=512, tmax=160):
        p = Path(folder) / "model.engine"
        if not p.exists():
            s = lambda b, t: f"landmarks:{b}x{t}x2x21x3,present:{b}x{t}x2"
            r = subprocess.run(["trtexec", f"--onnx={p.with_suffix('.onnx')}", f"--saveEngine={p}", f"--minShapes={s(1, 8)}", f"--optShapes={s(maxb, 2 * FPS)}",
                                f"--maxShapes={s(maxb, tmax)}"], capture_output=True, text=True)
            assert r.returncode == 0, "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "[E]" in l)
        e = runtime().deserialize_cuda_engine(p.read_bytes())
        self.ctx, self.maxb = e.create_execution_context(), maxb
        self.logits = torch.zeros(maxb, len(STEPS), device="cuda")
        self.label = torch.zeros(maxb, dtype=torch.int32, device="cuda")
        self.ctx.set_tensor_address("logits", self.logits.data_ptr())
        self.ctx.set_tensor_address("label", self.label.data_ptr())

    def __call__(self, lm, present):
        out = []
        for x, p in zip(lm.split(self.maxb), present.split(self.maxb)):
            x, p = x.contiguous(), p.contiguous()
            self.ctx.set_input_shape("landmarks", tuple(x.shape))
            self.ctx.set_input_shape("present", tuple(p.shape))
            self.ctx.set_tensor_address("landmarks", x.data_ptr())
            self.ctx.set_tensor_address("present", p.data_ptr())
            assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            out.append(self.logits[:len(x)].clone())
        return torch.cat(out)


def hands_engine(H, W):
    p = Path(f"cache/mp/hands_{H}x{W}.engine")
    if not p.exists():
        export(p.with_suffix(".onnx"), H, W)
        compile_engine(p.with_suffix(".onnx"), p)
    return Engine(p)


def segments(lab, fps):
    """[(step, start s, end s)] of consecutive frames with the same step."""
    out, s = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            out.append((int(lab[s]), s / fps, i / fps))
            s = i
    return out


def landmarks(video):
    """Whole video through the hands engine -> engine landmarks, presence, slot-ordered x [F,2,21,3], p [F,2], fps, H, W, ms per frame."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hands = hands_engine(H, W)
    t0 = time.time()
    lms, prs = [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        lm, pr = hands(torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).cuda())
        lms.append(lm.clone())
        prs.append(pr.clone())
    lm, pr = torch.stack(lms).cpu().numpy(), torch.stack(prs).cpu().numpy()
    ms = (time.time() - t0) * 1000 / len(lm)
    slots = Slots()
    x, p = map(np.stack, zip(*(slots([lm[i, k] for k in range(2) if pr[i, k] > 0]) for i in range(len(lm)))))
    return lm, pr, x, p, fps, H, W, ms


def classify(clf, x, p, fps, H, W, rate, seconds, smooth, anchor="end", interp=0):
    """One window per frame: `seconds` of frames 1/rate s apart ending at that frame (frames before the start repeat the first one)
    -> class probabilities [F,7] and the labels after averaging them over `smooth` seconds."""
    n = len(x)
    if interp:
        x = interpolate(x, p, round(interp * fps))
    k = round(seconds * rate)
    t = np.arange(n)[:, None] / fps - (np.arange(k)[::-1] / rate)[None]
    if anchor == "center":  # frame i is the middle of its window (frames after the end repeat the last one)
        t = t + (k - 1 - k // 2) / rate
    idx = torch.from_numpy(np.clip(np.round(t * fps), 0, n - 1).astype(np.int64)).cuda()
    x, p = torch.from_numpy(x).cuda(), torch.from_numpy(p).float().cuda()
    x[..., 0] *= W / H  # square pixels, as in training
    prob = torch.softmax(clf(x[idx], p[idx]), 1)
    k = max(1, round(smooth * fps))
    sm = torch.nn.functional.avg_pool1d(prob.T[None], k, 1, k // 2, count_include_pad=False)[0, :, :n].T
    return prob, sm


def model_window(folder):
    """(rate, seconds, anchor, interp) the model was trained with."""
    m = json.load(open(Path(folder) / "metrics.json"))
    return m.get("rate", FPS), m.get("seconds", m.get("frames", 32) / FPS), m.get("anchor", "end"), m.get("interp", 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", help="video with the step in the corner (and the true step with --labels), H264")
    ap.add_argument("--landmarks", action="store_true", help="also draw the hand landmarks in --out")
    ap.add_argument("--csv", help="per frame: time, probabilities of the 7 classes (before smoothing), smoothed label (for post.py-style post-processing)")
    ap.add_argument("--model", default="weight/tcn-r10-s1-center", help="folder of a train.py run (10 fps x 1 s labelled at its middle: best on hospital footage)")
    ap.add_argument("--smooth", type=float, default=1.0, help="seconds of class probabilities averaged (centred)")
    ap.add_argument("--labels", help="METC annotation .json: print the frame accuracy")
    ap.add_argument("--window", default="model", choices=["model", "native"],
                    help="model: frames at the rate the model was trained at (resampled from the video); native: every frame of the video")
    a = ap.parse_args()
    torch.cuda.set_stream(torch.cuda.Stream())  # TensorRT warns about (and syncs on) the default stream
    clf = Classifier(a.model)
    lm, pr, x, p, fps, H, W, ms = landmarks(a.video)
    n = len(x)
    rate, seconds, anchor, interp = model_window(a.model)
    t1 = time.time()
    prob, smooth = classify(clf, x, p, fps, H, W, fps if a.window == "native" else rate, seconds, a.smooth, anchor, interp)
    lab = smooth.argmax(1).cpu().numpy()
    t2 = time.time()
    print(f"{n} frames ({n / fps:.1f} s at {fps:.0f} fps): hands {ms:.2f} ms/frame incl. decode, "
          f"windows + classifier {(t2 - t1) * 1000:.0f} ms for all {n}")
    for s, b, e in segments(lab, fps):
        if e - b >= 0.5:
            print(f"  {b:6.1f}-{e:6.1f} s  {STEPS[s]}")
    done = {STEPS[s]: round(float((lab == s).sum()) / fps, 1) for s in range(1, len(STEPS))}
    print("seconds per step:", done)
    if a.csv:
        P = prob.cpu().numpy()
        np.savetxt(a.csv, np.column_stack([np.arange(n), np.arange(n) / fps, P, lab]), delimiter=",", fmt=["%d", "%.3f"] + ["%.4f"] * len(STEPS) + ["%d"],
                   header="frame,time," + ",".join(f"p{k}" for k in range(len(STEPS))) + ",label", comments="")
    if a.labels:
        ref = np.int64([l["code"] for l in json.load(open(a.labels))["labels"]])[:n]
        ref[ref >= len(STEPS)] = 0
        raw = prob.argmax(1).cpu().numpy()[:len(ref)]
        print(f"frame accuracy: raw {(raw == ref).mean() * 100:.1f}%, smoothed {(lab[:len(ref)] == ref).mean() * 100:.1f}%")
    if a.out:
        cap = cv2.VideoCapture(a.video)
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
        s = H / 480  # text scale
        txt = lambda f, t, y, c: [cv2.putText(f, t, (int(10 * s), int(y * s)), 0, 1.2 * s, col, th, cv2.LINE_AA) for col, th in (((0, 0, 0), max(2, int(6 * s))), (c, max(1, int(2 * s))))]
        for i in range(n):
            ok, f = cap.read()
            if a.landmarks:
                for h in range(2):
                    if pr[i, h] > 0:
                        q = (lm[i, h, :, :2] * [W, H]).astype(int)
                        for u, v in EDGES:
                            cv2.line(f, tuple(q[u]), tuple(q[v]), (0, 255, 0) if h == 0 else (255, 0, 255), 2)
            name = lambda k: "other" if k == 0 else f"step {k}"
            txt(f, name(lab[i]), 40, (255, 255, 255))
            if a.labels and i < len(ref):
                txt(f, "true: " + name(ref[i]), 80, (80, 220, 80) if ref[i] == lab[i] else (60, 60, 255))
            ff.stdin.write(f.tobytes())
        ff.stdin.close()
        ff.wait()
