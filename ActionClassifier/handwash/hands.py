"""MediaPipe HandLandmarker (VIDEO mode, 2 hands) rebuilt as one ONNX graph for TensorRT.

uint8 frame [H,W,3] RGB + state [2,7] (zeros at the start)
  -> palm detector 192x192 (letterbox) -> 2016 anchors decoded, weighted NMS kept to 2 hands with fixed shapes
  -> per slot: hand tracked from the previous frame (state) or a new palm, rotated crop 224x224 (GridSample)
  -> landmark model, batch 2 -> landmarks back to the frame
  -> landmarks [2,21,3] (x, y in [0,1] of the frame, z), present [2], state_out [2,7] (cx, cy, w, h in pixels, cos, sin, valid)

The two networks are MediaPipe's own (hand_landmarker.task, converted with tf2onnx), weights in FP16; the steps between them
follow MediaPipe's graph: palm rect scaled 2.6 and shifted -0.5 up, tracking rect from 12 landmarks scaled 2.0 and shifted -0.1,
palm score and hand presence >= 0.5, palms that overlap a tracked hand (IoU > 0.5) dropped.
"""
import math
import subprocess
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import helper as h

MP = Path("cache/mp")
DET, LM = 192, 224
DET_THR, PRESENT_THR, NMS_IOU, TRACK_IOU = 0.5, 0.5, 0.3, 0.5
SUBSET = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]  # landmarks MediaPipe uses for the tracking rect


def anchors():
    a = []
    for stride, n in ((8, 2), (16, 6)):
        g = DET // stride
        for y in range(g):
            for x in range(g):
                a += [[(x + 0.5) / g, (y + 0.5) / g]] * n
    return torch.tensor(a)  # [2016,2]


def crop(img, cx, cy, w, h, cos, sin, size, pad):
    """img [1,3,H,W] float, rect per row (pixels, rotation as cos/sin) -> [N,3,size,size], bilinear; pad: zeros or border (repeat the edge)."""
    H, W = img.shape[2], img.shape[3]
    t = (torch.arange(size, dtype=torch.float32, device=img.device) + 0.5) / size - 0.5
    u, v = t[None, None, :], t[None, :, None]  # [1,1,S] along x, [1,S,1] along y
    r = lambda a: a[:, None, None]
    X = r(cx) + u * r(w) * r(cos) - v * r(h) * r(sin)
    Y = r(cy) + u * r(w) * r(sin) + v * r(h) * r(cos)
    grid = torch.stack([2 * X / W - 1, 2 * Y / H - 1], -1)
    return F.grid_sample(img.expand(len(cx), -1, -1, -1), grid, mode="bilinear", padding_mode=pad, align_corners=False)


def rotation(x0, y0, x1, y1):
    """MediaPipe's rect rotation (fingers up = 0), as cos/sin: 90 deg - atan2(-(y1 - y0), x1 - x0)."""
    dx, dy = x1 - x0, y1 - y0
    n = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-6)
    return -dy / n, dx / n


def transform(cx, cy, w, h, cos, sin, scale, shift_y):
    """MediaPipe RectTransformation: shift along the rect's own y, square on the long side, scale."""
    cx, cy = cx - h * shift_y * sin, cy + h * shift_y * cos
    s = torch.maximum(w, h) * scale
    return cx, cy, s, s


def iou(a, b):
    """Axis-aligned IoU of rects (cx, cy, w, h), broadcast."""
    ix = (torch.minimum(a[0] + a[2] / 2, b[0] + b[2] / 2) - torch.maximum(a[0] - a[2] / 2, b[0] - b[2] / 2)).clamp(min=0)
    iy = (torch.minimum(a[1] + a[3] / 2, b[1] + b[3] / 2) - torch.maximum(a[1] - a[3] / 2, b[1] - b[3] / 2)).clamp(min=0)
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter).clamp(min=1e-6)


class Pre(torch.nn.Module):
    """frame uint8 [H,W,3] -> palm detector input [1,192,192,3] in [0,1] (letterbox), img [1,3,H,W] in [0,1]."""

    def forward(self, frame):
        img = frame.float().permute(2, 0, 1)[None] / 255  # cast first: TensorRT does not transpose uint8
        H, W = img.shape[2], img.shape[3]
        s = torch.maximum(torch.tensor(H, dtype=torch.float32), torch.tensor(W, dtype=torch.float32))
        one, z = torch.ones(1), torch.zeros(1)
        x = crop(img, (W / 2) * one, (H / 2) * one, s * one, s * one, one, z, DET, "zeros")
        return x.permute(0, 2, 3, 1), img


class Mid(torch.nn.Module):
    """Palm detections + tracked hands -> one rect per slot, the crops for the landmark model."""

    def __init__(self):
        super().__init__()
        self.register_buffer("anchor", anchors())

    def forward(self, img, raw_box, raw_score, state):
        H, W = img.shape[2], img.shape[3]
        s = max(H, W) if isinstance(H, int) else torch.maximum(torch.tensor(H), torch.tensor(W)).float()
        ox, oy = (s - W) / 2, (s - H) / 2
        score = torch.sigmoid(raw_score[0, :, 0].clamp(-100, 100))
        b = raw_box[0] / DET
        a = self.anchor
        # palm box and keypoints 0 (wrist) and 2 (middle finger base) in frame pixels
        px = lambda v, k: (v + a[:, k]) * s - (ox if k == 0 else oy)
        bx, by, bw, bh = px(b[:, 0], 0), px(b[:, 1], 1), b[:, 2] * s, b[:, 3] * s
        k0x, k0y, k2x, k2y = px(b[:, 4], 0), px(b[:, 5], 1), px(b[:, 8], 0), px(b[:, 9], 1)
        vals = torch.stack([bx, by, bw, bh, k0x, k0y, k2x, k2y], 1)  # [2016,8]
        score = score * (score >= DET_THR)
        # weighted NMS for two hands with fixed shapes: best palm, its overlapping palms averaged by score, then the best of the rest
        picks, left = [], score
        for _ in range(2):
            i = torch.argmax(left[None], 1)  # [1]: TensorRT ArgMax needs 2-d input
            best = vals[i][0]
            m = (iou((bx, by, bw, bh), (best[0], best[1], best[2], best[3])) > NMS_IOU) * left
            avg = (vals * m[:, None]).sum(0) / m.sum().clamp(min=1e-6)
            picks.append(torch.cat([avg, left[i]]))
            left = left * (m == 0)
        p = torch.stack(picks)  # [2,9]: box, kp0, kp2, score
        cos, sin = rotation(p[:, 4], p[:, 5], p[:, 6], p[:, 7])
        cx, cy, w, h = transform(p[:, 0], p[:, 1], p[:, 2], p[:, 3], cos, sin, 2.6, -0.5)
        det = torch.stack([cx, cy, w, h, cos, sin, (p[:, 8] > 0).float()], 1)  # [2,7]
        # palms on a tracked hand are that hand
        tv = state[:, 6]
        over = (iou(tuple(det[:, None, k] for k in range(4)), tuple(state[None, :, k] for k in range(4))) > TRACK_IOU) * tv[None, :]
        dv = det[:, 6] * (over.sum(1) == 0)
        # empty slots take the new palms in score order
        d0 = dv[0] > 0
        first = torch.where(d0, det[0], det[1])
        second = det[1]
        n_new = dv[0] + dv[1]
        s0_empty = tv[0] == 0
        slot0 = torch.where(s0_empty, first * (n_new > 0), state[0])
        # slot 1: if slot 0 took a new palm, slot 1 gets the other one, else the first one
        took0 = s0_empty & (n_new > 0)
        cand1 = torch.where(took0, second * (dv[0] > 0) * (dv[1] > 0), first * (n_new > 0))
        slot1 = torch.where(tv[1] == 0, cand1, state[1])
        rect = torch.stack([slot0, slot1])  # [2,7]
        x = crop(img, rect[:, 0], rect[:, 1], rect[:, 2].clamp(min=1), rect[:, 3].clamp(min=1), rect[:, 4], rect[:, 5], LM, "border")  # border matches MediaPipe CPU best (1.5-3.4 px vs 2-5 px with zeros)
        return x.permute(0, 2, 3, 1), rect


class Post(torch.nn.Module):
    """Landmarks back to the frame, presence, rect for the next frame."""

    def forward(self, img, lm, flag, rect):
        H, W = img.shape[2], img.shape[3]
        cx, cy, w, h, cos, sin, valid = rect.unbind(1)
        q = lm.view(2, 21, 3) / LM
        u, v = q[..., 0] - 0.5, q[..., 1] - 0.5
        r = lambda a: a[:, None]
        X = r(cx) + u * r(w) * r(cos) - v * r(h) * r(sin)
        Y = r(cy) + u * r(w) * r(sin) + v * r(h) * r(cos)
        Z = q[..., 2] * r(w)
        present = (flag[:, 0] >= PRESENT_THR) * valid
        # next rect, MediaPipe HandLandmarksToRect
        x1 = ((X[:, 5] + X[:, 13]) / 2 + X[:, 9]) / 2
        y1 = ((Y[:, 5] + Y[:, 13]) / 2 + Y[:, 9]) / 2
        c, s = rotation(X[:, 0], Y[:, 0], x1, y1)
        sx, sy = X[:, SUBSET], Y[:, SUBSET]
        ax = (sx.amax(1) + sx.amin(1)) / 2
        ay = (sy.amax(1) + sy.amin(1)) / 2
        dx, dy = sx - r(ax), sy - r(ay)
        rx, ry = dx * r(c) + dy * r(s), -dx * r(s) + dy * r(c)  # rotate by -rotation
        pcx, pcy = (rx.amax(1) + rx.amin(1)) / 2, (ry.amax(1) + ry.amin(1)) / 2
        nw, nh = rx.amax(1) - rx.amin(1), ry.amax(1) - ry.amin(1)
        ncx, ncy = pcx * c - pcy * s + ax, pcx * s + pcy * c + ay
        ncx, ncy, nw, nh = transform(ncx, ncy, nw, nh, c, s, 2.0, -0.1)
        nxt = torch.stack([ncx, ncy, nw, nh, c, s, present], 1)
        # two slots on the same hand: keep the first
        same = (iou(tuple(nxt[0, k] for k in range(4)), tuple(nxt[1, k] for k in range(4))) > TRACK_IOU) * present[0] * present[1]
        keep = torch.stack([present[0], present[1] * (1 - same)])
        nxt = nxt * keep[:, None]
        out = torch.stack([X / W, Y / H, Z / W], -1) * keep[:, None, None]
        return out, keep, nxt


class Eager:
    """The same pipeline with onnxruntime (or any callables) for the two networks, to check against MediaPipe."""

    def __init__(self, det, lm, device="cpu"):
        self.det, self.lm, self.dev = det, lm, device
        self.pre, self.mid, self.post = Pre(), Mid().to(device), Post()
        self.state = torch.zeros(2, 7, device=device)

    def __call__(self, frame):
        x, img = self.pre(torch.as_tensor(frame, device=self.dev))
        box, score = self.det(x)
        crops, rect = self.mid(img, box, score, self.state)
        lm, flag = self.lm(crops)
        out, keep, self.state = self.post(img, lm, flag, rect)
        return out, keep


def networks():
    """MediaPipe's two networks as FP16 ONNX with float32 inputs/outputs, landmark model at batch 2 (cache/mp/)."""
    from onnxconverter_common import float16
    out = {}
    for name, src in (("det", "hand_detector"), ("lm", "hand_landmarks_detector")):
        p = MP / f"{src}_fp16.onnx"
        if name == "lm":
            p = MP / f"{src}_b2_fp16.onnx"
        if not p.exists():
            if not (MP / f"{src}.onnx").exists():
                if not (MP / f"{src}.tflite").exists():
                    import zipfile
                    zipfile.ZipFile("cache/hand_landmarker.task").extractall(MP)
                subprocess.run(["uv", "run", "--no-project", "--python", "3.12", "--with", "tf2onnx", "--with", "tensorflow-cpu", "python", "-m", "tf2onnx.convert",
                                "--tflite", str(MP / f"{src}.tflite"), "--output", str(MP / f"{src}.onnx"), "--opset", "17"], check=True, capture_output=True)
            m = onnx.load(MP / f"{src}.onnx")
            if name == "lm":
                for v in list(m.graph.input) + list(m.graph.output):
                    v.type.tensor_type.shape.dim[0].dim_value = 2
            onnx.save(float16.convert_float_to_float16(m, keep_io_types=True), p)
        out[name] = onnx.load(p)
    return out


def export(path, height=480, width=640, opset=17, det_thr=DET_THR, present_thr=PRESENT_THR):
    """One ONNX: frame uint8 [height,width,3] + state [2,7] -> landmarks [2,21,3], present [2], state_out [2,7].
    det_thr / present_thr: palm score for a new hand and landmark presence to keep one (baked into the graph)."""
    global DET_THR, PRESENT_THR
    DET_THR, PRESENT_THR = det_thr, present_thr
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / "tmp.onnx"
    frame, img = torch.zeros(height, width, 3, dtype=torch.uint8), torch.zeros(1, 3, height, width)
    parts = []
    for mod, args, ins, outs in (
            (Pre(), (frame,), ["frame"], ["det_in", "img"]),
            (Mid(), (img, torch.zeros(1, 2016, 18), torch.zeros(1, 2016, 1), torch.zeros(2, 7)), ["img", "raw_box", "raw_score", "state"], ["crops", "rect"]),
            (Post(), (img, torch.zeros(2, 63), torch.zeros(2, 1), torch.zeros(2, 7)), ["img", "lm", "flag", "rect"], ["landmarks", "present", "state_out"])):
        torch.onnx.export(mod, args, str(tmp), dynamo=False, opset_version=opset, input_names=ins, output_names=outs)
        parts.append((onnx.load(tmp), {k: k for k in ins + outs}))
    tmp.unlink()
    nets = networks()
    parts.insert(1, (nets["det"], {"input_1": "det_in", "Identity": "raw_box", "Identity_1": "raw_score"}))
    parts.insert(3, (nets["lm"], {"input_1": "crops", "Identity": "lm", "Identity_1": "flag"}))
    nodes, inits = [], []
    for k, (m, keep) in enumerate(parts):
        name = lambda n: keep.get(n, f"p{k}/{n}") if n else n
        for n in m.graph.node:
            n.name = name(n.name)
            n.input[:], n.output[:] = [name(i) for i in n.input], [name(o) for o in n.output]
        for i in m.graph.initializer:
            i.name = name(i.name)
        nodes += m.graph.node
        inits += m.graph.initializer
    first, last = parts[0][0].graph, parts[-1][0].graph
    g = h.make_graph(nodes, "hands", [first.input[0], parts[2][0].graph.input[3]], list(last.output), inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", opset)])
    onnx.checker.check_model(m)
    onnx.save(m, path)


def runtime():
    """One TensorRT runtime per process (a second logger only makes TensorRT warn)."""
    global _RT
    if "_RT" not in globals():
        import tensorrt as trt
        _RT = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    return _RT


def compile_engine(onnx_path, engine_path):
    r = subprocess.run(["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}"], capture_output=True, text=True)
    assert r.returncode == 0, "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "[E]" in l)


class Engine:
    """uint8 frame [H,W,3] cuda tensor -> landmarks [2,21,3], present [2] (cuda tensors, reused between calls). Keeps the tracking state of one camera.
    The engine runs as a CUDA graph: launched from Python it costs ~4.5 ms of CPU per frame (200+ small layers), replayed 0.01 ms.
    Call it on a non-default CUDA stream (torch.cuda.set_stream); nothing is synchronized, the outputs are ready in stream order."""

    def __init__(self, path):
        e = runtime().deserialize_cuda_engine(Path(path).read_bytes())
        self.ctx = e.create_execution_context()
        self.state, self.next = torch.zeros(2, 7, device="cuda"), torch.zeros(2, 7, device="cuda")
        self.lm, self.present = torch.zeros(2, 21, 3, device="cuda"), torch.zeros(2, device="cuda")
        self.frame = torch.zeros(tuple(e.get_tensor_shape("frame")), dtype=torch.uint8, device="cuda")
        for n, t in (("frame", self.frame), ("state", self.state), ("state_out", self.next), ("landmarks", self.lm), ("present", self.present)):
            self.ctx.set_tensor_address(n, t.data_ptr())
        s = torch.cuda.current_stream()
        assert self.ctx.execute_async_v3(s.cuda_stream)  # first run outside the graph (TensorRT may allocate)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=s):
            self.ctx.execute_async_v3(s.cuda_stream)
            self.state.copy_(self.next)
        self.state.zero_()

    def __call__(self, frame):
        self.frame.copy_(frame)
        self.graph.replay()
        return self.lm, self.present
