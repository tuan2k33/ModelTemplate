"""Hand-washing step from a window of hand landmarks (cache of extract.py).

Window: --seconds at --rate frames per second (subsampled from METC's 16 fps).
Each frame: 2 hands x 21 landmarks x (x, y, z) + 2 presence flags. A missing hand is filled (--fill):
  zero  zeros
  copy  the other hand when it is there (hands on top of each other), else zeros
  hold  the last place it was seen in the window, else as copy
The presence flags always go in, so the model knows what was filled. Training drops a hand on a random
span with --drop probability and fills it the same way. Label of the window: its last or middle frame (--anchor);
randomly rotated and mirrored (--rot),
normalized by the window's mean wrist and hand size; frame-to-frame differences are appended.
Validation: the subjects with id % 5 == 0 (each subject washed at all three sinks).

python train.py [--kpt cache/kpt-video-0.5] [--head mlp|tcn] [--fill zero|copy|hold] [--drop 0.3] [--rot 180] [--name run]
-> weight/<name>/model.pt, model.onnx (see Window), metrics.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STEPS = ["other", "palm to palm", "palm over dorsum, interlaced", "palm to palm, interlaced",
         "backs of fingers to palm", "thumb rubbing", "fingertips to palm"]


def load(kpt):
    seqs = []
    for f in sorted(Path(kpt).glob("*.npz")):
        m = re.match(r".*_\d\d-\d\d-\d\d-(\d+)", f.stem)
        d = np.load(f)
        lab = d["label"].astype(np.int64)
        lab[lab >= len(STEPS)] = 0  # 7 (faucet) never occurs here
        seqs.append((int(m.group(1)) if m else -1, d["lm"], d["present"], lab))
    return seqs


def windows(seqs, frames, stride, dev):
    """Concatenated sequences on the GPU plus the start index of every window."""
    lm = torch.from_numpy(np.concatenate([s[1] for s in seqs])).to(dev)
    lm[..., 0] *= 640 / 480  # square pixels, so rotations stay rotations
    pr = torch.from_numpy(np.concatenate([s[2] for s in seqs])).float().to(dev)
    lab = torch.from_numpy(np.concatenate([s[3] for s in seqs])).to(dev)
    starts, o = [], 0
    for s in seqs:
        n = len(s[3])
        starts.append(np.arange(o, o + n - frames + 1, stride))
        o += n
    return lm, pr, lab, torch.from_numpy(np.concatenate(starts)).to(dev)


def fill(x, p, mode):
    """x [B,T,2,21,3], p [B,T,2] -> x with missing hands filled."""
    x = x * p[..., None, None]
    if mode == "zero":
        return x
    other = x.flip(2) * (p.flip(2) * (1 - p))[..., None, None]
    if mode == "hold":
        t = torch.arange(x.shape[1], device=x.device)[None, :, None] * p.long() - (1 - p.long())  # -1 where missing
        last = t.cummax(1).values
        held = torch.gather(x, 1, last.clamp(min=0)[..., None, None].expand_as(x))
        seen = (last >= 0).float()
        miss = (1 - p) * seen
        x = x + held * miss[..., None, None]
        other = other * (1 - seen)[..., None, None]
    return x + other


def features(x, p):
    """Normalize by the window's mean wrist and hand size (wrist -> middle knuckle); append differences."""
    have = (x[..., 0, :2].abs().sum(-1) > 0).float()  # [B,T,2] slots holding a hand after filling
    n = have.sum((1, 2)).clamp(min=1)
    c = (x[..., 0, :2] * have[..., None]).sum((1, 2)) / n[:, None]
    s = ((x[..., 9, :2] - x[..., 0, :2]).norm(dim=-1) * have).sum((1, 2)) / n
    s = s.clamp(min=1e-3)
    xy = (x[..., :2] - c[:, None, None, None]) / s[:, None, None, None, None]
    z = x[..., 2:] / s[:, None, None, None, None]
    f = torch.cat([xy, z], -1) * have[..., None, None]
    f = f.flatten(2)
    d = torch.cat([torch.zeros_like(f[:, :1]), f[:, 1:] - f[:, :-1]], 1)
    return torch.cat([f, d, p], -1)  # [B,T,254+2]


class MLP(nn.Module):
    def __init__(self, t, c, k, w=512):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(t * c, w), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(w, w // 2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(w // 2, k))

    def forward(self, f):
        return self.net(f)


class TCN(nn.Module):
    def __init__(self, t, c, k, w=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, w, 5, padding=2), nn.BatchNorm1d(w), nn.ReLU(),
            nn.Conv1d(w, w, 5, padding=2, stride=2), nn.BatchNorm1d(w), nn.ReLU(),
            nn.Conv1d(w, w, 5, padding=2, stride=2), nn.BatchNorm1d(w), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(w, k))

    def forward(self, f):
        return self.net(f.transpose(1, 2))


class Window(nn.Module):
    """What the engine runs: landmarks [B,T,2,21,3] (x in units of the frame height, i.e. x * W / H) and present [B,T,2]
    -> logits [B,7], label [B] int32. The filling, normalization and differences are inside."""

    def __init__(self, model, mode):
        super().__init__()
        self.model, self.mode = model, mode

    def forward(self, landmarks, present):
        logits = self.model(features(fill(landmarks, present, self.mode), present))
        return logits, logits.argmax(1).int()


def export(model, mode, frames, path):
    """model.onnx, any batch (one row per window) and window length (trained at `frames`). FP32: 330k parameters, 0.27 ms for 64 windows; the FP16 conversion trips
    TensorRT on mixed types in the filling/normalization and is not worth it."""
    w = Window(model, mode).eval().cpu()
    torch.onnx.export(w, (torch.zeros(1, frames, 2, 21, 3), torch.ones(1, frames, 2)), str(path), dynamo=False, opset_version=17,
                      input_names=["landmarks", "present"], output_names=["logits", "label"],
                      dynamic_axes={"landmarks": {0: "B", 1: "T"}, "present": {0: "B", 1: "T"}, "logits": {0: "B"}, "label": {0: "B"}})


def batch(lm, pr, lab, st, idx, frames, mode, drop=0.0, rot=0.0, at=-1):
    t = st[idx][:, None] + torch.arange(frames, device=st.device)
    x, p = lm[t], pr[t]
    if rot:  # camera turned or mirrored: random rotation of the whole window, flip half of them
        b = len(idx)
        a = (torch.rand(b, device=st.device) * 2 - 1) * np.radians(rot)
        r = torch.stack([torch.cos(a), -torch.sin(a), torch.sin(a), torch.cos(a)], -1).view(b, 1, 1, 1, 2, 2)
        r = r * torch.where(torch.rand(b, device=st.device) < 0.5, -1.0, 1.0).view(b, 1, 1, 1, 1, 1) ** torch.tensor([[1.0], [0.0]], device=st.device)
        x = torch.cat([(r @ x[..., :2, None])[..., 0], x[..., 2:]], -1)
    if drop:
        b = len(idx)
        a = torch.randint(0, frames, (b, 1), device=st.device)
        e = a + torch.randint(1, frames + 1, (b, 1), device=st.device)
        tt = torch.arange(frames, device=st.device)
        span = ((tt >= a) & (tt < e)).float() * (torch.rand(b, 1, device=st.device) < drop).float()
        hand = F.one_hot(torch.randint(0, 2, (b,), device=st.device), 2).float()
        p = p * (1 - span[..., None] * hand[:, None])
    return features(fill(x, p, mode), p), lab[t[:, at]]


def evaluate(model, data, frames, mode, at=-1):
    lm, pr, lab, st = data
    model.eval()
    pred, y = [], []
    with torch.no_grad():
        for i in torch.arange(len(st), device=st.device).split(4096):
            f, l = batch(lm, pr, lab, st, i, frames, mode, at=at)
            pred.append(model(f).argmax(1))
            y.append(l)
    pred, y = torch.cat(pred), torch.cat(y)
    per = [(pred[y == k] == k).float().mean().item() for k in range(len(STEPS))]
    return (pred == y).float().mean().item(), float(np.mean(per)), per


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kpt", default="cache/kpt-video-0.5")
    ap.add_argument("--head", default="mlp", choices=["mlp", "tcn"])
    ap.add_argument("--fill", default="copy", choices=["zero", "copy", "hold"])
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--rot", type=float, default=180, help="random rotation range in degrees when training, 0 = off")
    ap.add_argument("--seconds", type=float, default=2, help="window length")
    ap.add_argument("--anchor", default="end", choices=["end", "center"],
                    help="which frame of the window its label comes from: end (live, ~seconds/2 late) or center (offline, needs seconds/2 of future)")
    ap.add_argument("--rate", type=int, default=16, help="frames per second inside the window, at most 16 (METC's rate; resampled to the nearest frame)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda"
    a.frames = round(a.rate * a.seconds)
    at = a.frames // 2 if a.anchor == "center" else -1
    # resample METC's 16 fps to --rate (nearest frame), every phase of it: 8 fps -> frames 0,2,4.. and 1,3,5..; 10 fps -> 0,2,3,5,6.. and 1,2,4,..
    P = max(1, round(16 / a.rate))
    take = lambda n, o: np.unique(np.minimum(np.round((np.arange(int(n * a.rate / 16)) + o / P) * 16 / a.rate), n - 1).astype(int))
    seqs = [(i, lm[k], p[k], l[k]) for i, lm, p, l in load(a.kpt) for o in range(P) for k in [take(len(l), o)]]
    val = [s for s in seqs if s[0] % 5 == 0]
    tr = windows([s for s in seqs if s[0] % 5], a.frames, 2, dev)
    va = windows(val, a.frames, 4, dev)
    c = features(*[torch.zeros(1, a.frames, 2, 21, 3, device=dev), torch.ones(1, a.frames, 2, device=dev)]).shape[-1]
    model = (MLP if a.head == "mlp" else TCN)(a.frames, c, len(STEPS)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-2)
    steps = a.epochs * (len(tr[3]) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    print(f"{len(seqs)} videos, train windows {len(tr[3])}, val windows {len(va[3])} ({len(val)} videos), "
          f"{sum(p.numel() for p in model.parameters())} params")
    for ep in range(a.epochs):
        model.train()
        for idx in torch.randperm(len(tr[3]), device=dev)[:steps // a.epochs * 256].split(256):
            f, y = batch(*tr, idx, a.frames, a.fill, a.drop, a.rot, at)
            loss = F.cross_entropy(model(f), y, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if ep % 5 == 4 or ep == a.epochs - 1:
            acc, macro, per = evaluate(model, va, a.frames, a.fill, at)
            print(f"epoch {ep + 1}: loss {loss.item():.3f}  val acc {acc:.3f}  macro {macro:.3f}  " +
                  " ".join(f"{p:.2f}" for p in per), flush=True)
    if a.name:
        out = Path("weight") / a.name
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "model.pt")
        export(model, a.fill, a.frames, out / "model.onnx")
        json.dump(dict(vars(a), acc=acc, macro=macro, per_class=dict(zip(STEPS, per))), open(out / "metrics.json", "w"), indent=1)
