"""
Model keypoint to giay (YOLO26n-pose) train tren UVDoc_final, de nan ve A4 (dewarp.dewarp_kpts / dewarp_grid).

--layout edge: 16 keypoint, chay vong theo chieu kim dong ho, canh 0 luon la mot canh NGAN cua trang (model hoc tu noi dung, khong
dua vao do dai trong anh); trong 2 goc mo dau canh ngan, chon goc gan goc tren-trai cua anh hon. Trang co the nguoc
180 do: xoay dung chieu de buoc sau. Moi canh 4 diem (goc + 3 diem chia canh thanh 4 doan BANG NHAU TREN
TRANG PHANG, lay tu grid2d). Do tren UVDoc benchmark voi keypoint that: 5 diem/canh lech 13.6 px, bang 9 diem/canh;
diem chia deu theo mep trong anh (kieu lay tu mask) 38-49 px.
--layout RxC (vd 7x5): luoi R hang x C cot cach deu tren trang phang, ca diem BEN TRONG trang (sua duoc cho cong o
giua); voi luoi that: 5x4 7.5 px, 7x5 5.2 px, 9x7 2.5 px. Hang 0 = canh ngan, goc [0,0] chon nhu tren.

    python paper_kpt.py --layout edge --epochs 30     # -> weight/kpt-edge/weights/best.pt
    python paper_kpt.py --layout 7x5 --epochs 30      # -> weight/kpt-7x5/
    python paper_kpt.py --layout 7x5 --epochs 0 --model weight/kpt-7x5/weights/best.pt   # chi do tren benchmark
"""

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image

N = 4                                   # diem moi canh (tinh ca goc dau)
SRC = Path("data/uvdoc/UVDoc_final")
BENCH = Path("data/uvdoc/UVDoc_benchmark")
DS = Path("data/uvdoc/kpt")                # nhan YOLO o ext4 (data -> ~/.dev/data), anh la symlink


def grid(root, stem):
    name = json.load(open(root / "metadata_sample" / f"{stem}.json"))["geom_name"]
    return np.array(h5py.File(root / "grid2d" / f"{name}.mat")["grid2d"]).transpose(2, 1, 0)   # (hang, cot, xy)


def canon(g):
    """Luoi UVDoc (hang 0 = canh ngan, cung chieu kim dong ho trong anh tru khi anh bi lat) -> luoi chuan: lat cot neu
    nguoc chieu (flip_* trong metadata_geom), xoay 180 neu goc [-1,-1] gan goc tren-trai anh hon goc [0,0]."""
    x, y = np.array([g[0, 0], g[0, -1], g[-1, -1], g[-1, 0]]).T
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        g = g[:, ::-1]
    return g if g[0, 0].sum() <= g[-1, -1].sum() else g[::-1, ::-1]


def kpts_of(g, n=N):
    """n diem moi canh, cach deu tren trang phang, vong [0,0] -> [0,-1] -> [-1,-1] -> [-1,0]."""
    t = np.linspace(0, 1, n, endpoint=False)
    sides = [g[0], g[:, -1], g[-1, ::-1], g[::-1, 0]]
    return np.concatenate([np.stack([np.interp(t, np.linspace(0, 1, len(s)), s[:, i]) for i in (0, 1)], 1) for s in sides])


def sub(g, r, c):
    """Luoi con r x c cach deu tren trang phang."""
    ys, xs = np.linspace(0, g.shape[0] - 1, r), np.linspace(0, g.shape[1] - 1, c)
    return cv2.remap(np.float32(g), *np.meshgrid(np.float32(xs), np.float32(ys)), cv2.INTER_LINEAR)


def points(g, layout):
    g = canon(g)
    return kpts_of(g) if layout == "edge" else sub(g, *map(int, layout.split("x"))).reshape(-1, 2)


def label(g, w, h, layout):
    k = points(g, layout) / [w, h]
    vis = np.where(((k >= 0) & (k <= 1)).all(1), 2, 1)          # ngoai anh: van co toa do (kep vao bien), danh dau 1
    k = k.clip(0, 1)
    allp = (g.reshape(-1, 2) / [w, h]).clip(0, 1)
    (x0, y0), (x1, y1) = allp.min(0), allp.max(0)
    return " ".join(map(str, [0, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, *np.column_stack([k, vis]).ravel().round(6)]))


def prepare(layout, n_val=1000):
    ds = DS.with_name(f"kpt-{layout}")
    if (ds / "data.yaml").exists():
        return ds / "data.yaml"
    stems = sorted(p.stem for p in (SRC / "img").glob("*.png"))
    random.Random(0).shuffle(stems)
    for split, ss in (("val", stems[:n_val]), ("train", stems[n_val:])):
        (ds / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / split).mkdir(parents=True, exist_ok=True)
        for s in ss:
            src = (SRC / "img" / f"{s}.png").resolve()
            w, h = Image.open(src).size                                   # chi doc header
            dst = ds / "images" / split / f"{s}.png"
            if not dst.exists():
                os.symlink(src, dst)
            (ds / "labels" / split / f"{s}.txt").write_text(label(grid(SRC, s), w, h, layout) + "\n")
    nk = len(points(grid(SRC, stems[0]), layout))
    (ds / "data.yaml").write_text(f"path: {ds.resolve()}\ntrain: images/train\nval: images/val\nkpt_shape: [{nk}, 3]\nnames: {{0: paper}}\n")
    print(len(stems), "samples")
    return ds / "data.yaml"


def bench(model, layout):
    """Nan bang keypoint du doan tren UVDoc benchmark, sai so nhu bench_uvdoc.py (SIFT, px tren A4), tach theo kieu bien dang."""
    from bench_uvdoc import err, sift
    from dewarp import dewarp_grid, dewarp_kpts
    rows = []
    for f in sorted((BENCH / "img").glob("*.png")):
        img = cv2.imread(str(f))
        gt = cv2.imread(str(BENCH / "texture_sample" / f.name))
        gkp, gdes = sift.detectAndCompute(cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY), None)
        geom = json.load(open(BENCH / "metadata_sample" / f"{f.stem}.json"))["geom_name"]
        info = json.load(open(BENCH / "metadata_geom" / f"{geom}.json"))["sample_info"]
        kind = "folded/crumpled" if info["Folded"] or info["Crumpled"] else "curved/perspective"
        r = model.predict(img, imgsz=640, verbose=False)[0]
        if not len(r.boxes):
            rows.append((kind, np.nan, np.nan)); continue
        k = r.keypoints.xy[r.boxes.conf.argmax()].cpu().numpy()
        ke = np.median(np.hypot(*(k - points(grid(BENCH, f.stem), layout)).T))
        out = dewarp_kpts(img, k) if layout == "edge" else dewarp_grid(img, k.reshape(*map(int, layout.split("x")), 2))
        # trang co the nguoc 180 do (xoay dung chieu la buoc sau): lay huong khop ban goc hon
        rows.append((kind, ke, np.nanmin([err(np.rot90(out, r).copy(), gkp, gdes, (gt.shape[1], gt.shape[0]))[0] for r in (0, 2)])))
    for kind in ("curved/perspective", "folded/crumpled", None):
        sel = [x for x in rows if kind in (None, x[0])]
        print(f"{kind or 'all':20s} n={len(sel):2d}  keypoint err {np.nanmedian([x[1] for x in sel]):5.1f} px (anh goc)  "
              f"dewarp {np.nanmedian([x[2] for x in sel]):5.1f} px  fail {int(np.isnan([x[2] for x in sel]).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="edge", help="edge (16 diem quanh mep) hoac RxC (luoi, vd 7x5)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--model", default="yolo26n-pose.pt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true", help="train tiep tu weight/kpt-<layout>/weights/last.pt")
    a = ap.parse_args()
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)    # forkserver (mac dinh py3.14) lam worker dataloader Ultralytics chet
    from ultralytics import YOLO
    m = YOLO(a.model)
    name = f"kpt-{a.layout}"
    if a.resume:
        YOLO(f"weight/{name}/weights/last.pt").train(resume=True)
        m = YOLO(f"weight/{name}/weights/best.pt")
    elif a.epochs:
        m.train(data=str(prepare(a.layout)), epochs=a.epochs, imgsz=640, batch=a.batch, workers=a.workers,
                project=str(Path("weight").resolve()), name=name, exist_ok=True,
                fliplr=0.0, mosaic=0.0)                      # lat ngang lam nguoc thu tu diem; moi anh chi 1 trang
        m = YOLO(f"weight/{name}/weights/best.pt")
    bench(m, a.layout)
