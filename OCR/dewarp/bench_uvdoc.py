"""
Do do nan tren UVDoc benchmark (50 anh chup giay cong + ban phang goc), data/uvdoc/UVDoc_benchmark.
Mep giay: tu detect_boundary ("det-") hoac mep that tu grid2d ("gt-", do rieng phan nan);
"gt-grid" = remap theo ca luoi that (tran duoi cua thuoc do).
Sai so = do lech (px, khung A4 1241x1754) giua diem SIFT khop cua anh da nan va ban phang:
trung vi tung anh, roi trung vi/p90 tren ca bo. So sanh: warp_to_a4 (homography 4 goc) vs dewarp (Coons).

Dung: python bench_uvdoc.py [--out debug_out/uvdoc]
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import h5py

from boundary import detect_boundary, warp_to_a4
from dewarp import dewarp_kpts, dewarp_to_a4

B = Path("data/uvdoc/UVDoc_benchmark")
sift = cv2.SIFT_create(4000)
bf = cv2.BFMatcher()


def err(out, gt_kp, gt_des, size):
    g = cv2.cvtColor(cv2.resize(out, size), cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(g, None)
    if des is None:
        return np.nan, 0
    m = [a for a, b in bf.knnMatch(des, gt_des, k=2) if a.distance < 0.75 * b.distance]
    if len(m) < 10:
        return np.nan, len(m)
    d = np.array([np.hypot(*np.subtract(kp[x.queryIdx].pt, gt_kp[x.trainIdx].pt)) for x in m])
    return float(np.median(d)), len(m)


def gt_boundary(name):
    """Mep that: luoi (89 hang, 61 cot, xy) -> 4 canh TL->TR->BR->BL, moi canh 101 diem (goc dau + 100)."""
    g = np.array(h5py.File(B / "grid2d" / f"{name}.mat")["grid2d"]).transpose(2, 1, 0).astype(np.float32)
    sides = [g[0], g[:, -1], g[-1, ::-1], g[::-1, 0]]
    t = np.linspace(0, 1, 101)
    rs = [np.stack([np.interp(t, np.linspace(0, 1, len(sd)), sd[:, i]) for i in (0, 1)], 1) for sd in sides]
    return g, np.concatenate(rs).astype(np.float32), np.float32([g[0, 0], g[0, -1], g[-1, -1], g[-1, 0]])


def gt_kpts(g, n):
    """n diem moi canh, cach deu tren trang phang, TL -> TR -> BR -> BL."""
    t = np.linspace(0, 1, n, endpoint=False)
    sides = [g[0], g[:, -1], g[-1, ::-1], g[::-1, 0]]
    return np.concatenate([np.stack([np.interp(t, np.linspace(0, 1, len(sd)), sd[:, i]) for i in (0, 1)], 1) for sd in sides])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="debug_out/uvdoc")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {k: [] for k in ("det-homography", "det-coons", "gt-homography", "gt-coons", "gt-kpt8", "gt-kpt16", "gt-grid")}
    t_det, t_warp = [], []
    for f in sorted((B / "img").glob("*.png")):
        img = cv2.imread(str(f))
        gt = cv2.imread(str(B / "texture_sample" / f.name))
        size = (gt.shape[1], gt.shape[0])
        gkp, gdes = sift.detectAndCompute(cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY), None)
        t0 = time.perf_counter()
        contour, quad = detect_boundary(img)
        t1 = time.perf_counter()
        co = dewarp_to_a4(img, contour, quad)
        t_warp.append(time.perf_counter() - t1)
        t_det.append(t1 - t0)
        ho = warp_to_a4(img, quad)
        g, gc, gq = gt_boundary(json.load(open(B / "metadata_sample" / f"{f.stem}.json"))["geom_name"])
        gho, gco = warp_to_a4(img, gq), dewarp_to_a4(img, gc, gq)
        mx, my = (cv2.resize(g[..., i], size) for i in (0, 1))
        ggr = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
        outs = dict(zip(res, (ho, co, gho, gco, dewarp_kpts(img, gt_kpts(g, 8)), dewarp_kpts(img, gt_kpts(g, 16)), ggr)))
        for k, o in outs.items():
            res[k].append(err(o, gkp, gdes, size))
        cv2.imwrite(str(out / f"{f.stem}.jpg"), np.hstack([cv2.resize(x, (400, 566)) for x in (img, *outs.values(), gt)]))
        print(f.stem, *(f"{k} {res[k][-1][0]:5.1f}" for k in res), flush=True)

    for k, v in res.items():
        e = np.array([x[0] for x in v])
        print(f"{k:15s} median {np.nanmedian(e):.1f}px  p90 {np.nanpercentile(e, 90):.1f}px  failed {np.isnan(e).sum()}")
    for d in ("det", "gt"):
        c, h = (np.array([x[0] for x in res[f"{d}-{k}"]]) for k in ("coons", "homography"))
        print(f"{d}: coons better on {(c < h).sum()} / {len(c)} images")
    print(f"CPU: detect_boundary {1000 * np.median(t_det):.0f} ms, coons remap {1000 * np.median(t_warp):.0f} ms (median)")
