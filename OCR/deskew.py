"""Skew angle of a scanned page (text lines vs horizontal, within +-max_deg), two classical estimators, and a benchmark on SROIE test receipts rotated
by known random angles.   python deskew.py [--n 100] [--max 15]
profile: ink pixels (Otsu) projected onto the vertical axis for each candidate angle; text lines give the spikiest row histogram (sum of squares).
fourier: text lines put energy in the spectrum along the direction perpendicular to them; the angle of the strongest ray of |FFT| is the skew."""
import argparse, time
from pathlib import Path

import cv2, numpy as np


def ink(gray, side=1000):
    r = side / max(gray.shape)
    g = cv2.resize(gray, None, fx=r, fy=r, interpolation=cv2.INTER_AREA) if r < 1 else gray
    return g, cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


def skew_profile(gray, max_deg=15.0):
    _, b = ink(gray)
    y, x = np.nonzero(b)
    if len(x) > 60000:
        k = np.random.default_rng(0).choice(len(x), 60000, replace=False); y, x = y[k], x[k]
    x, y = x - x.mean(), y - y.mean()

    def score(a):
        t = np.deg2rad(a)
        yr = y * np.cos(t) + x * np.sin(t)  # row of each ink pixel once the page is rotated back by a
        hgt = np.bincount((yr - yr.min()).astype(int))
        return float((hgt.astype(np.float64) ** 2).sum())
    best = 0.0
    for span, step in ((max_deg, 0.5), (0.5, 0.05)):
        grid = np.arange(best - span, best + span + 1e-9, step)
        best = grid[np.argmax([score(a) for a in grid])]
    return float(best)


def skew_fourier(gray, max_deg=15.0, step=0.05):
    g, _ = ink(gray)
    n = 1024
    im = np.full((n, n), 255, np.float32); im[:g.shape[0], :g.shape[1]] = g[:n, :n]
    f = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(255 - im))))
    c = n // 2
    r = np.arange(n // 16, n // 2 - 1, 0.5)
    best, sc = 0.0, -1.0
    for a in np.arange(-max_deg, max_deg + 1e-9, step):
        t = np.deg2rad(a)
        # lines along (cos a, -sin a) in image coordinates (y down) put their energy along the perpendicular (sin a, cos a)
        xs, ys = c + r * np.sin(t), c + r * np.cos(t)
        s = cv2.remap(f, xs.astype(np.float32)[None], ys.astype(np.float32)[None], cv2.INTER_LINEAR).sum()
        if s > sc:
            best, sc = a, s
    return float(best)


def rotate(gray, deg):
    """Rotate the page by deg (positive = counter-clockwise in the image), white fill, canvas grown to keep the corners."""
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    W, H = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += W / 2 - w / 2; M[1, 2] += H / 2 - h / 2
    return cv2.warpAffine(gray, M, (W, H), borderValue=255, flags=cv2.INTER_LINEAR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max", type=float, default=15.0)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    files = sorted(Path("data/sroie/test/img").glob("*.jpg"))[:a.n]
    err, tm, base = {"profile": [], "fourier": []}, {"profile": 0.0, "fourier": 0.0}, []
    for f in files:
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        th = rng.uniform(-a.max, a.max)
        rg = rotate(g, th)
        for name, fn in (("profile", skew_profile), ("fourier", skew_fourier)):
            e0 = fn(g, a.max)  # the scan's own skew
            t0 = time.perf_counter(); e = fn(rg, a.max); tm[name] += time.perf_counter() - t0
            err[name].append(abs(e - e0 - th))
            if name == "profile": base.append(e0)
    print(f"{len(files)} SROIE test receipts, rotated by U(-{a.max}, {a.max}) deg; error = |est(rotated) - est(original) - angle|")
    print(f"own skew of the scans (profile): median |{np.median(np.abs(base)):.2f}| deg, max |{np.max(np.abs(base)):.2f}| deg")
    for k, v in err.items():
        v = np.array(v)
        print(f"{k:8s} mean {v.mean():.3f}  median {np.median(v):.3f}  p95 {np.percentile(v, 95):.3f}  max {v.max():.2f} deg  | >1 deg: {np.mean(v > 1):.0%}  | {tm[k] / len(files) * 1e3:.0f} ms/img (CPU)")
