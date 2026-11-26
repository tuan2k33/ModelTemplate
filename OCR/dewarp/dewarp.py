"""
Nan to giay cong ve khung A4 bang 4 mep CONG cua detect_boundary (thuan CPU, khong model).

Y tuong: homography tu 4 goc dua trang ve khung chu nhat (dung phoi canh). Mep cong sau
homography lech khoi canh thang mot doan nho; do lech do duoc noi suy vao ben trong bang
Coons patch (moi diem ben trong = pha tron tuyen tinh cua 4 mep), roi cv2.remap.
Trang phang -> do lech = 0 -> dung bang warpPerspective. Nep gap o giua khong sua duoc
(mep khong thay).

Dung: python dewarp.py input.jpg output.jpg
"""

import argparse

import cv2
import numpy as np

from boundary import OUT_H, OUT_W, detect_boundary


def _offsets(pts, along, across, n, edge):
    """Do lech `across` (so voi canh thang o `edge`) theo vi tri `along` 0..n-1, bang 0 o 2 dau."""
    o = np.argsort(pts[:, along])
    a, d = pts[o, along], pts[o, across] - edge
    d = np.interp(np.arange(n), a, d)
    return d - np.linspace(d[0], d[-1], n)              # ep 0 o 2 goc (giu do vong, bo lech goc)


def coons_map(contour, quad):
    """(map_x, map_y) float32 OUT_HxOUT_W: toa do anh nguon cho moi pixel A4."""
    W, H = OUT_W, OUT_H
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    Hm = cv2.getPerspectiveTransform(np.float32(quad), dst)
    r = cv2.perspectiveTransform(contour.reshape(-1, 1, 2).astype(np.float32), Hm).reshape(4, -1, 2)
    corner = dst[:, None]                               # moi canh: goc dau + diem cong
    sides = [np.concatenate([corner[i], r[i], corner[(i + 1) % 4]]) for i in range(4)]
    top = _offsets(sides[0], 0, 1, W, 0)                # dy(x)
    bot = _offsets(sides[2], 0, 1, W, H - 1)
    left = _offsets(sides[3], 1, 0, H, 0)               # dx(y)
    right = _offsets(sides[1], 1, 0, H, W - 1)

    u = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    v = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    x = np.arange(W, dtype=np.float32)[None, :] + (1 - u) * left[:, None] + u * right[:, None]
    y = np.arange(H, dtype=np.float32)[:, None] + (1 - v) * top[None, :] + v * bot[None, :]
    Hi = np.linalg.inv(Hm).astype(np.float32)           # khung A4 (da nan) -> anh goc
    den = Hi[2, 0] * x + Hi[2, 1] * y + Hi[2, 2]
    return ((Hi[0, 0] * x + Hi[0, 1] * y + Hi[0, 2]) / den).astype(np.float32), \
           ((Hi[1, 0] * x + Hi[1, 1] * y + Hi[1, 2]) / den).astype(np.float32)


def kpt_map(kpts):
    """Nhu coons_map nhung tu keypoint CO TUONG UNG: kpts (4n,2) chay TL -> TR -> BR -> BL -> (ve TL), moi canh n diem
    cach deu tren trang PHANG (diem dau = goc). Do lech 2 chieu (doc + ngang canh) duoc noi suy, nen ca cho mep bi
    gian/co (phoi canh, cong) cung duoc sua. Canh 0 (diem 0 -> n) la canh ngan cua trang (khung A4 doc)."""
    W, H = OUT_W, OUT_H
    n = len(kpts) // 4
    kpts = np.float32(kpts)
    c = np.ascontiguousarray(kpts[::n])
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    tgt = np.concatenate([dst[i] + t * (dst[(i + 1) % 4] - dst[i]) for i in range(4)])
    Hm = cv2.getPerspectiveTransform(c, dst)
    d = cv2.perspectiveTransform(kpts.reshape(-1, 1, 2), Hm).reshape(-1, 2) - tgt
    d = np.concatenate([d, d[:1]]).reshape(-1, 2)
    tt = np.linspace(0, 1, n + 1)

    def side(i, m, rev):                                # do lech (m,2) doc canh i, tham so 0..1 theo huong trai->phai / tren->duoi
        s = d[i * n:i * n + n + 1]
        s = s - np.linspace(s[0], s[-1], n + 1)         # ep 0 o goc (homography da khop goc)
        x = np.linspace(0, 1, m)
        out = np.stack([np.interp(x, tt, s[:, j]) for j in (0, 1)], 1)
        return out[::-1] if rev else out

    top, right, bot, left = side(0, W, False), side(1, H, False), side(2, W, True), side(3, H, True)
    u = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
    v = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    D = (1 - v) * top[None] + v * bot[None] + (1 - u) * left[:, None] + u * right[:, None]
    x = np.arange(W, dtype=np.float32)[None, :] + D[..., 0]
    y = np.arange(H, dtype=np.float32)[:, None] + D[..., 1]
    Hi = np.linalg.inv(Hm).astype(np.float32)
    den = Hi[2, 0] * x + Hi[2, 1] * y + Hi[2, 2]
    return ((Hi[0, 0] * x + Hi[0, 1] * y + Hi[0, 2]) / den).astype(np.float32), \
           ((Hi[1, 0] * x + Hi[1, 1] * y + Hi[1, 2]) / den).astype(np.float32)


def dewarp_kpts(image, kpts):
    mx, my = kpt_map(kpts)
    return cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def grid_map(grid):
    """Luoi diem (rows, cols, xy) cach deu tren trang phang (hang 0 = canh ngan, A4 doc) -> map A4. Nhu kpt_map:
    homography tu 4 goc, roi noi suy bicubic DO LECH cua cac diem luoi trong khung da nan (align corners)."""
    W, H = OUT_W, OUT_H
    r, c = grid.shape[:2]
    grid = np.float32(grid)
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    Hm = cv2.getPerspectiveTransform(np.ascontiguousarray([grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0]]), dst)
    tgt = np.stack(np.meshgrid(np.linspace(0, W - 1, c), np.linspace(0, H - 1, r)), -1)
    d = (cv2.perspectiveTransform(grid.reshape(-1, 1, 2), Hm).reshape(r, c, 2) - tgt).astype(np.float32)
    gx = np.broadcast_to(np.linspace(0, c - 1, W, dtype=np.float32)[None, :], (H, W))
    gy = np.broadcast_to(np.linspace(0, r - 1, H, dtype=np.float32)[:, None], (H, W))
    D = cv2.remap(d, np.ascontiguousarray(gx), np.ascontiguousarray(gy), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    x = np.arange(W, dtype=np.float32)[None, :] + D[..., 0]
    y = np.arange(H, dtype=np.float32)[:, None] + D[..., 1]
    Hi = np.linalg.inv(Hm).astype(np.float32)
    den = Hi[2, 0] * x + Hi[2, 1] * y + Hi[2, 2]
    return ((Hi[0, 0] * x + Hi[0, 1] * y + Hi[0, 2]) / den).astype(np.float32), \
           ((Hi[1, 0] * x + Hi[1, 1] * y + Hi[1, 2]) / den).astype(np.float32)


def dewarp_grid(image, grid):
    mx, my = grid_map(grid)
    return cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def dewarp_to_a4(image, contour=None, quad=None):
    if contour is None:
        contour, quad = detect_boundary(image)
    mx, my = coons_map(contour, quad)
    return cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Nan to giay cong ve A4 (Coons patch tu 4 mep cong).")
    p.add_argument("input")
    p.add_argument("output")
    a = p.parse_args()
    img = cv2.imread(a.input)
    if img is None:
        raise FileNotFoundError(a.input)
    cv2.imwrite(a.output, dewarp_to_a4(img))
