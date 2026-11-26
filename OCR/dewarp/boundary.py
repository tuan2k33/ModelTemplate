"""
Tim 4 goc to giay A4 trong anh roi warp ve khung A4 chuan — bang IMAGE PYRAMID
(coarse-to-fine), thuan xu ly anh (khong dung model).

Y tuong:
  - O muc THO nhat (anh thu rat nho): chu/nhieu tan bien, to giay la mot khoi lon
    => Otsu + fill + open (bo nhanh mong nhu day cap) + minAreaRect => quad THO
    (vi tri + huong nghieng).
  - O moi muc MIN dan: tinh chinh 4 canh bang cach, trong mot DAI HEP quanh canh
    hien tai, tim vi tri co:
        (do manh gradient theo phap tuyen)  +  beta * (do sang TRONG - NGOAI)
    Thanh phan gradient bat mep giay; thanh phan do sang "ben trong sang lien tuc"
    giup phan biet mep giay THAT voi vat sang mong (day cap, vien laptop) o gan.
    Vi la cong THEM, truong hop giay-tren-giay (do sang trong≈ngoai) van chay
    nho gradient.
  - Dai hep dan theo muc => loai duong chu noi bo (o xa dai) va vat nhieu xa.

Dung: python boundary.py input.jpg output.jpg [--debug]
"""

import argparse
import sys

import cv2
import numpy as np

OUT_W, OUT_H = 1240, 1754              # A4 @ 150dpi
LEVELS = [160, 320, 640, 1000]         # canh dai nhat moi muc (~1:12 ... 1:2)
BANDS = [0.16, 0.07, 0.04, 0.025]      # be rong dai tim (theo canh dai) tung muc:
                                       # rong dan o muc tho de canh "di xa" toi bien that,
                                       # hep dan de bam chinh xac + loai nhieu
BETA = 1.5                             # trong so uu tien "ben trong sang lien tuc"


def order_points(pts: np.ndarray) -> np.ndarray:
    """Sap 4 diem theo TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _largest_cc(binary: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return binary
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == idx, 255, 0).astype(np.uint8)


def _fill(binary: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(binary)
    if cnts:
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return out


def coarse_quad(sm: np.ndarray) -> np.ndarray:
    """
    Quad tho tu Otsu. Tach to giay khoi cac vung sang lien ke (ke giay, mep go...)
    bang opening-by-reconstruction: erode manh de dut cac cau noi -> giu khoi lon
    nhat (loi to giay) -> dilate tra lai kich thuoc.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(b) < 127:                          # dam bao giay = trang
        b = cv2.bitwise_not(b)
    h, w = gray.shape
    b = _fill(_largest_cc(b))
    r = max(int(max(h, w) * 0.08), 3)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    core = _largest_cc(cv2.erode(b, k, iterations=1))   # loi, da dut cau noi mong/vua
    b = cv2.dilate(core, k, iterations=1)               # tra lai kich thuoc
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    return order_points(cv2.boxPoints(cv2.minAreaRect(max(cnts, key=cv2.contourArea))))


def _line_from_points(pts: np.ndarray) -> np.ndarray:
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    a, b = -vy, vx                                 # phap tuyen
    return np.array([a, b, -(a * x0 + b * y0)], dtype=np.float64)


def _intersect(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    D = a1 * b2 - a2 * b1
    if abs(D) < 1e-6:
        return None
    return np.array([(-c1 * b2 + c2 * b1) / D, (-a1 * c2 + a2 * c1) / D])


def _search_side(p, q, gx, gy, gray, band, center, n_samples):
    """
    Do vuong goc canh (p->q) trong dai hep. Tra ve (ts, out, nrm) voi out canh
    theo ts: moi hang [x, y, offset, grad] (offset = do lech doc phap tuyen so voi
    day chord), hang NaN neu khong tim thay. Diem chon toi da:
    gradient_theo_phap_tuyen + BETA*(sang_trong - sang_ngoai).

    Vector hoa toan bo bang numpy broadcasting (N mau x O offset x K diem lay mau)
    thay vi loop Python tung pixel — ban goc goi ~4.4 trieu lan ham lay-1-pixel cho
    1 anh (~15s/anh). Ban nay dua het ve fancy-indexing numpy (~vai chuc ms/anh);
    inside/outside duoc gop chung 1 mang (2K diem, dau +k=trong/-k=ngoai) va rut
    gon inside.mean()-outside.mean() thanh 1 phep nhan-ma-tran voi vector trong
    so [+1/K]*K + [-1/K]*K, de chi con 1 lan round/clip/gather/reduce thay vi 4.
    """
    h, w = gray.shape
    d = q - p
    d = d / (np.hypot(*d) + 1e-9)
    nrm = np.array([-d[1], d[0]])
    if np.dot(center - (p + q) / 2, nrm) < 0:        # +nrm huong VAO trong
        nrm = -nrm
    win = max(int(band * 0.8), 6)
    ts = np.linspace(0.03, 0.97, n_samples)

    s = p[None, :] + ts[:, None] * (q - p)[None, :]                          # (N,2)
    offs = np.arange(-band, band + 1, dtype=np.float64)                     # (O,)
    base = s[:, None, :] + offs[None, :, None] * nrm[None, None, :]          # (N,O,2) = (px,py)

    xr = np.round(base[..., 0]).astype(np.int64)
    yr = np.round(base[..., 1]).astype(np.int64)
    in_bounds = (xr >= 0) & (xr < w) & (yr >= 0) & (yr < h)                  # (N,O)
    xc, yc = np.clip(xr, 0, w - 1), np.clip(yr, 0, h - 1)
    g = np.abs(gx[yc, xc] * nrm[0] + gy[yc, xc] * nrm[1])                    # (N,O)

    k = np.arange(3, win, dtype=np.float64)                                  # (K,)
    all_k = np.concatenate([k, -k])                                          # (2K,) +k=trong, -k=ngoai
    weight = np.concatenate([np.ones_like(k), -np.ones_like(k)]) / len(k)    # (2K,)

    samp = base[:, :, None, :] + all_k[None, None, :, None] * nrm            # (N,O,2K,2)
    ix = np.clip(np.round(samp[..., 0]).astype(np.int64), 0, w - 1)
    iy = np.clip(np.round(samp[..., 1]).astype(np.int64), 0, h - 1)
    diff = gray[iy, ix] @ weight                                            # (N,O) = mean(trong)-mean(ngoai)

    val = np.where(in_bounds, g + BETA * diff, -np.inf)                     # (N,O)
    best_idx = np.argmax(val, axis=1)                                       # (N,) — argmax = first max, giong ban goc
    rows = np.arange(n_samples)
    best_val, best_g = val[rows, best_idx], g[rows, best_idx]
    best_off = offs[best_idx]
    best_pt = base[rows, best_idx]                                          # (N,2)

    out = np.full((n_samples, 4), np.nan)
    valid = np.isfinite(best_val) & (best_g > 0)
    out[valid] = np.column_stack([best_pt[valid], best_off[valid], best_g[valid]])
    return ts, out, nrm


def refine_quad(quad, gx, gy, gray, band, n_samples=60):
    """
    Tinh chinh quad: dung DUONG THANG lam GIAN GIAO — fit line vao diem bien moi
    canh roi giao 2 canh ke -> 4 goc. (Giai doan nay chi de neo vi tri/huong;
    do cong cua mep giay lay o buoc boundary_contour.)
    """
    center = quad.mean(axis=0)
    sides = [(quad[0], quad[1]), (quad[1], quad[2]), (quad[2], quad[3]), (quad[3], quad[0])]
    lines = []
    for p, q in sides:
        _, out, _ = _search_side(p, q, gx, gy, gray, band, center, n_samples)
        valid = out[~np.isnan(out[:, 0])]
        if len(valid) < 5:
            lines.append(None)
            continue
        keep = valid[valid[:, 3] >= np.percentile(valid[:, 3], 40)][:, :2]
        lines.append(_line_from_points(keep))

    L_top, L_right, L_bottom, L_left = lines
    old = quad.copy()

    def corner(a, b, fallback):
        p = _intersect(a, b) if (a is not None and b is not None) else None
        return p if p is not None else fallback

    return np.array([
        corner(L_left, L_top, old[0]),
        corner(L_top, L_right, old[1]),
        corner(L_right, L_bottom, old[2]),
        corner(L_bottom, L_left, old[3]),
    ], dtype=np.float32)


def detect_quad(image: np.ndarray, debug_dump=None):
    """Tra ve quad (toa do full-res) tu image pyramid coarse-to-fine."""
    base = max(image.shape[:2])
    levels = [d for d in LEVELS if d <= base] or [base]

    s0 = levels[0] / base
    quad = coarse_quad(cv2.resize(image, None, fx=s0, fy=s0)) / s0
    if debug_dump:
        debug_dump(quad, "L0")

    for li, dim in enumerate(levels):
        s = dim / base
        sm = cv2.resize(image, None, fx=s, fy=s)
        g = cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        band = max(int(dim * BANDS[min(li, len(BANDS) - 1)]), 4)
        quad = refine_quad(quad * s, gx, gy, g, band) / s
        if debug_dump:
            debug_dump(quad, f"L{li + 1}_dim{dim}")

    return order_points(quad)


def _robust_offsets(ts, offs, side_len, deg=2, max_bow_ratio=0.03):
    """
    Lam sach chuoi do-lech (offset) cua 1 canh so voi gian giao (duong thang):
    mep giay chi THANG hoac CONG DEU 1 nhip -> fit da thuc bac thap (parabol)
    CO CHONG NHIEU (reweight loai outlier) thay vi lam muot diem-theo-diem.
    Giu do vong that, dap sach wobble/rang cua do tay/chu/nen gay ra.
    """
    valid = ~np.isnan(offs)
    if valid.sum() < deg + 3:
        return np.zeros(len(offs))
    tv, ov = ts[valid], offs[valid]
    coef = np.polyfit(tv, ov, deg)
    for _ in range(3):                                    # IRLS tho: loai residual lon roi fit lai
        resid = ov - np.polyval(coef, tv)
        m = np.median(resid)
        mad = np.median(np.abs(resid - m)) + 1e-6
        keep = np.abs(resid - m) < 2.5 * mad
        if keep.sum() < deg + 3:
            break
        coef = np.polyfit(tv[keep], ov[keep], deg)
    out = np.polyval(coef, ts)
    max_bow = max_bow_ratio * side_len
    return np.clip(out, np.median(ov) - max_bow, np.median(ov) + max_bow)


def detect_boundary(image: np.ndarray, dense=100):
    """
    Tra ve BIEN to giay dang duong CONG (contour Nx2, full-res) + 4 goc.
    Duong thang chi la gian giao: moi canh bam diem bien thuc te, do lech so voi
    gian giao duoc fit bang parabol robust (_robust_offsets) -> mep cong muot,
    khong nhay sang mep sai, khong wobble theo nhieu.
    """
    quad = detect_quad(image)
    base = max(image.shape[:2])
    dim = min(1000, base)
    s = dim / base
    sm = cv2.resize(image, None, fx=s, fy=s)
    g = cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    q = quad * s
    center = q.mean(axis=0)
    band = max(int(dim * 0.05), 6)

    contour = []
    for i in range(4):
        c0, c1 = q[i], q[(i + 1) % 4]
        contour.append(c0)                                     # neo goc
        ts, out, nrm = _search_side(c0, c1, gx, gy, g, band, center, dense)
        off = _robust_offsets(ts, out[:, 2], np.hypot(*(c1 - c0)))
        chord = c0[None, :] + ts[:, None] * (c1 - c0)[None, :]
        curve = chord + off[:, None] * nrm[None, :]            # diem bien = chord + offset*nrm
        contour.extend(curve)
    return np.array(contour, dtype=np.float32) / s, quad


def boundary_mask(image: np.ndarray):
    """Mask nhi phan bam duong cong cua to giay."""
    contour, _ = detect_boundary(image)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(mask, [contour.astype(np.int32)], 255)
    return mask, contour


def warp_to_a4(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    rect = order_points(quad)
    dst = np.array([[0, 0], [OUT_W - 1, 0], [OUT_W - 1, OUT_H - 1], [0, OUT_H - 1]],
                   dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (OUT_W, OUT_H))


def main(in_path: str, out_path: str, warp: bool, debug: bool):
    image = cv2.imread(in_path)
    if image is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {in_path}")
    base_out = out_path.rsplit(".", 1)[0]

    if warp:
        quad = detect_quad(image)
        cv2.imwrite(out_path, warp_to_a4(image, quad))
        print(f"Da luu (warp A4): {out_path}")
        return

    # Mac dinh: xuat MASK bam duong cong + overlay (khong warp)
    contour, quad = detect_boundary(image)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(mask, [contour.astype(np.int32)], 255)
    cv2.imwrite(out_path, mask)

    overlay = image.copy()
    fill = image.copy()
    cv2.fillPoly(fill, [contour.astype(np.int32)], (0, 0, 255))
    overlay = cv2.addWeighted(fill, 0.25, overlay, 0.75, 0)
    cv2.polylines(overlay, [contour.astype(np.int32)], True, (0, 0, 255), 4)
    cv2.imwrite(f"{base_out}_overlay.jpg", overlay)
    print(f"Da luu: mask -> {out_path} | overlay -> {base_out}_overlay.jpg")

    if debug:
        cv2.imwrite(f"{base_out}_quad.jpg",
                    cv2.polylines(image.copy(), [quad.astype(np.int32)], True, (0, 200, 255), 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tim bien to giay A4 (image pyramid). Mac dinh xuat MASK; --warp de nan A4.")
    parser.add_argument("input")
    parser.add_argument("output", help="Duong dan mask (mac dinh) hoac anh warp (voi --warp)")
    parser.add_argument("--warp", action="store_true", help="Nan ve khung A4 thay vi xuat mask")
    parser.add_argument("--debug", action="store_true", help="Xuat them quad thang de so sanh")
    args = parser.parse_args()
    main(args.input, args.output, args.warp, args.debug)
