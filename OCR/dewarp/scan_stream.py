"""
Live scan A4 tu webcam/video stream: cho toi khi quad on dinh + du net thi tu dong
chup, roi moi chay pipeline chinh xac (cham hon, nhung chi chay 1 lan/lan chup).

Kien truc (khong dung model AI nao, tan dung lai boundary.py):
  - Moi frame: detect_quad() day du (pyramid coarse-to-fine, da vector hoa bang
    numpy, ~44-47ms/anh o do phan giai anh test ~1080-1536px canh dai — xem
    README.md) de ve overlay + kiem tra do on dinh (goc it doi cho qua N frame
    lien tiep). Ban dau dinh dung rieng coarse_quad() (~1ms, chi buoc tho nhat
    ben trong detect_quad) cho live loop de re hon, nhung test thuc te thay
    coarse_quad khong on dinh khi anh sang nhieu/thay doi — nen dung thang
    detect_quad moi frame, chap nhan tran ~21-22fps de doi lay do on dinh.
  - Khi on dinh + du net (Laplacian variance): chay detect_boundary() (cung da
    vector hoa, ~150-200ms/anh) de lay bien cong chinh xac + quad calib, roi
    luu mask/frame/(anh warp A4).

Dung:
  python scan_stream.py                        # webcam mac dinh (device 0)
  python scan_stream.py --source clip.mp4       # test bang video file
  python scan_stream.py --warp                  # xuat them anh da nan A4
  python scan_stream.py --headless --max-captures 1   # chay khong can man hinh (vd qua SSH/WSL)
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from boundary import detect_boundary, detect_quad, warp_to_a4

STABLE_FRAMES = 15          # so frame lien tiep on dinh truoc khi chup
STABILITY_RATIO = 0.006     # dich goc toi da cho phep giua cac frame, ti le theo duong cheo frame
MIN_AREA_RATIO = 0.15       # quad phai chiem toi thieu % dien tich frame (loai canh khong co giay)
MAX_AREA_RATIO = 0.97       # loai truong hop detect_quad tra ve gan het ca frame (khong tim thay giay)
SHARPNESS_MIN = 40.0        # nguong Laplacian variance, duoi nguong coi la mo
COOLDOWN_SEC = 2.0          # sau khi chup, cho vai giay truoc khi tim quad on dinh tiep theo


def frame_quad(frame: np.ndarray) -> np.ndarray | None:
    """Quad chinh xac (detect_quad day du) cho MOI frame, dung cho ca overlay lan calib cuoi."""
    h, w = frame.shape[:2]
    quad = detect_quad(frame)
    area = cv2.contourArea(quad.astype(np.float32))
    frame_area = h * w
    if not (MIN_AREA_RATIO * frame_area < area < MAX_AREA_RATIO * frame_area):
        return None
    return quad


def sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_stable(history: deque, frame_diag: float) -> bool:
    if len(history) < history.maxlen:
        return False
    ref = history[-1]
    max_shift = max(np.linalg.norm(q - ref, axis=1).max() for q in history)
    return max_shift < STABILITY_RATIO * frame_diag


def run(source: str, out_dir: Path, do_warp: bool, max_captures: int, headless: bool):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc nguon video: {source}")

    out_dir.mkdir(parents=True, exist_ok=True)
    history = deque(maxlen=STABLE_FRAMES)
    n_captured = 0
    cooldown_until = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Het stream / khong doc duoc frame.")
            break

        now = time.time()
        h, w = frame.shape[:2]
        diag = float(np.hypot(h, w))
        quad = frame_quad(frame)
        sharp = sharpness(frame)

        status, color = "searching", (0, 200, 255)          # vang: dang tim
        if quad is not None and now >= cooldown_until:
            history.append(quad)
            if sharp < SHARPNESS_MIN:
                status = "blur"
            elif is_stable(history, diag):
                status, color = "capturing", (0, 255, 0)    # xanh: du dieu kien chup
            else:
                status, color = "holding", (0, 165, 255)    # cam: on dinh dan, chua du
        else:
            history.clear()

        if not headless:
            disp = frame.copy()
            if quad is not None:
                cv2.polylines(disp, [quad.astype(np.int32)], True, color, 3)
            cv2.putText(disp, f"{status}  sharp={sharp:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("scan_stream", disp)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

        if status == "capturing":
            print(f"[{n_captured}] On dinh + du net (sharp={sharp:.0f}) -> chay pipeline chinh xac...")
            contour, precise_quad = detect_boundary(frame)
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [contour.astype(np.int32)], 255)

            stem = out_dir / f"capture_{n_captured:03d}"
            cv2.imwrite(f"{stem}_mask.png", mask)
            cv2.imwrite(f"{stem}_frame.jpg", frame)
            if do_warp:
                cv2.imwrite(f"{stem}_a4.jpg", warp_to_a4(frame, precise_quad))
            print(f"    da luu -> {stem}_mask.png (+_frame.jpg{' +_a4.jpg' if do_warp else ''})")

            n_captured += 1
            history.clear()
            cooldown_until = now + COOLDOWN_SEC
            if max_captures and n_captured >= max_captures:
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live scan A4 tu webcam/video: doi quad on dinh + du net roi tu dong chup.")
    parser.add_argument("--source", default="0", help="Device index webcam (vd 0) hoac duong dan video file")
    parser.add_argument("--out-dir", default="stream_out", help="Thu muc luu ket qua")
    parser.add_argument("--warp", action="store_true", help="Xuat them anh da nan ve khung A4")
    parser.add_argument("--max-captures", type=int, default=0, help="Dung sau N lan chup (0 = khong gioi han)")
    parser.add_argument("--headless", action="store_true",
                         help="Khong mo cua so hien thi (vd chay qua SSH/WSL khong co X server)")
    args = parser.parse_args()
    run(args.source, Path(args.out_dir), args.warp, args.max_captures, args.headless)
