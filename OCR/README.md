# OCR

Work in progress. So far: text-line detection on scanned receipts with YOLO26, as one **FP16 TensorRT engine** built by `../Detector` (letterbox inside the engine), a page deskew benchmark, and `dewarp/` (phone photo of a curved page to a flat A4, CPU, see [dewarp/README.md](dewarp/README.md)). No text recognition yet.

## Text-line detector

`det.py` converts SROIE (626 train / 347 test receipts, one box per text line; `data/sroie`) to a one-class YOLO set in `cache/yolo/` and fine-tunes `yolo26n` at 960 px. Stopped after 21 of 50 epochs (already flat since epoch 18): on the SROIE test split precision 0.963, recall 0.955, mAP50 0.976, mAP50-95 0.68.

```bash
cd OCR && python det.py --epochs 50                       # -> weight/det/weights/best.pt
cd ../Detector && python train.py --epochs 0 --model ../OCR/weight/det/weights/best.pt --name text   # FP16 engine -> Detector/weight/text/
python predict.py weight/text/model.engine receipt.jpg --out result.jpg
python quantize.py --model ../OCR/weight/det/weights/best.pt --images ../OCR/cache/yolo/images/train --name text-int8   # optional INT8, see Detector/README.md
```

GPU time per image (`trtexec`, RTX 3050 Ti Laptop): FP16 1.4 ms, INT8 2.1 ms, so FP16 is the default here too.

## Deskew

`deskew.py` estimates the skew of a page (text lines against the horizontal, within +-15 deg) with two classical methods and benchmarks them on 100 SROIE test receipts rotated by known random angles:

| Method | Median error | p95 | Max | > 1 deg | CPU |
|---|---|---|---|---|---|
| Fourier (strongest ray of the log spectrum) | 0.03 deg | 0.16 deg | 3.6 deg | 1% | 64 ms |
| Projection profile (spikiest row histogram of the ink) | 0.08 deg | 0.34 deg | 31 deg | 4% | 16 ms |

The SROIE scans themselves are nearly straight (median skew 0.1 deg, max 3 deg). Neither method tells a page upside down or turned by 90 deg, and neither fixes curved or perspective-distorted photos.

Limits: the engine letterboxes to 640 (`SIDE` in `Detector/model.py`) while the model was trained at 960, so small print on long receipts loses confidence. `det.py` uses `workers=0` (the Ultralytics dataloader workers crash under Python 3.14's forkserver).
