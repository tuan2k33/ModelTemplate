# Dewarp

Phone photo of an A4 page (all four corners usually visible, white or cream paper, may be curved) to a flat 1240x1754 rectangle, on the **CPU**. Folded or crumpled paper is out of scope. The output only has to be a rectangle: an upside-down page is fixed later (`../deskew.py`). Moved here from the former DocScan repo.

```
photo -> paper boundary (4 curved edges)  -> homography from the 4 corners + Coons patch on the edge offsets -> cv2.remap -> A4
         boundary.py (image processing)      dewarp.py
```

Later stages (not built): illumination normalization (divide by a large-blur background estimate, keep the paper tone, optional white patch, light unsharp mask) and OCR that keeps the layout (paragraphs, tables, indents), which is why phone scanner apps were not enough.

## Usage

Run from `OCR/dewarp/`. Needs `opencv-python`, `numpy`; the benchmark also `h5py`, the keypoint model `ultralytics`.

```bash
python boundary.py photo.jpg mask.png [--warp] [--debug]   # paper mask + overlay; --warp: straight A4 warp from the 4 corners
python dewarp.py photo.jpg flat.jpg                          # curved edges -> A4
python scan_stream.py [--source 0|clip.mp4] [--warp]         # live: capture when the quad is stable and sharp
python bench_uvdoc.py                                        # UVDoc benchmark, see Results
python paper_kpt.py --layout edge|7x5 [--epochs 30 | --resume]   # optional keypoint model, paused
```

## Boundary (`boundary.py`)

Image pyramid, coarse to fine, no model:

- `coarse_quad`: at ~160 px the text disappears and the page is one blob. Otsu, fill, then opening by reconstruction (erode ~8% of the side to cut bridges to bright neighbours such as a shelf edge, keep the largest core, dilate back), `minAreaRect`.
- `refine_quad` over `LEVELS=[160,320,640,1000]` with bands `BANDS=[0.16,0.07,0.04,0.025]` (wide at the coarse level so an edge can travel to the real border, narrow later): along each side, search across the edge for the maximum of `normal gradient + BETA * (brightness inside - outside)`. The gradient finds the edge; the inside-brighter term rejects thin bright objects (cables, a laptop rim); being added rather than required, paper on paper (inside ~ outside) still works. `cv2.fitLine(DIST_HUBER)` per side, intersections = 4 corners.
- `detect_boundary`: the real curved edge. Per side, the offset of the found edge points from the straight side is fitted with a robust parabola (`_robust_offsets`: polyfit degree 2, reweighted with outliers > 2.5 MAD dropped, bow clamped to 3% of the side). A paper edge is straight or bent once, so a parabola keeps the real bow and drops wobble from hands, text and background.

Why not the usual tools: GrabCut/Otsu merge the page with paper of the same colour, Canny/Hough at full resolution drown in text, straight lines are wrong for curved paper. Weak spot: the top edge sometimes takes a few px of background.

Speed (1536x2048 / 1080x1920 photos): `_search_side` is vectorized with numpy broadcasting (samples x offsets x probe points, one gather), bit-for-bit equal to the original per-pixel loop: `detect_quad` 5-6 s -> 43-47 ms, `detect_boundary` 14-16 s -> 140-165 ms. Measured dead ends: one `np.clip` with array bounds was slower than two scalar clips (56 vs 51 ms); `np.minimum/np.maximum` instead of `np.clip` was 45% faster in a microbenchmark but 3% slower inside the pipeline (interleaved A/B), reverted. Next options if needed: `cv2.remap` sampling or Numba. `scan_stream.py` runs the full `detect_quad` per frame (~21 fps; `coarse_quad` alone was unstable under changing light) and `detect_boundary` once per capture.

## Dewarp (`dewarp.py`)

- `dewarp_to_a4(img)`: corners give a homography to the A4 frame (keeps perspective); the curved edges, mapped into that frame, are a small offset from the straight sides, interpolated into the page with a Coons patch, then `cv2.remap`. A flat page gives exactly `warpPerspective`. ~80 ms CPU at 1240x1754.
- `dewarp_kpts(img, kpts)`: same from points on the edges with known positions on the flat page (point k of a side at k/n of it), so stretch along an edge is corrected too.
- `dewarp_grid(img, grid)`: rows x cols points spread over the page (row 0 = a short side), homography + bicubic interpolation of the residuals.

## Results

[UVDoc benchmark](https://github.com/tanguymagne/UVDoc-Dataset) (`data/uvdoc/UVDoc_benchmark`: 50 photos of curved / folded / crumpled pages with the flat original and the true page grid `grid2d`, mask `seg`, `uvmap`). Error = median displacement of SIFT matches between the dewarped page and the flat original, px on 1241x1754 (floor about 1-2 px with the full true grid; failed matches make single images noisy, so only medians are used).

| Dewarp, from the TRUE page edges / points | Error |
|---|---|
| homography from 4 corners | 33.9 px |
| Coons on the edge shape (what a mask gives) | 25.5 px |
| points spaced evenly along the edge in the image (from a mask) | 38-49 px |
| 16 points on the edges with known flat positions (5 per side; 9 is not better) | 13.6 px |
| grid of 5x4 / 7x5 / 9x7 points over the page | 7.5 / 5.2 / 2.5 px |

With `detect_boundary` the full pipeline is far worse (homography 142 px, Coons 187 px): mask IoU against `seg` median 0.91, 22/50 above 0.95, 14/50 below 0.8, nearly all failures in `coarse_quad` (locks on the text block or merges a bright background). But 49/50 benchmark pages touch the image border (<1%), unlike real photos with background around the page, so the benchmark is unfair to it.

Keypoint model (`paper_kpt.py`, YOLO26n-pose fine-tuned on UVDoc_final: 20k synthetic 488x712 photos, `data/uvdoc/UVDoc_final`, only `img`, `grid2d`, `metadata_*` extracted): 16 points, point 0 = the corner nearest the image top-left, side 0 = a short side, clockwise (labels follow geometry, not text direction; UVDoc mirrors 3/4 of its geometries, handled in `canon`). After 2 epochs: keypoint error 27 px on the 1041x1330 benchmark photos, dewarp 47 px. Paused at epoch 4/30 (`weight/kpt-edge/weights/last.pt`, ~7 min/epoch on an RTX 3050 Ti; `--resume` continues). It only saw pages filling the frame.

## Plan

1. Run `detect_boundary` + `dewarp_to_a4` on real photos (`data/photos/`, not in git); judge by eye and by how straight the text lines come out (no flat original).
2. If good enough, keep the image-processing boundary.
3. If the coarse stage fails: replace `coarse_quad` with a coarse page mask from DINOv3 (ViT-S/16 at ~320 px -> 20x20 patches, linear probe trained in minutes on UVDoc `seg`), keep `refine_quad` / `detect_boundary` for the precise edge and Coons for the warp. CPU time not measured (~50-100 ms expected).
4. Only then the keypoint model again; the 7x5 grid layout is ready in `paper_kpt.py`. Untried: Ultralytics `yolo26n-depth` to recover the page surface.

Limits: the output is always A4 (A3/A5 have the same ratio; Letter or receipts get stretched). WSL has ~8 GB RAM: a second torch process during training got the trainer killed by earlyoom. Ultralytics on Python 3.14 needs `multiprocessing.set_start_method("fork")` for dataloader workers (`paper_kpt.py` sets it).
