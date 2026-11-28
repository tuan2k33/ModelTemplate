# Motion

Background model for fixed cameras as **one TensorRT engine** with its memory passed in and out, no trained weights. Optionally fused with a YOLO26 engine from `../Detector` so one call gives detections, the motion mask and the next state for several cameras.

```
uint8 frame [B,H,W,3] RGB + state [B,8,180,320] float   (zeros at the start; one row per camera)
  -> background update and comparison, at 180x320
  -> mask [B,1,180,320] uint8 (1 = differs from the background), state_out (same layout as state)
```

Channels of the state: 0-2 background RGB, 3 its weight (updates it has been seen for, at most `cap`), 4-6 candidate RGB, 7 candidate count. A pixel that matches the background raises its weight; one that does not feeds the candidate, which replaces the background once its count passes that weight, so a newcomer has to outlast what it replaces and the first frame is the background. When most of the frame stops matching at once (lights, camera moved), the weights drop to `form` seconds so the background re-forms quickly. The state stays FP32: FP16 loses the small updates.

Rows never see each other; the caller hands each row the state of its own camera (`model.Engine` does).

## Fused with the detector

`build.py` puts the YOLO26 weights and the background model in one engine: `image` [B,H,W,3] and `state` in, `detections` [B,300,6], `state_out` and `mask` out. The centre of each box is looked up on the mask and its score gets `+boost` on the logit (default 0.5).

Why only +0.5: re-scoring boxes with the mask (box holds a blob centre, box centre on the mask, or share of the area on the mask; constants from likelihood ratios measured on another video) did not improve AP50 or recall at precision 0.9 on clear traffic and pedestrian videos, against YOLO26x at 1280 as reference (`collect.py`, `fuse.py`): 93-95% of the low-score false boxes on the mask sit on real moving objects (wrong class, a second or loose box), so "something moves here" does not separate them. On a degraded camera (1/4 size, blur, noise) it does help a small model: YOLO26n on traffic AP50 0.148 -> 0.167, recall at precision 0.9 0.007 -> 0.054 with +1. +0.5 is harmless on clear video and helps on poor video.

## Usage

Run from `Motion/`:

```bash
python run.py in.mp4 out.mp4                          # left: moving regions boxed, right: the learned background (H264)
python motion.py in.mp4 --mask out/motion.mp4 --background out/bg.png   # motion video (white = moving, H264, video size) + the background after the last frame
python run.py in.mp4 out.mp4 --detector ../Detector/weight/coco/model.onnx   # fused engine: detector boxes, yellow where the mask raised the score
python build.py cache/yolo26n.pt cache/fused.engine --batch 4 --top 720 1280  # fused multi-camera engine; run with model.Engine(engine, cameras=N)
```

Options: `--every` (update the background every Nth frame, default 5; the mask is computed on every frame), `--form` (seconds to re-form after a global change, 5), `--cap` (most weight a background can gather in seconds, so the longest a still object stays foreground, 600), `--thr` (a pixel matches when no channel differs by more than this, 0..255, default 25), `--min-area`, `--boost`, `--conf`.

Code: `model.py` (the background graph, the fused graph, engine runner keeping one state per camera), `run.py`, `motion.py`, `build.py`, and the experiment behind the boost: `collect.py` (Motion mask plus boxes of a reference and of the detectors under test, optionally on a degraded frame), `fuse.py` (re-scoring variants).

## Limits

- Fixed cameras only: a moving or shaking camera makes everything foreground.
- `--batch` x `--top` is bounded by GPU memory: 4 x 1280x720 builds on a 4 GB GPU, 4 x 1920x1080 does not.
- The engine is specific to the GPU and TensorRT version that built it; raise `VER` in `model.py` when the graph changes (part of the engine cache name).
