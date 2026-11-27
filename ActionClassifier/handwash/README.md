# Hand washing steps

Which WHO hand-washing movement (routine / antiseptic washing with water) is being done, from 2 s of hand landmarks. Two TensorRT engines:

```
video -> hands engine (hands.py): MediaPipe Hands as one graph, uint8 frame + tracking state in, 2 hands x 21 landmarks out, 1.08 ms/frame
      -> 2 slots kept apart by wrist position (MediaPipe's left/right flips when seen from above)
      -> one window per frame: 32 frames (2 s at 16 fps)
      -> classifier engine (train.Window): missing hand filled with a copy of the other one (presence flag kept),
         normalized by mean wrist and hand size, + frame differences, TCN -> 7 classes: other + WHO movements 1-6
```

On METC validation (14 people never seen in training): 71.8% of frames right, 75.0% after averaging the probabilities over 1 s.

## Data

[METC subset](https://zenodo.org/records/5808789) of the Latvian hand-washing dataset (CC-BY-4.0): 212 videos (211 distinct), 72 people each washing at 3 sinks, camera above the sink, 640x480 at 16 fps, one label per frame. Downloaded to `data/handwash/metc/` (shared `~/.dev/data`). Validation = the 14 people with id % 5 == 0 (42 videos), never seen in training.

## Hand landmarks

Hand recall = hands found / 2 (both hands are always in the sink), 12 videos:

| Detector | Recall | Step 2 frames with 0/1/2 hands |
|---|---|---|
| MediaPipe VIDEO mode, conf 0.5 | 51-58% | 3/88/9 |
| MediaPipe VIDEO, conf 0.3 / 0.7 | 49% / 45% | |
| MediaPipe IMAGE mode, conf 0.3 | 49% | |

Tried and dropped: a YOLOv8n hand detector (Bingsu/adetailer, 52%, boxes only) and a YOLO26s hand pose trained on the Ultralytics hand-keypoints set (marceloeatworld/yolo26-training, 33% at conf 0.1: it misses clear hands and boxes forearms at the frame edge; that set has few views from above).

All 211 videos with MediaPipe VIDEO 0.5: 0/1/2 hands per step 1-6: 15/71/14, 9/82/9, 16/69/15, 20/58/22, 18/57/25, 9/47/44. Landmarks of the hands found look right, also on top of each other; the misses are mostly visible hands that are blurred by motion or enter from the frame edge.

## Hands engine

`hands.py` rebuilds MediaPipe's HandLandmarker (VIDEO mode, 2 hands) as one ONNX graph: the two networks of `hand_landmarker.task` (converted with tf2onnx, FP16 weights) and the steps between them in torch, exported and spliced in (as Motion does):

- letterbox to 192x192 with `GridSample`, palm detector, 2016 anchors decoded with Mul/Add;
- NMS without an NMS op, fixed shapes: ArgMax, palms overlapping it (IoU > 0.3) averaged by score (MediaPipe's weighted NMS) and removed, ArgMax again;
- per slot, the hand tracked from the previous frame (`state`) or a new palm (palms overlapping a tracked hand dropped), palm rect x2.6 shifted 0.5 up;
- rotated 224x224 crop with `GridSample` (edge pixels repeated: closer to MediaPipe than zeros, median landmark error 1.5-3.4 px vs 2-5 px), landmark model at batch 2;
- landmarks back to the frame, next rect from 12 landmarks (x2.0, shifted 0.1 up) -> `state_out`.

RTX 3050 Ti Laptop, TensorRT 11.5: 1.08 ms per frame for everything (palm detector alone 0.59 ms FP16 / 0.78 FP32, landmarks 0.33 / 0.53 ms per hand). Against MediaPipe on the CPU (~50 ms/frame): hand recall 48% vs 51% on 12 videos, step-2 frames with 0/1/2 hands 4/88/8 vs 3/89/8; the differences are mostly palms right at the 0.5 threshold, after which the two trackers follow different hands. The classifier trained on the engine's landmarks scores the same as on MediaPipe's (70.6 / 71.3 vs 70.7 / 70.2, two seeds). `check_hands.py` compares the same graph run eagerly (onnxruntime) with the CPU landmarks.

Two TensorRT details: it does not transpose uint8 (cast first), and ArgMax needs a 2-d input. The classifier engine stays FP32 (330k parameters, 0.27 ms for 64 windows); converting it to FP16 left mixed types in the normalization that TensorRT rejects.

## Results

Classifier, validation windows (every 4th frame), 30 epochs, label of the window's last frame, MediaPipe CPU landmarks:

| Head | Fill | Rotation (train) | Hand drop | Acc | Macro | Params |
|---|---|---|---|---|---|---|
| MLP | copy | none | 0.3 | 70.6 | 67.7 | 4.3M |
| MLP | copy | 360 + mirror | 0.3 | 68.4 | 64.7 | 4.3M |
| MLP | zero | 360 + mirror | 0.3 | 65.2 | 60.6 | 4.3M |
| TCN | copy | none | 0.3 | 72.8 | 70.0 | 330k |
| TCN | copy | +-30 + mirror | 0.3 | 71.1 | 68.8 | 330k |
| TCN | copy | 360 + mirror | 0.3 | 70.7 / 70.2 | 69.1 / 68.2 | 330k |
| TCN | hold | 360 + mirror | 0.3 | 70.8 | 68.4 | 330k |
| TCN | zero | 360 + mirror | 0.3 | 68.7 / 67.9 | 66.5 / 65.7 | 330k |
| TCN | copy | 360 + mirror | 0 | 68.7 | 66.5 | 330k |
| TCN | copy | 360 + mirror | 0.6 | 70.7 | 68.8 | 330k |

(two values: seeds 0 / 1.) Copying the other hand beats zeros by about 2 points; dropping a hand during training helps (0 -> 0.3), more does not. Full rotation costs about 2 points here because all METC cameras look roughly the same way; it is kept for cameras placed differently. Mirroring is safe: the labels do not say which hand is on top.

## Usage

Run from `ActionClassifier/handwash/`:

```bash
python extract.py --mode engine                       # landmarks of all METC videos with the hands engine -> cache/kpt-engine-0.5/ (--mode video|image: MediaPipe CPU)
python stats.py cache/kpt-engine-0.5                    # 0/1/2 hands per step
python train.py --kpt cache/kpt-engine-0.5 --head tcn --fill copy --name tcn   # -> weight/tcn/model.pt, model.onnx, metrics.json
python run.py video.mp4 [--out out/video.mp4] [--labels annotation.json]   # step per frame and a summary; engines built on first use
```

`run.py` reads the whole video through the hands engine (one frame per call, the tracking needs the previous frame; ~6 ms per frame with OpenCV decoding, which dominates), then classifies one window per frame in batches of 512 (all 1980 windows of a 2-minute video in ~0.2 s) and prints the steps with their times. Other frame rates are resampled to 16 fps inside the windows. On a validation video the steps 1-3 come out with the right times, while thumb rubbing (25 s) is mostly read as other / scattered steps.

## Limits

- Only routine / antiseptic hand washing (one model). Surgical scrub needs its own data: no public RGB set with step labels was found (MIVIA's surgical hand-washing set is depth only).
- The hands engine is built for one frame size and one camera per call; the classifier engine takes any batch of windows.
- Landmarks find only about half the hands (motion blur, hands entering from the frame edge, hands on top of each other).
