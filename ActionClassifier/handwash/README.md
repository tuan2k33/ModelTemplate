# Hand washing steps

Which WHO hand-washing movement (soap and water) is being done, from 2 s of hand landmarks. Work in progress: landmarks from MediaPipe Hands on the CPU, classifier trained, no engine yet.

```
video 16 fps -> MediaPipe HandLandmarker (VIDEO mode): up to 2 hands x 21 landmarks per frame
             -> 2 slots kept apart by wrist position (MediaPipe's left/right flips when seen from above)
             -> missing hand filled with a copy of the other one, presence flag 0
             -> 32 frames (2 s), normalized by mean wrist and hand size, + frame differences
             -> TCN -> 7 classes: other + WHO movements 1-6
```

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

On the GPU: the two networks of `hand_landmarker.task` convert to ONNX (tf2onnx) and build in TensorRT 11; FP16 (onnxconverter_common, float32 I/O) 0.59 ms palm detector + 0.33 ms per hand landmarks on an RTX 3050 Ti Laptop, landmarks within 0.06 px of FP32. The steps between them (anchor decoding, NMS, rotated hand crop, tracking from the previous frame) are not written yet.

## Results

Validation, 30 epochs, window label = label of its last frame:

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
python extract.py [--mode video|image] [--conf 0.5]   # landmarks -> cache/kpt-<mode>-<conf>/, CPU, ~20 frames/s per worker
python stats.py cache/kpt-video-0.5                         # 0/1/2 hands per step
python train.py --head tcn --fill copy [--rot 180] [--drop 0.3] [--name run]   # -> weight/<name>/
```
