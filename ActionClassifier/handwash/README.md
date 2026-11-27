# Hand washing steps

Which WHO hand-washing movement (routine / antiseptic washing with water) is being done, from 2 s of hand landmarks. Two TensorRT engines:

```
video -> hands engine (hands.py): MediaPipe Hands as one graph, uint8 frame + tracking state in, 2 hands x 21 landmarks out, 1.08 ms/frame
      -> 2 slots kept apart by wrist position (MediaPipe's left/right flips when seen from above)
      -> one window per frame: 32 frames (2 s at 16 fps)
      -> classifier engine (train.Window): missing hand filled with a copy of the other one (presence flag kept),
         normalized by mean wrist and hand size, + frame differences, TCN -> 7 classes: other + WHO movements 1-6
```

Default model 10 fps x 1 s, labelled at the window's middle: on another hospital's videos (PSKUS, staff at work, 30 fps) 85.3% of frames right, macro 70.9 (88.2 / 77.3 allowing +-0.5 s at step changes), 23 of 25 incomplete washes caught; on METC validation (14 people never seen in training) 69.5 / 66.0, where 2 s windows do better (77.1 / 75.0). Frames after a 1 s vote (argmax per frame, most frequent label around it).

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
python train.py --kpt cache/kpt-engine-0.5 --head tcn --fill copy --rate 10 --seconds 1 --anchor center --name tcn-r10-s1-center   # -> weight/<name>/model.pt, model.onnx, metrics.json
python run.py video.mp4 [--out out/video.mp4] [--labels annotation.json] [--model weight/tcn-r10-s1-center] [--landmarks] [--window model|native]   # step per frame and a summary; engines built on first use
python fps_test.py                                       # the frame-rate / window table below (PSKUS DataSet7 in data/handwash/pskus)
python eval.py weight/tcn-r10-s1-center weight/tcn-r10-s2  # whole videos: frames strict and +-0.5 s, missing steps per video; METC val and PSKUS
python extract.py --mode engine --fps 20                 # METC re-encoded to 20 fps first; train with --kpt cache/kpt-engine-0.5-20fps --src-fps 20
```

`run.py` reads the whole video through the hands engine (one frame per call, the tracking needs the previous frame; ~6 ms per frame with OpenCV decoding, which dominates), then classifies one window per frame in batches of 512 (all 1980 windows of a 2-minute video in ~0.2 s) and prints the steps with their times. Other frame rates are resampled to 16 fps inside the windows. On a validation video the steps 1-3 come out with the right times, while thumb rubbing (25 s) is mostly read as other / scattered steps.

## Window length and frame rate

The window is `--seconds` long with frames `--rate` per second (8 = every 2nd METC frame). `run.py` resamples any video to the model's rate (`--window model`, default) or feeds every frame (`--window native`). Two test sets: METC validation windows, and 30 PSKUS videos (zenodo 4537209 DataSet7: another hospital, staff at work, 320x240 at 30 fps, re-encoded to 25 and 10 fps with ffmpeg), frames after 1 s smoothing. Accuracy / macro (mean over the steps; PSKUS frames are 60% "other"):

| Window | METC val | PSKUS 30 fps | 25 fps | 10 fps | 30 fps native | 25 fps native | 10 fps native |
|---|---|---|---|---|---|---|---|
| 16 fps x 2 s | 70.6 / 68.8 | 80.2 / 62.4 | 80.1 / 59.8 | 78.9 / 57.5 | 82.0 / 63.0 | 82.5 / 62.2 | 76.5 / 55.1 |
| 16 fps x 3 s | 72.5 / 71.0 | 77.6 / 59.3 | 77.9 / 56.9 | 77.5 / 53.9 | | | |
| 16 fps x 4 s | 72.5 / 71.4 | 76.8 / 57.0 | 76.8 / 54.4 | 76.1 / 51.6 | | | |
| 16 fps x 5 s | 73.1 / 72.8 | 74.5 / 48.7 | 75.7 / 49.6 | 75.1 / 45.3 | | | |
| **10 fps x 2 s** | 73.0 / 71.6 | 78.5 / 61.0 | 78.7 / 60.8 | 77.8 / 59.4 | 78.2 / 53.1 | 80.8 / 57.1 | 77.8 / 59.4 |
| 10 fps x 2.5 s | 73.9 / 72.6 | 78.3 / 60.7 | 78.9 / 58.9 | 77.0 / 58.1 | | | |
| 10 fps x 3 s | 74.4 / 73.0 | 76.8 / 58.4 | 77.8 / 57.5 | 76.3 / 56.2 | | | |
| 8 fps x 2 s | 72.7 / 71.2 | 79.9 / 62.4 | 80.7 / 60.9 | 79.0 / 60.2 | 76.6 / 45.0 | 78.8 / 49.6 | 79.3 / 60.7 |
| 8 fps x 3 s | 74.5 / 73.4 | 78.3 / 59.0 | 79.0 / 58.1 | 77.0 / 56.4 | | | |
| 8 fps x 4 s | 75.0 / 74.1 | 77.6 / 58.7 | 78.6 / 57.3 | 78.0 / 56.1 | | | |
| 8 fps x 5 s | 74.5 / 73.8 | 75.2 / 55.0 | 76.4 / 55.0 | 76.1 / 53.7 | | | |

- 8-10 fps beats 16 fps inside the window: frames 1/16 s apart barely move. 8 and 10 fps are within a point of each other (about the spread between seeds). 10 fps is resampled from METC's 16 fps to the nearest frame.
- The best length follows how long a step lasts: METC volunteers hold a step for 6.8 s (median segment), so 4-5 s windows win there; PSKUS staff switch every 2.6 s, so anything above 2 s mixes steps and loses. 2 s is the safe default until there is footage from the target sinks.
- Native windows (every frame) only work when the video's rate is close to the model's: a 16 fps model on 25-30 fps video gains a little, an 8 fps model on 25-30 fps video loses 13-17 macro points. Resampling to the model's rate is the general setting; `weight/tcn-r10-s2` (10 fps x 2 s = 20 frames) is the default of `run.py` (59-61 macro at 10-30 fps).
- The 10 fps column also runs the hands engine on 10 fps video (tracking with 3x larger steps between frames): 1.6 macro points below 30 fps, for a third of the work.

## Short windows, label at the middle

Most errors sit at step changes (10 fps x 2 s, label at the window's end): on PSKUS 43% right within 1 s of a change against 90% inside a step. Two fixes, measured on whole videos with `eval.py` (1 s smoothing; "+-0.5 s" also counts a prediction that matches any label within half a second, since the two PSKUS annotators agree on 92.9% of frames and differ mostly at changes). Accuracy / macro:

| Model | METC val strict | METC +-0.5 s | PSKUS strict | PSKUS +-0.5 s |
|---|---|---|---|---|
| 10 fps x 2 s, label at the end | 77.1 / 75.0 | 81.3 / 79.6 | 78.5 / 61.0 | 82.6 / 70.3 |
| 10 fps x 2 s, label at the middle | 73.0 / 70.1 | 77.7 / 75.2 | 82.3 / 64.8 | 86.1 / 73.4 |
| 16 fps x 2 s, end | 75.0 / 72.5 | 79.1 / 76.9 | 80.2 / 62.4 | 84.2 / 71.7 |
| 16 fps x 1 s, end | 72.2 / 68.6 | 76.8 / 73.6 | 84.2 / 69.1 | 87.8 / 77.2 |
| **16 fps x 1 s, middle** | 68.8 / 65.1 | 74.1 / 71.0 | **85.9 / 71.9** | **88.7 / 78.5** |
| 10 fps x 1 s, middle | 69.5 / 66.0 | 74.7 / 71.7 | 85.3 / 70.9 | 88.2 / 77.3 |

- A label at the window's end is about half a window late; at the middle (`--anchor center`, needs half a window of future: fine offline, a 0.5 s delay live) it lines up with the step.
- 1 s windows fit staff who switch steps every 2.6 s (one or two rubs per second, the user's reasoning); METC volunteers hold a step for 6.8 s and prefer 2 s. The two sets pull in opposite directions; the hospital footage is the one that looks like deployment, so `run.py` defaults to a 1 s window labelled at its middle (now at 10 fps, see below).
- 20-25 fps cannot be trained from METC (16 fps); PSKUS (30 fps) would allow it.

## 20 fps, interpolation, whole videos

`eval.py`, same sets, accuracy / macro (strict | +-0.5 s). 20 fps training data is METC re-encoded with ffmpeg (`extract.py --fps 20`: one frame in four repeated, no new motion); `--interp` fills a hand missing up to that many seconds by straight lines between where it was last and next seen (offline; its presence flag stays 0):

| Model | METC val | PSKUS |
|---|---|---|
| 10 fps x 2 s, end | 77.1 / 75.0 \| 81.3 / 79.6 | 78.5 / 61.0 \| 82.6 / 70.3 |
| **10 fps x 1 s, middle** | 69.5 / 66.0 \| 74.7 / 71.7 | **85.3 / 70.9 \| 88.2 / 77.3** |
| 10 fps x 1 s, middle, interp 0.5 s | 69.2 / 65.2 \| 74.4 / 70.9 | 85.2 / 70.9 \| 88.1 / 77.3 |
| 10 fps x 1 s, middle, interp 3 s | 68.2 / 65.2 \| 73.3 / 70.7 | 83.0 / 66.8 \| 86.1 / 73.9 |
| 10 fps x 1 s, middle, interp 10 s | 67.2 / 63.9 \| 72.1 / 69.1 | 82.0 / 64.7 \| 85.3 / 72.1 |
| 20 fps x 2 s, end | 76.3 / 74.1 \| 80.4 / 78.6 | 78.5 / 60.5 \| 82.7 / 70.3 |
| 20 fps x 2 s, middle | 71.6 / 68.0 \| 76.3 / 73.1 | 83.0 / 65.2 \| 86.7 / 73.4 |
| 20 fps x 1 s, middle | 69.3 / 65.4 \| 74.6 / 71.2 | 84.1 / 66.4 \| 87.1 / 74.0 |
| 20 fps x 1 s, middle, interp 3 s | 67.8 / 64.1 \| 72.8 / 69.7 | 83.2 / 67.1 \| 86.2 / 74.2 |

- 20 fps does not beat 10 fps (with repeated frames, so not a fair test of real 20 fps footage). The TCN sees 17 frames per output position: 1.7 s at 10 fps, 0.85 s at 20 fps.
- Interpolation does not help: short gaps change nothing, long ones (3-10 s) cost 2-6 macro points; a straight line drawn across seconds is motion that did not happen. Copying the other hand stays.

What matters in use is whether a wash missed a step. Per video, a step is done when it has >= 1 s in the labels and reported done when it has >= 1 s predicted (10 fps x 1 s, middle):

| | METC val (42 videos) | PSKUS (30 videos) |
|---|---|---|
| missing steps caught | 20 / 31 | 74 / 90 |
| done steps reported missing (false alarms) | 10 / 221 | 12 / 90 |
| incomplete washes caught | 9 / 11 | 23 / 25 |
| complete washes reported incomplete | 2 / 31 | 2 / 5 |

Requiring 2 s of a step before calling it done catches more missing steps (22 / 31, 83 / 90) for about twice the false alarms; `eval.py` prints 0.5, 1 and 2 s for every model. Only 5 of the 30 PSKUS videos are complete washes, so the last line there is thin.

## Post-processing

`eval.py` saves the per-frame logits (cache/probs/<model>.npy; `run.py --csv` writes them for one video) and `post.py` tries rules on them (softmax only there). 10 fps x 1 s, middle, PSKUS:

| After the classifier | strict | +-0.5 s | missing steps caught | false alarms | incomplete washes caught |
|---|---|---|---|---|---|
| most frequent label within 1 s (run.py) | 85.4 / 70.8 | 88.2 / 77.2 | 73 / 90 | 12 / 90 | 23 / 25 |
| mean of the softmax over 1 s (earlier) | 85.3 / 70.9 | 88.2 / 77.3 | 74 / 90 | 12 / 90 | 23 / 25 |
| argmax per frame, no vote | 79.4 / 65.8 | 81.9 / 71.3 | 62 / 90 | 8 / 90 | 20 / 25 |
| "other" only in the first and last 5 s | 39.1 / 68.7 | 41.9 / 73.1 | 33 / 90 | 5 / 90 | 14 / 25 |
| inside the wash (first to last 1 s of steps), next best step when P(other) <= 0.5 | 81.1 / 74.4 | 83.9 / 80.2 | 66 / 90 | 10 / 90 | 22 / 25 |
| HMM, step order from METC training labels | 75.0 / 74.6 | 77.8 / 79.6 | 50 / 90 | 10 / 90 | 19 / 25 |

"Other" is not only at the ends: 10-16% of the frames between the first and last step, and PSKUS staff spend a median 31 s after the last step (rinsing, drying, the tap). Replacing "other" by a step raises the macro but reports steps that were not done, so fewer incomplete washes are caught: keep the plain vote to judge a wash, use the in-wash rule only to draw a cleaner timeline.

## Limits

- Only routine / antiseptic hand washing (one model). Surgical scrub needs its own data: no public RGB set with step labels was found (MIVIA's surgical hand-washing set is depth only).
- The hands engine is built for one frame size and one camera per call; the classifier engine takes any batch of windows.
- Landmarks find only about half the hands (motion blur, hands entering from the frame edge, hands on top of each other).
