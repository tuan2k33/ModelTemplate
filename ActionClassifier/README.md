# ActionClassifier

Work in progress. Classify what a person does from a **2 s sequence of pose keypoints** with a small home-made network (MLP or temporal convolution), instead of a video model: the pose model already removes background, clothing and lighting, so the classifier only sees motion and stays tiny.

```
video -> pose model per frame (keypoints of each person, tracked)
      -> last 2 s stacked: [T frames, K keypoints, (x, y, conf)], normalized per person (centre, scale)
      -> small classifier -> label
```

One subfolder per task:

| Folder | Task | Keypoints | Status |
|---|---|---|---|
| [handwash](handwash/README.md) | WHO hand-washing movement (routine / antiseptic), camera above the sink | MediaPipe Hands as one TensorRT engine, 2 x 21 | 85% of frames on hospital footage, 23/25 incomplete washes caught; hands 1.08 ms/frame |

Planned: a person sitting still for a long time (possible stroke), a fall.

## Open questions

- **Pose model.** YOLO26-pose (17 COCO body keypoints, several people, already a TensorRT engine in `../Detector`) has wrists but no fingers, enough for falls and stillness, not for hand-washing steps. Those need hand keypoints (21 per hand, e.g. MediaPipe Hands) or an image of the hands.
- **Datasets** for falls and stillness: to decide. Keypoints should come from the same pose model at train and run time, so datasets are taken as video and the keypoints extracted here, not as ready-made Kinect skeletons.
- **Stillness** over minutes does not fit a 2 s window; it may be a rule on top (keypoint motion below a threshold for N seconds, posture from the classifier).
