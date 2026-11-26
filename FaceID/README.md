# FaceID

Face verification with the ONNX Model Zoo **ArcFace ResNet100 INT8** as **one TensorRT engine**: aligned face as raw `uint8` in, 512-d embedding out. Evaluation only, nothing is trained.

```
uint8 aligned face [N,H,W,3] RGB (any size up to 512)
  -> resize 112x112, (x - 127.5) / 128      (inside the engine)
  -> ArcFace ResNet100, INT8 Q/DQ            (ONNX Model Zoo arcfaceresnet100-11-int8, QLinear converted to Q/DQ)
  -> float embedding [N,512]                 (not normalized; compare with cosine)
```

The Model Zoo graph is QLinear with uint8 activations and non-zero zero-points, which TensorRT does not take; `model.qdq_model` rewrites it as int8 Q/DQ with zero-point 0 (activation scale s becomes 2s after a ReLU-like zero-point 0, `max(zp, 255 - zp) * s / 127` otherwise). Weights keep their exact values; the float ops between the quantized ones (PReLU, BatchNorm) stay as they are.

## Results

Verification accuracy, 10-fold (cosine threshold picked on 9 folds, tested on the 10th), engine built with TensorRT 11.4 on an RTX 3050 Ti Laptop:

| Set | Pairs | Accuracy (%) |
|---|---|---|
| LFW | 6000 | 99.62 +- 0.33 |
| CFP-FP | 7000 | 94.83 +- 1.09 |
| AgeDB-30 | 6000 | 97.52 +- 0.97 |

The sets are the InsightFace `.bin` files (faces already aligned to 112x112), downloaded on first use from the `gaunernst/face-recognition-eval` dataset on Hugging Face. JPEG decoding (nvJPEG), the engine, the cosine and the folds all run on the GPU.

## Usage

Run from `FaceID/`:

```bash
python eval.py                                   # all three sets with the engine (built on first use into cache/)
python eval.py --sets lfw --pairs 600            # quick check on the first 600 pairs
python eval.py --models engine,qdq,orig          # also the Q/DQ ONNX and the original QLinear ONNX on the CPU (onnxruntime, slow)
```

Code: `model.py` (Model Zoo download, QLinear to Q/DQ, the uint8 front end, engine build and runner), `dataset.py` (download and GPU decoding of the sets), `eval.py`. Embeddings are cached per set and model in `cache/emb-<set>-<model>.npz` and recomputed when the model is newer.

## Limits

- Faces must already be aligned (5-point, 112x112 crop as in InsightFace); there is no face detector or alignment here.
- The engine is specific to the GPU and TensorRT version that built it.
