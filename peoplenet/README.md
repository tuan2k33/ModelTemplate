# PeopleNet

NVIDIA PeopleNet (DetectNet_v2, pruned ResNet34; classes person, bag, face) as a real **INT8 TensorRT 11 engine**. NGC ships it as an ONNX plus an implicit-quantization calibration cache, which TensorRT 10/11 (strongly typed) no longer uses; `patch_qdq.py` turns the cache into explicit Q/DQ nodes. The result is on Hugging Face: [tuannw/ModelTemplate/peoplenet-qdq-int8](https://huggingface.co/tuannw/ModelTemplate/tree/main/peoplenet-qdq-int8), with its model card, license and notice.

## Results

RTX 3050 Ti Laptop, TensorRT 11.4, batch 1, 960x544, 421-frame street video:

| Engine | GPU time | Persons per frame |
|---|---|---|
| unpruned FP16 (NGC `resnet34_peoplenet.onnx`, v2.6.2) | 5.0 ms | 6.82 |
| this INT8 Q/DQ | 3.6 ms | 6.81 (same count as FP16 in 74% of frames) |

Accuracy against ground truth was not measured.

## Usage

The NGC files are not in this repo (NVIDIA Open Model License): download `nvidia/tao/peoplenet:deployable_quantized_onnx_v2.6.3` from NGC (`resnet34_peoplenet_int8.onnx` and its calibration cache `resnet34_peoplenet_int8.txt`), or take the ready Q/DQ ONNX from Hugging Face and skip the first step. Run from `peoplenet/`:

```bash
python patch_qdq.py resnet34_peoplenet_int8.onnx resnet34_peoplenet_int8.txt peoplenet_qdq.onnx
trtexec --onnx=peoplenet_qdq.onnx --saveEngine=peoplenet_qdq.engine
python run.py peoplenet_qdq.engine in.mp4 out.mp4      # boxes drawn, H264; also prints ms/frame and persons/frame, saves the counts as .npy
```

`patch_qdq.py`: every Conv gets a per-tensor Q/DQ on its activation input (scale from the cache, `EntropyCalibration2`, big-endian float32 = amax/127) and a per-output-channel symmetric Q/DQ on its weights (max|w|/127). Each Conv gets its own pair and its activation scale is multiplied by `1 + 1e-4 * conv_index`: with two Convs reading one tensor through identical Q/DQ scales, TensorRT 11 built an engine whose `output_cov` collapsed to 0 (onnxruntime on the same graph was correct). Debug knobs: a 4th argument `a` / `w` quantizes only activations / weights, a 5th multiplies the activation scales, and the environment variables `LIMIT`, `ONLY`, `SKIP` choose which Convs are touched.

`run.py` decodes DetectNet_v2 on the GPU: input RGB / 255 resized to 960x544 (aspect ratio not kept), coverage [3,34,60] and box offsets [12,34,60] per 16x16 cell, box norm 35, coverage > 0.4, per-class NMS at IoU 0.4.

## Limits

- Batch 1, fixed 960x544 input; the resize and the decoding run outside the engine (unlike the other folders).
- The engine is specific to the GPU and TensorRT version that built it.
- PeopleNet is under the NVIDIA Open Model License, see the Hugging Face folder.
