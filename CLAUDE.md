# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ModelTemplate turns pre-quantized / pretrained models into **single TensorRT engines** where pre-processing runs inside the engine (raw `uint8` image in). Independent sub-projects, each with its own README (results, options, limits); read it before changing one. The root README is only the overview and the HF link:

- `IntClassifier/` — frozen INT8 backbone (ResNet50 or DINOv3 ConvNeXt-T, Q/DQ ONNX pulled from the user's HF repos, built by `export.py`) + small MLP head trained on cached embeddings.
- `Detector/` — YOLO26 FP16 engine with the letterbox built into the graph. INT8 is only an option (`quantize.py`, hand-placed Q/DQ, SiLU float or as an int8 lookup table); it is slower than FP16 here, see its README.
- `OCR/` — work in progress: YOLO26 text-line detector fine-tuned on SROIE (`det.py`, engine built with `../Detector/train.py`) a deskew benchmark (`deskew.py`) and `dewarp/` (photo of a curved page -> flat A4 on the CPU: image-processing boundary + homography/Coons remap; optional YOLO26n-pose paper keypoints, paused; plan in its README). No recognizer yet.
- `FaceID/` — ArcFace ResNet100 INT8 → QDQ engine, evaluated on LFW / CFP-FP / AgeDB-30 (`eval.py` only, no train).
- `Motion/` — background model for fixed cameras as a stateful TensorRT engine (no trained weights); see its README; the module docstrings in `model.py`, `run.py`, `build.py` document inputs, state layout and options.
- `peoplenet/` — NGC PeopleNet INT8: `patch_qdq.py` (calibration cache -> explicit Q/DQ, result on HF), `run.py` (video demo). Only the code is in git; the NGC files (`cnn/`, `cnn_int8/`, NVIDIA license) and outputs are ignored.

## Running

No build, lint or test suite. Each sub-project is a set of scripts run **from inside its own directory** (they import siblings like `model`, `dataset` and use relative `data/`, `cache/`, `weight/`). Use the project venv (`.venv` → `~/.dev/venvs/ModelTemplate`, py3.14). Needs an NVIDIA GPU, TensorRT 11 with `trtexec` on `PATH`, and `curl` (datasets and Model Zoo weights download on first use).

```bash
cd IntClassifier && python train.py cifar10 [--backbone convnext]   # also cifar100, mnist, emnist, tiny-imagenet, or a folder with train/<class>/ and test/<class>/
cd IntClassifier && python engine.py weight/cifar10          # model.onnx -> model.engine (train.py stops at the ONNX)
cd IntClassifier && python predict.py weight/cifar10/model.engine image.jpg
cd Detector      && python train.py --epochs 0 --name coco     # export only; or pass a data.yaml to fine-tune
cd Detector      && python predict.py weight/coco/model.engine image.jpg --out result.jpg
cd Detector      && python eval.py weight/coco/model.engine --limit 200   # COCO val2017 mAP; --start/--limit for subsets
cd Detector      && python quantize.py [--model X.pt --images DIR] [--silu float|lut] --name int8   # optional INT8 engine
cd OCR           && python det.py --epochs 50 && python deskew.py
cd OCR/dewarp    && python dewarp.py photo.jpg flat.jpg && python bench_uvdoc.py   # UVDoc benchmark in data/uvdoc
cd FaceID        && python eval.py --sets lfw --pairs 600
cd Motion        && python run.py in.mp4 out.mp4 [--detector ../Detector/weight/coco/model.onnx]   # background model; build.py makes the fused multi-camera engine
```

"Single test" equivalents: `--limit N` (Detector eval), `--pairs N` (FaceID eval), `--per-class N` / `--test-per-class N` (IntClassifier train).

## Architecture notes

- Each sub-project has `model.py` holding the ONNX graph construction (QDQ rewrite or letterbox/normalize wrapper), `compile_engine` (shells out to `trtexec`) and an `Engine` runner using TensorRT + torch CUDA tensors. Scripts are thin around it.
- Backbones come from `model.BACKBONES` (folders of the private HF repo `tuannw/ModelTemplate`, which also holds the PeopleNet INT8 ONNX; token from `../.env`); `python export.py resnet50|convnext|push` rebuilds them. ConvNeXt (`export.NHWC`) is the HF model rewritten channels-last: stem/downsample as strided slices + MatMul, transposes only around the 18 depthwise convs (ONNX Conv is NCHW). Q/DQ is hand-placed: int8 only on the stem/downsample MatMuls and pointwise Linears; depthwise, LayerNorm, GELU and the residual stay FP16 (int8 depthwise and a channels-first rewrite were both slower on TensorRT). Both backbones have FP16 pre-processing (ResNet50: opset 19, the first Q takes an FP16 scale). `export.unsubnormal` keeps every INT8 MatMul's act x weight scale out of the FP16 subnormal range (TensorRT's Myelin GEMMs lose ~1 point otherwise); keep it when touching the ConvNeXt export.
- QLinear → QDQ conversion is needed because TensorRT rejects uint8 activations with non-zero zero-points; activations become int8 symmetric. Weights/biases keep exact values.
- IntClassifier caches backbone embeddings in `cache/<dataset>-<backbone>-*.npz`, trains the head on the GPU on those, then appends the int8-quantized head to the backbone graph (`model.onnx`); `engine.py` builds and checks the engine of a run worth keeping. Output is label + logits; confidence = sigmoid(logit) computed by the caller (no `exp()` in the engine).
- Outputs go to `weight/<name>/` (IntClassifier: `model.onnx`, `metrics.json`, `train.log`, plus `model.engine` from `engine.py`; Detector: also `weights/best.pt`) and a rerun replaces the previous one. `predict.py` reads class names from the `.onnx` next to the `.engine`.
- Motion keeps its memory outside the engine: `state` [B,8,180,320] goes in and `state_out` comes out, one row per camera; state stays FP32 on purpose (FP16 loses the small updates). Raise `VER` in `Motion/model.py` when its graph changes (part of the engine cache name).
- Hand-placed Q/DQ on TensorRT 11.5: a conv whose input DQ TensorRT cannot fuse (shared by two convs, or also feeding a Concat) runs in FP16 with the DQ'd per-channel weights and silently returns NaN/0 (`Detector/lut_unit4.py`). Check with `Detector/lut_layers.py` (lists such convs) and `Detector/lut_dbg.py` (TensorRT vs onnxruntime layer by layer; run ORT with graph optimizations off, its own Q/DQ fusion is not a reference).
- Engines are tied to the GPU + TensorRT version that built them; `data/`, `cache/`, `weight/`, `__pycache__/` are git-ignored per sub-project.

## Conventions

- Test/run precision order: FP16 >> BF16 >> FP32; no autocast on FP32 weights.
- Videos written by scripts must be H264.
- Keep inputs/outputs inside the project, not `/tmp`. No `Co-Authored-By` or AI trailers in commits; AI disclosure goes in the README ("AI Disclosure" section).
- Project decisions that cannot be read from code are kept in `/mnt/c/Dev/.claude/memory/project_ModelTemplate.md`.
