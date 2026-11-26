# ModelTemplate

Templates for turning pretrained and pre-quantized vision models into **single TensorRT engines** for edge GPUs: the pre-processing (resize, letterbox, normalization) is built into the engine, so a raw `uint8` image goes in and results come out, all on the GPU. Each folder is an independent project with its own README (results, usage, limits).

| Folder | What | Engine |
|---|---|---|
| [IntClassifier](IntClassifier/README.md) | Image classifier: frozen INT8 backbone (ResNet50 or DINOv3 ConvNeXt-Tiny) + small int8 MLP head trained in under a minute | INT8 |
| [Detector](Detector/README.md) | YOLO26 object detection, fine-tune with Ultralytics, letterbox inside the engine; optional hand-placed INT8 | FP16 |
| [FaceID](FaceID/README.md) | ArcFace ResNet100 face verification, ONNX Model Zoo INT8 converted to Q/DQ, LFW / CFP-FP / AgeDB-30 | INT8 |
| [Motion](Motion/README.md) | Background model for fixed cameras as a stateful engine, optionally fused with the detector for several cameras per call | FP32 state |
| [OCR](OCR/README.md) | Work in progress: YOLO26 text-line detector, page deskew, and [dewarp](OCR/dewarp/README.md) (phone photo of a curved page to a flat A4, CPU) | FP16 |
| [peoplenet](peoplenet/README.md) | NVIDIA PeopleNet detector: NGC calibration cache turned into explicit Q/DQ for TensorRT 11 | INT8 |

## Models on Hugging Face

The INT8 Q/DQ ONNX models are in [tuannw/ModelTemplate](https://huggingface.co/tuannw/ModelTemplate), one folder per model with its model card and license: the ResNet50 and DINOv3 ConvNeXt-Tiny backbones used by IntClassifier (downloaded on first use) and the PeopleNet detector. Build the engine on the target GPU.

## Setup

Needs an NVIDIA GPU, CUDA 13, Python 3.14, PyTorch with CUDA, TensorRT 11 (Python package and `trtexec` on `PATH`), `onnx`, `onnxruntime`, `cuda-python`, `torchvision`, `opencv-python`, `huggingface_hub`, and per project `ultralytics` (Detector, OCR, Motion) or `h5py` (OCR/dewarp benchmark). Datasets and Model Zoo weights are downloaded on first use with `curl`. A Hugging Face token (`HF_TOKEN` in the environment or in `.env`) is needed for the private backbones. Every script runs from inside its own folder and writes to its `data/`, `cache/` and `weight/` (git-ignored).

Measurements in the READMEs come from an RTX 3050 Ti Laptop (4 GB) with TensorRT 11.4 or 11.5 (stated where it matters). Engines are tied to the GPU and TensorRT version that built them.

## Licenses

No license file for the code yet. The models keep their own licenses: YOLO26 is AGPL-3.0 (Ultralytics), DINOv3 under the DINOv3 License, PeopleNet under the NVIDIA Open Model License, the ONNX Model Zoo models under Apache-2.0; see the model cards on Hugging Face.

## AI Disclosure

Written with Claude Code (Anthropic) under the author's direction: the author chooses what to build and which approaches to keep or drop, and reviews the results. The measurements quoted in the READMEs come from runs on the author's machine.
