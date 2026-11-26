# IntClassifier

A small INT8 image classifier built from a frozen, pre-quantized ResNet50 and a small MLP head you train in under a minute.
The result is **one TensorRT engine**: raw `uint8` image in, label and per-class logits out, all on the GPU.

```
uint8 image [N,H,W,3]
  -> resize 224 / normalize                      (inside the engine)
  -> ResNet50, INT8 QDQ                         (frozen, from the ONNX Model Zoo)
  -> int8 global-average-pool tensor [N,2048]
  -> int8 MLP head 2048-1024-classes (trained here)  (int8 x int8 GEMMs)
  -> label, logits [N,classes]                  (confidence of a class = sigmoid(logit), computed by the caller)
```

Only the head is trained, on cached embeddings, so training a new dataset takes minutes, not hours.

## Results

Frozen backbone + MLP head, default settings, no hyperparameter tuning, one sigmoid output per class. Test accuracy is the int8 head computed exactly on the backbone engine's test embeddings; `engine.py` runs the test set through the final engine and checks it matches (within 1 point; in practice equal).

**ResNet50 INT8** (default, D = 2048):

| Dataset | Classes | Train/val/test (k) | Top-1 Accuracy (%) | Head train time (s) |
|---|---|---|---|---|
| CIFAR-10 | 10 | 45.0 / 5.0 / 10.0 | 95.2 / 90.2 / **90.0** | 13 |
| CIFAR-100 | 100 | 45.0 / 5.0 / 10.0 | 87.0 / 70.1 / **70.0** | 14 |
| MNIST | 10 | 54.0 / 6.0 / 10.0 | 100.0 / 98.8 / **98.6** | 24 |
| EMNIST Balanced | 47 | 101.5 / 11.3 / 18.8 | 93.2 / 87.5 / **86.9** | 38 |
| Tiny ImageNet | 200 | 90.0 / 10.0 / 10.0 | 82.6 / 73.1 / **72.5** | 21 |
| Intel Image Classification | 6 | 12.6 / 1.4 / 3.0 | 95.5 / 92.2 / **92.2** | 6 |
| EuroSAT RGB | 10 | 19.4 / 2.2 / 5.4 | 98.4 / 95.3 / **95.2** | 11 |

**DINOv3 ConvNeXt-Tiny INT8** (`--backbone convnext`, D = 768):

| Dataset | Classes | Train/val/test (k) | Top-1 Accuracy (%) | Head train time (s) |
|---|---|---|---|---|
| CIFAR-10 | 10 | 45.0 / 5.0 / 10.0 | 96.8 / 95.1 / **94.6** | 10 |
| CIFAR-100 | 100 | 45.0 / 5.0 / 10.0 | 90.0 / 81.6 / **82.2** | 9 |
| MNIST | 10 | 54.0 / 6.0 / 10.0 | 99.8 / 98.4 / **98.5** | 25 |
| EMNIST Balanced | 47 | 101.5 / 11.3 / 18.8 | 92.1 / 86.7 / **86.8** | 40 |
| Tiny ImageNet | 200 | 90.0 / 10.0 / 10.0 | 87.6 / 80.7 / **80.0** | 16 |
| Intel Image Classification | 6 | 12.6 / 1.4 / 3.0 | 96.8 / 95.2 / **95.0** | 4 |
| EuroSAT RGB | 10 | 19.4 / 2.2 / 5.4 | 98.9 / 96.2 / **95.6** | 8 |

Speed on an RTX 3050 Ti Laptop (4 GB), TensorRT 11.4, CUDA 13:

| | ResNet50 | ConvNeXt-Tiny |
|---|---|---|
| Engine latency, batch 1 / batch 64 | about 2 ms / 17 ms (about 3,700 img/s at batch 64) | about 3.5 ms / 43 ms (about 1,500 img/s at batch 64) |
| Embeddings for 60k CIFAR images | about 20 s (once, then cached) | about 45 s (once, then cached) |
| Head training | 6-37 s (tables above) | 6-34 s (tables above) |
| Engine build (`engine.py`, only for a run worth keeping) | about 2 min | about 1 min |
| GPU memory for the engine | about 250 MB | about 390 MB |

## Usage

The backbone is a ready INT8 Q/DQ ONNX downloaded from the Hugging Face Hub on first use (`huggingface_hub`, public repo, no token needed): `--backbone resnet50` (default, from the ONNX Model Zoo `resnet50-v1-12-int8`) or `--backbone convnext` (DINOv3 ConvNeXt-Tiny), both in the [tuannw/ModelTemplate](https://huggingface.co/tuannw/ModelTemplate) repo with its model card and license; `export.py` rebuilds both from the upstream models and pushes them. The datasets are downloaded on first use with `curl`, so it must be installed. Datasets live in `data/`. Run from `IntClassifier/`:

```bash
python train.py cifar10
python train.py cifar10 --backbone convnext   # writes weight/cifar10-convnext/
python train.py tiny-imagenet          # also: cifar100, mnist, emnist (downloaded on first use)
python train.py intel                   # your own folder: data/intel/{train,test}/<class>/*.jpg (or any path)
python engine.py weight/cifar10         # model.onnx -> model.engine for a run you keep, checked against the run and timed
python predict.py weight/cifar10/model.engine image.jpg
```

Options: `--epochs N` (default: train until early stop), `--stages` (default `1e-3:10,5e-4:10,1e-4:10`: lr:patience pairs, a lower lr each time the val loss stalls for `patience` epochs), `--eps` (Adam eps, default 1e-8), `--bs` (head batch size, default 1024), `--backbone resnet50|convnext`, `--seed` (default: random, saved in `metrics.json`), `--per-class N`, `--test-per-class N`.

Each run writes `weight/<dataset>/` (`weight/<dataset>-convnext/` for ConvNeXt; replacing the previous run): `model.onnx` (backbone + int8 head in one graph), `metrics.json` (dataset, train/val/test accuracy and loss, timings, GPU and RAM use) and `train.log`. A run takes seconds once the embeddings are cached, so trying seeds or settings is cheap; `engine.py` then adds `model.engine` and an `engine` entry in `metrics.json` (engine accuracy, latency, GPU memory).
`predict.py` reads the class names from the `.onnx` next to the `.engine` and turns the top logit into a confidence with a sigmoid.

Code: `train.py` (head training and the run, up to the ONNX), `engine.py` (TensorRT engine of a run), `model.py` (backbone download, QLinear to QDQ conversion, engine runner, head graph), `export.py` (builds and pushes the backbone ONNX), `dataset.py` (loaders), `predict.py`.
Your own data: put it in `<root>/train/<class>/` and `<root>/test/<class>/` (same class folders in both) and pass `<root>` (a bare name is looked up in `data/`). To add a downloadable dataset, write a loader in `dataset.py` and register it in `DATASETS`. The Intel and EuroSAT numbers above come from the Kaggle downloads arranged this way (Intel: `seg_train` and `seg_test` renamed to `train` and `test`; EuroSAT: every 5th image of each class moved to `test`).

## Limits

- The engine is specific to the GPU and TensorRT version that built it.
- Input is resized straight to 224x224 with plain bilinear (no antialiasing, aspect ratio not kept), no crop, in training and inference. Fine for small or square images, it aliases when downscaling large photos. The engine takes sides up to 512, bigger images are shrunk to 512 on the CPU first (antialiased). Batch at most 64.
- The normalization at the start of the engine, the global-average-pool, the bias add and the final scaling of the logits are float operations on integer values. TensorRT has no `MatMulInteger` or `BitShift`.
- Accuracy numbers are not tuned.
