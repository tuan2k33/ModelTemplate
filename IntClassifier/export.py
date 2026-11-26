"""Backbones as ready-to-run INT8 Q/DQ ONNX (uint8 NHWC `image` [N,H,W,3] in, any size; int8 `feat_q` [N,D] out, feat = feat_q * metadata feat_scale),
built once from the upstream weights and pushed to the Hugging Face Hub, so training a head only needs the file:
    python export.py resnet50            # ONNX Model Zoo resnet50-v1-12-int8 (QLinear) -> weight/backbone/resnet50-qdq-int8.onnx, D=2048
    python export.py convnext            # facebook/dinov3-convnext-tiny-pretrain-lvd1689m -> weight/backbone/dinov3-convnext-tiny-qdq-int8.onnx, D=768
    python export.py push [resnet50|convnext]   # into the HF repo model.HUB (needs a write HF_TOKEN), folder <file stem>/, with the license and a model card
ConvNeXt: Q/DQ only where int8 math runs, the activation input and the weight of the stem/downsample MatMuls and the pointwise Linears (40 of 58).
Depthwise convs stay FP16 (the most sensitive layers, and int8 depthwise was slower on TensorRT: layout changes around it); LayerNorm, GELU and the
residual stream stay FP16 with no Q/DQ around them (TensorRT fuses GELU into the first pointwise GEMM and the following Q into LayerNorm).
Layout (class NHWC): channels-last everywhere, the only transposes are the 2 around each of the 18 depthwise convs (ONNX Conv is NCHW): 36 against 47
for the HF graph, 10% faster. A channels-first rewrite (1x1 convs, LayerNorm over channels) removed all transposes but was slower at every batch.
Calibration: per-tensor activation amax = 99.99th percentile of |x| over 256 COCO val2017 images (data/coco/val2017), weights per output channel."""
import json, os, sys
from pathlib import Path

import numpy as np, onnx, torch, torch.nn.functional as F
from onnx import helper as h, numpy_helper as nh

from model import BACKBONES as NAMES, BATCH, HUB, MEAN, STD, qdq_backbone

OUT = Path("weight/backbone")
CONVNEXT = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
PCT, NCAL = 99.99, 256


def resnet50():
    m = qdq_backbone()
    s = next(float(nh.to_array(i)) for i in m.graph.initializer if i.name == "resnetv17_pool1_fwd_quantized_s")
    return m, dict(feat_scale=s, source="https://huggingface.co/onnxmodelzoo/resnet50-v1-12-int8", license="apache-2.0")


def lin(w, b):
    """Strided conv weight [O,C,k,k] -> Linear over the k*k strided slices concatenated in (i, j, c) order."""
    m = torch.nn.Linear(w.shape[1] * w.shape[2] * w.shape[3], w.shape[0]).to(w)
    m.weight.data, m.bias.data = w.permute(0, 2, 3, 1).reshape(w.shape[0], -1).clone(), b.clone()
    return m


def ln(src):
    m = torch.nn.LayerNorm(src.normalized_shape, eps=src.eps).to(src.weight)
    m.weight.data, m.bias.data = src.weight.data.clone(), src.bias.data.clone()
    return m


class NHWC(torch.nn.Module):
    """The HF ConvNeXt (same weights) with the residual stream channels-last: uint8 NHWC image -> resize 224 (bilinear, no antialias, as the ResNet50
    backbone) -> ImageNet normalize -> stem/downsample as strided slices + MatMul -> blocks (only the depthwise conv sees NCHW: 2 transposes each)
    -> mean over H, W -> final LayerNorm = the HF pooler_output (the HF LayerNorm of [pooled, patch tokens] is per token, so the patch tokens are dropped)."""
    def __init__(self, hf):
        super().__init__()
        self.register_buffer("scale", torch.tensor(1 / (255 * STD), dtype=torch.float16))
        self.register_buffer("shift", torch.tensor(-MEAN / STD, dtype=torch.float16))
        self.down, self.k = torch.nn.ModuleList(), [4, 2, 2, 2]  # per stage: (norm before, slices -> linear, norm after)
        for i, s in enumerate(hf.model.stages):
            d = s.downsample_layers
            self.down.append(torch.nn.ModuleList([torch.nn.Identity(), lin(d[0].weight.data, d[0].bias.data), ln(d[1])] if i == 0 else
                                                 [ln(d[0]), lin(d[1].weight.data, d[1].bias.data), torch.nn.Identity()]))
        self.stages = torch.nn.ModuleList([s.layers for s in hf.model.stages])
        self.norm = hf.layer_norm
        self.convs, self.linears = [], []  # in execution order = the order of the ONNX Conv / MatMul nodes
        for d, layers in zip(self.down, self.stages):
            self.linears.append(d[1])
            for l in layers:
                self.convs.append(l.depthwise_conv); self.linears += [l.pointwise_conv1, l.pointwise_conv2]

    def forward(self, image):
        x = image.half().permute(0, 3, 1, 2)  # this transpose pair is removed from the ONNX by resize_nhwc()
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        x = x * self.scale + self.shift
        for (pre, l, post), k, layers in zip(self.down, self.k, self.stages):
            x = pre(x)
            x = post(l(torch.cat([x[:, i::k, j::k, :] for i in range(k) for j in range(k)], -1)))
            for b in layers:
                y = b.depthwise_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                x = x + b.pointwise_conv2(b.activation_fn(b.pointwise_conv1(b.layer_norm(y)))) * b.gamma
        return self.norm(x.mean((1, 2)))


def resize_nhwc(g):
    """Transpose -> Resize -> Transpose (torch interpolate needs NCHW) -> one Resize on NHWC, sizes [N, 224, 224, 3]."""
    rs = next(n for n in g.node if n.op_type == "Resize")
    t0 = next(n for n in g.node if rs.input[0] in n.output); t1 = next(n for n in g.node if n.op_type == "Transpose" and n.input[0] == rs.output[0])
    assert t0.op_type == "Transpose"
    g.initializer.append(nh.from_array(np.array([224, 224, 3], np.int64), "nhwc_hwc"))
    rs.input[0], rs.input[3] = t0.input[0], "nhwc_sizes"
    nodes = []
    for n in g.node:
        if n is rs:
            nodes += [h.make_node("Shape", [t0.input[0]], ["nhwc_n"], start=0, end=1), h.make_node("Concat", ["nhwc_n", "nhwc_hwc"], ["nhwc_sizes"], axis=0)]
        if n is not t0 and n is not t1:
            n.input[:] = [rs.output[0] if x == t1.output[0] else t0.input[0] if x == t0.output[0] else x for x in n.input]
            nodes.append(n)
    while True:  # drop what only computed the old NCHW sizes
        used = {x for n in nodes for x in n.input} | {o.name for o in g.output}
        keep = [n for n in nodes if any(o in used for o in n.output)]
        if len(keep) == len(nodes):
            break
        nodes = keep
    del g.node[:]; g.node.extend(nodes); del g.value_info[:]  # recorded shapes of the NCHW resize are stale


def unsubnormal(g, target=2.0 ** -12):
    """TensorRT's INT8 GEMMs (Myelin) lose precision when act scale x weight scale is an FP16 subnormal (< 6.1e-5; the downsample MatMuls are ~5e-7,
    5-20% error per channel, ~1 point of CIFAR-100 / Tiny ImageNet; conv INT8 kernels keep FP32 scales and are fine). Per MatMul whose product is below
    `target`: weight scale x K (K = 2^n, the int8 weights unchanged) and bias x K, then the 1/K goes into the next op where that is free (the following
    LayerNorm ignores the scale, the layer-scale gamma takes it) or into a Mul only where the product is a true subnormal (the downsamples). Exact in math."""
    init = {i.name: i for i in g.initializer}
    prod, cons = {o: n for n in g.node for o in n.output}, {}
    for n in g.node:
        for x in n.input:
            cons.setdefault(x, []).append(n)
    f16 = lambda name, v: init[name].CopyFrom(nh.from_array(np.asarray(v, np.float16), name))
    arr = lambda name: nh.to_array(init[name]).astype(np.float32)
    out = []
    for n in g.node:
        out.append(n)
        if n.op_type != "MatMul":
            continue
        a, wdq = float(arr(prod[n.input[0]].input[1])), prod[n.input[1]]
        lo = a * arr(wdq.input[1]).min()
        k = 2.0 ** max(0, int(np.ceil(np.log2(target / lo))))
        add = cons[n.output[0]][0]; bias = next(x for x in add.input if x in init)
        nxt = cons[add.output[0]]
        fold = len(nxt) == 1 and (nxt[0].op_type == "LayerNormalization" or (nxt[0].op_type == "Mul" and any(x in init for x in nxt[0].input)))
        if k == 1 or not (fold or lo < 6.1e-5 / 16):
            continue
        f16(wdq.input[1], arr(wdq.input[1]) * k)
        if fold:
            f16(bias, arr(bias) * k)
            if nxt[0].op_type == "Mul":
                gamma = next(x for x in nxt[0].input if x in init); f16(gamma, arr(gamma) / k)
        else:
            g.initializer.append(nh.from_array(np.array(1 / k, np.float16), n.output[0] + "_invk"))
            y = n.output[0]; n.output[0] = y + "_k"
            out.append(h.make_node("Mul", [y + "_k", y + "_invk"], [y]))
    del g.node[:]; g.node.extend(out)


def convnext(calib=None, desc=None):
    """calib: list of uint8 NHWC tensors (any side, the graph resizes; moved to the GPU per batch) to calibrate on; default the first NCAL COCO val2017 images."""
    from PIL import Image
    from transformers import AutoModel
    hf = AutoModel.from_pretrained(CONVNEXT, token=os.environ.get("HF_TOKEN")).eval().cuda().half()
    net = NHWC(hf).cuda().eval()
    files = sorted(Path("data/coco/val2017").glob("*.jpg"))[:NCAL]
    assert len(files) == NCAL, "calibration needs COCO val2017 in data/coco/val2017"
    imgs = torch.stack([torch.as_tensor(np.asarray(Image.open(f).convert("RGB").resize((224, 224), Image.BILINEAR))) for f in files]).cuda()
    with torch.no_grad():
        x = imgs[:32].half().permute(0, 3, 1, 2) / 255
        ref = hf(pixel_values=(x - torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1).half()) / torch.tensor(STD, device="cuda").view(1, 3, 1, 1).half()).pooler_output
        assert F.cosine_similarity(net(imgs[:32]).float(), ref.float()).min() > 0.999, "NHWC rewrite != HF model"
    calib = calib or [imgs]
    step = 97 * max(1, sum(map(len, calib)) // NCAL)  # samples per layer stay under the 2^24 torch.quantile limit, from every image
    mods = {"Conv": net.convs, "MatMul": net.linears}
    samples = {id(m): [] for v in mods.values() for m in v}
    hooks = [m.register_forward_pre_hook(lambda mod, a: samples[id(mod)].append(a[0].detach().abs().flatten().float()[::step].cpu())) for v in mods.values() for m in v]
    feats = []
    with torch.no_grad():
        for t in calib:
            for i in range(0, len(t), 32):
                feats.append(net(t[i:i + 32].cuda()).float().abs().flatten().cpu())
    for k in hooks:
        k.remove()
    pct = lambda xs: torch.quantile(torch.cat(xs)[:2**24], PCT / 100).item()
    amax = {k: pct(v) for k, v in samples.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "convnext-fp16.onnx"
    torch.onnx.export(net, (imgs[:2],), tmp, input_names=["image"], output_names=["feat"], dynamo=True, external_data=False,
                      dynamic_shapes={"image": {0: torch.export.Dim("N", max=BATCH), 1: torch.export.Dim("H", max=512), 2: torch.export.Dim("W", max=512)}})
    m = onnx.load(tmp); tmp.unlink()
    resize_nhwc(m.graph)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    f16 = lambda name, v: nh.from_array(np.asarray(v, np.float16), name)
    z8 = lambda name, n=None: nh.from_array(np.array(0, np.int8) if n is None else np.zeros(n, np.int8), name)
    out, seen = [], {"Conv": 0, "MatMul": 0}
    for node in g.node:
        if node.op_type in seen:  # ONNX Conv/MatMul nodes come in the order of the torch modules: matched by order, checked by weight value
            mod = mods[node.op_type][seen[node.op_type]]; k = f"{node.op_type}{seen[node.op_type]}"; seen[node.op_type] += 1
            w = nh.to_array(init[node.input[1]]).astype(np.float32)
            ref = mod.weight.detach().float().cpu().numpy()
            ref = ref.T if node.op_type == "MatMul" else ref  # Linear exports as MatMul with the weight [in, out]
            assert w.shape == ref.shape and np.abs(w - ref).max() < 1e-2, k
            if not (node.op_type == "Conv" and mod.groups > 1):  # depthwise convs stay FP16
                ax = 1 if node.op_type == "MatMul" else 0  # output channel
                ws = np.maximum(np.abs(w).max(axis=tuple(i for i in range(w.ndim) if i != ax)), 1e-8) / 127
                shp = [-1 if i == ax else 1 for i in range(w.ndim)]
                g.initializer.extend([nh.from_array(np.clip(np.round(w / ws.reshape(shp)), -127, 127).astype(np.int8), k + "_wq"),
                                      f16(k + "_ws", ws), f16(k + "_as", amax[id(mod)] / 127), z8(k + "_z"), z8(k + "_wz", len(ws))])
                out += [h.make_node("QuantizeLinear", [node.input[0], k + "_as", k + "_z"], [k + "_xq"]),
                        h.make_node("DequantizeLinear", [k + "_xq", k + "_as", k + "_z"], [k + "_x"]),
                        h.make_node("DequantizeLinear", [k + "_wq", k + "_ws", k + "_wz"], [k + "_w"], axis=ax)]
                node.input[0], node.input[1] = k + "_x", k + "_w"
        out.append(node)
    assert seen == {k: len(v) for k, v in mods.items()}, seen
    fs = pct(feats) / 127  # int8 embedding out, like the ResNet50 backbone
    g.initializer.extend([f16("feat_s", fs), z8("feat_z")])
    out.append(h.make_node("QuantizeLinear", ["feat", "feat_s", "feat_z"], ["feat_q"]))
    dim = g.output[0].type.tensor_type.shape.dim[1].dim_value
    del g.node[:]; g.node.extend(out)
    del g.output[:]; g.output.append(h.make_tensor_value_info("feat_q", onnx.TensorProto.INT8, ["N", dim]))
    used = {i for n in g.node for i in n.input}
    keep = [i for i in g.initializer if i.name in used]
    del g.initializer[:]; g.initializer.extend(keep)
    unsubnormal(g)
    return m, dict(feat_scale=fs, source=f"https://huggingface.co/{CONVNEXT}", license="DINOv3 License (LICENSE.md)",
                   calibration=f"{desc or f'COCO val2017, first {NCAL} images'}; activation amax = {PCT}th percentile of |x|")


def save(which):
    m, meta = globals()[which]()
    for k, v in meta.items():
        m.metadata_props.add(key=k, value=v if isinstance(v, str) else json.dumps(v))
    onnx.checker.check_model(m)
    path = OUT / f"{NAMES[which]}.onnx"
    onnx.save(m, path)
    print(path, f"{path.stat().st_size / 2**20:.1f} MB", meta)


CARD = """# {name}

INT8 Q/DQ ONNX backbone for [ModelTemplate IntClassifier](https://github.com/tuan2k33/ModelTemplate): a frozen feature extractor that TensorRT builds into one INT8 engine.

- Input `image`: uint8 [N, H, W, 3] RGB, any size; resize to 224x224 (bilinear) and ImageNet normalization run inside the graph.
- Output `feat_q`: int8 [N, {dim}]; the embedding is `feat_q * feat_scale` (`feat_scale` = {feat_scale:.6g}, also in the ONNX metadata).
- Source: {source}
{notes}
Build: `trtexec --onnx={name}.onnx --minShapes=image:1x8x8x3 --optShapes=image:64x64x64x3 --maxShapes=image:64x512x512x3 --saveEngine=model.engine` (TensorRT 11, strongly typed).

## License
{license_text}
"""
NOTES = {
    "resnet50": "- Conversion: the Model Zoo QLinear graph to Q/DQ (int8, zero-point 0, what TensorRT accepts); weights and biases keep their exact values. "
                "Every activation stays int8 (convs, residual adds, pools); `feat_q` is the global-average-pool tensor (>= 0).\n",
    "convnext": "- DINOv3 ConvNeXt-Tiny distilled from DINOv3 ViT-7B; `feat_q` is the pooled, LayerNorm-ed embedding (the HF `pooler_output`).\n"
                "- Quantization: int8 activation and weight on the stem/downsample convs and the 36 pointwise Linears (per-channel weights, per-tensor "
                "activations, 99.99th percentile on 256 COCO val2017 images); depthwise convs, LayerNorm, GELU and the residual stream stay FP16.\n"
                "- INT8 dequant scales kept out of the FP16 subnormal range (weight scale x 2^n, the 2^-n folded into the next op): exact, and without it "
                "TensorRT's INT8 GEMMs lose about 1 point on CIFAR-100 / Tiny ImageNet.\n"
                "- Layout: channels-last throughout (stem/downsample convs as strided slices + MatMul, same weights); the only transposes are around the "
                "18 depthwise convs.\n",
}


def push(which):
    from huggingface_hub import HfApi, hf_hub_download
    path = OUT / f"{NAMES[which]}.onnx"
    meta = {p.key: p.value for p in onnx.load(path, load_external_data=False).metadata_props}
    dim = onnx.load(path).graph.output[0].type.tensor_type.shape.dim[1].dim_value
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(HUB, private=False, exist_ok=True)
    if which == "convnext":
        lic = Path(hf_hub_download(CONVNEXT, "LICENSE.md", token=os.environ["HF_TOKEN"]))
        files, license_, license_text = {"LICENSE.md": lic}, "other", "DINOv3 License, see [LICENSE.md](LICENSE.md): this model is a derivative of the DINO Materials and is distributed under that Agreement."
    else:
        from dataset import fetch
        lic = fetch("https://www.apache.org/licenses/LICENSE-2.0.txt", OUT / "LICENSE-apache-2.0.txt")
        files, license_, license_text = {"LICENSE": lic}, "apache-2.0", "Apache-2.0 (see [LICENSE](LICENSE)), as the source model in the ONNX Model Zoo."
    card = OUT / f"{NAMES[which]}.README.md"
    card.write_text(CARD.format(name=NAMES[which], dim=dim, feat_scale=float(meta["feat_scale"]), source=meta["source"], notes=NOTES[which],
                                license=license_, license_text=license_text))
    for dst, src in {f"{NAMES[which]}.onnx": path, "README.md": card, **files}.items():
        api.upload_file(path_or_fileobj=str(src), path_in_repo=f"{NAMES[which]}/{dst}", repo_id=HUB)
    print(f"https://huggingface.co/{HUB}/tree/main/{NAMES[which]}")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["resnet50"]
    if cmd == "push":
        for w in rest or list(NAMES):
            push(w)
    else:
        save(cmd)
