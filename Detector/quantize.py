"""Optional INT8 build (the default engine is FP16: on yolo26n INT8 is slower and loses about 1.6 mAP, see README).
    python quantize.py [--model cache/yolo26n.pt] [--images DIR] [--calib 64] [--silu float|lut] [--name int8]   -> weight/<name>/model.{onnx,engine}
Every Conv followed by SiLU gets int8 input (per-tensor) and int8 weights (per output channel) as Q/DQ; heads, attention and convs without SiLU stay FP16.
`float`: SiLU in FP16 between the int8 convs. `lut`: SiLU as a 256-entry int8 table (Q -> Gather -> DQ), activations stay int8 (exact, but slower on TensorRT 11.5).
Calibration: p99.99 of |x| per layer over --calib images (--images, or the last ones of COCO val2017; eval.py uses the first ones), hooks on the fused torch model."""
import argparse, json, zipfile
from pathlib import Path

import cv2, numpy as np, onnx, torch
from onnx import TensorProto as T, helper as h, numpy_helper as nh
from ultralytics import YOLO

from model import CACHE, SIDE, compile_engine, with_front_end

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(CACHE / "yolo26n.pt"), help="Ultralytics weights (e.g. weight/<name>/weights/best.pt)")
ap.add_argument("--images", help="folder of calibration images; default: data/coco/val2017.zip")
ap.add_argument("--calib", type=int, default=64)
ap.add_argument("--silu", default="float", choices=["float", "lut"])
ap.add_argument("--name", default="int8")
a = ap.parse_args()


def load_imgs(n):
    if a.images:
        raw = (p.read_bytes() for p in sorted(Path(a.images).glob("*"))[-n:] if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    else:
        z = zipfile.ZipFile("data/coco/val2017.zip")
        raw = (z.read(nm) for nm in sorted(x for x in z.namelist() if x.endswith(".jpg"))[-n:])
    for b in raw:
        im = cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB)
        r = min(SIDE / im.shape[0], SIDE / im.shape[1])
        im = cv2.resize(im, (round(im.shape[1] * r), round(im.shape[0] * r)))
        c = np.full((SIDE, SIDE, 3), 114, np.uint8)
        t, l = (SIDE - im.shape[0]) // 2, (SIDE - im.shape[1]) // 2
        c[t:t + im.shape[0], l:l + im.shape[1]] = im
        yield torch.from_numpy(c).permute(2, 0, 1)[None].cuda().half() / 255


def calibrate(yolo):
    """amax per Conv module path: (input, conv output, act output)."""
    net = yolo.model.fuse().half().cuda().eval()
    st = {}

    def rec(key):
        def f(mod, inp, out):
            x = (inp[0] if key[1] == "in" else out).detach().abs().flatten()[::16].float()
            st.setdefault(key, []).append(torch.kthvalue(x, int(0.9999 * len(x))).values.item())
        return f
    for p, m in net.named_modules():
        if hasattr(m, "conv") and hasattr(m, "act") and isinstance(m.act, torch.nn.SiLU):
            m.conv.register_forward_hook(rec((p, "in"))); m.conv.register_forward_hook(rec((p, "pre")))
            m.register_forward_hook(rec((p, "post")))  # the Conv wrapper, not m.act: the SiLU object is shared by all Convs
    with torch.no_grad():
        for x in load_imgs(a.calib):
            net(x)
    return {k: float(np.mean(v)) for k, v in st.items()}


def silu(v):
    return v / (1 + np.exp(-v))


yolo = YOLO(a.model)
names = [yolo.names[i] for i in range(len(yolo.names))]
onnx_fp16 = yolo.export(format="onnx", imgsz=SIDE, nms=False, half=True, device=0)  # same export as train.py
amax = calibrate(YOLO(a.model))
m = onnx.load(onnx_fp16)
g = m.graph
inits = {i.name: i for i in g.initializer}
prod = {o: n for n in g.node for o in n.output}
sig_of_mul = {}  # SiLU Mul output -> (Sigmoid node, Mul node)
for n in g.node:
    if n.op_type == "Mul" and n.input[1] in prod and prod[n.input[1]].op_type == "Sigmoid" and prod[n.input[1]].input[0] == n.input[0]:
        sig_of_mul[n.output[0]] = (prod[n.input[1]], n)
conv_of = {c.output[0]: c for c in g.node if c.op_type == "Conv"}
qconv = {}  # conv name -> (act Mul output name)
for mo, (s, mul) in sig_of_mul.items():
    if mul.input[0] in conv_of:
        qconv[conv_of[mul.input[0]].name] = mo
mod = lambda c: c.input[1].removesuffix(".conv.weight")  # weight initializers keep the torch parameter names (node names do not)
qconv = {k: v for k, v in qconv.items() if all((mod(conv_of[sig_of_mul[v][1].input[0]]), t) in amax for t in ('in', 'pre', 'post'))}
print(len(qconv), 'of', len(sig_of_mul), 'SiLU convs quantized')
new_inits, nodes, lut_out, scale_of, qdq = [], [], set(), {}, {}
zp = nh.from_array(np.int8(0), "zp0")
new_inits.append(zp)


def qd(x, amax_x, tag, s=None):
    """x (fp16) -> Q -> DQ (scale amax/127, or s). `lut`: one pair per conv, as two LUT-mode convs on one shared DQ (C3k's cv1 and cv2) are not
    fused into int8 by TensorRT 11.5 and fall into the FP16-conv + DQ-weights case that returns garbage. `float`: one pair per tensor, which
    TensorRT fuses fine and is 0.4 ms faster on yolo26n."""
    if a.silu == "float" and x in qdq:
        return qdq[x]
    s = s if s is not None else max(amax_x, 1e-5) / 127
    sn = f"{tag}_s"
    new_inits.append(nh.from_array(np.float16(s), sn))
    nodes.append(h.make_node("QuantizeLinear", [x, sn, "zp0"], [tag + "_q"]))
    nodes.append(h.make_node("DequantizeLinear", [tag + "_q", sn, "zp0"], [tag + "_dq"]))
    scale_of[tag + "_dq"] = float(np.float16(s))
    qdq[x] = tag + "_dq"
    return tag + "_dq"


drop = {sg.name for sg, mul in sig_of_mul.values() if a.silu == "lut" and mul.input[0] in conv_of and conv_of[mul.input[0]].name in qconv}
for n in g.node:
    if n.op_type == "Conv" and n.name in qconv:
        k = mod(n)
        x = n.input[0]
        # a LUT output gets its own Q/DQ at the same scale (exact: Q(DQ(zq)) = zq). Without it, when the LUT output also feeds a Concat, TensorRT 11.5
        # runs this conv in FP16 with the per-channel DQ weights, and that case returns garbage (NaN / 0; repro: lut_unit4.py)
        xq = qd(x, None, f"{k}_in", scale_of[x]) if x in lut_out else qd(x, amax[(k, "in")], f"{k}_in")
        w = nh.to_array(inits[n.input[1]]).astype(np.float32)
        ws = np.maximum(np.abs(w).reshape(len(w), -1).max(1), 1e-8) / 127
        wq = np.clip(np.round(w / ws.reshape(-1, *[1] * (w.ndim - 1))), -127, 127).astype(np.int8)
        new_inits += [nh.from_array(wq, k + "_wq"), nh.from_array(ws.astype(np.float16), k + "_ws"), nh.from_array(np.zeros(len(ws), np.int8), k + "_wz")]
        nodes += [h.make_node("DequantizeLinear", [k + "_wq", k + "_ws", k + "_wz"], [k + "_w"], axis=0)]
        c = h.make_node("Conv", [xq, k + "_w"] + list(n.input[2:]), list(n.output), name=n.name)
        c.attribute.extend(n.attribute)
        nodes.append(c)
        continue
    if n.name in drop:
        continue
    if a.silu == "lut" and n.output[0] in sig_of_mul and sig_of_mul[n.output[0]][1] is n and n.input[0] in conv_of and conv_of[n.input[0]].name in qconv:
        k = mod(conv_of[n.input[0]])
        sy, sz = max(amax[(k, "pre")], 1e-5) / 127, max(amax[(k, "post")], 1e-5) / 127
        v = (np.arange(256) - 128) * sy
        new_inits += [nh.from_array(np.float16(sy), k + "_sy"), nh.from_array(np.float16(sz), k + "_sz"),
                      nh.from_array(np.clip(np.round(silu(v) / sz), -127, 127).astype(np.int32), k + "_lut"), nh.from_array(np.int32(128), k + "_off")]
        nodes += [h.make_node("QuantizeLinear", [n.input[0], k + "_sy", "zp0"], [k + "_yq"]),
                  h.make_node("Cast", [k + "_yq"], [k + "_yi"], to=T.INT32), h.make_node("Add", [k + "_yi", k + "_off"], [k + "_idx"]),
                  h.make_node("Gather", [k + "_lut", k + "_idx"], [k + "_zi"], axis=0), h.make_node("Cast", [k + "_zi"], [k + "_zq"], to=T.INT8),
                  h.make_node("DequantizeLinear", [k + "_zq", k + "_sz", "zp0"], [n.output[0]])]
        lut_out.add(n.output[0]); scale_of[n.output[0]] = float(np.float16(sz))
        continue
    nodes.append(n)
del g.node[:]
g.node.extend(nodes)
g.initializer.extend(new_inits)
out = Path("weight") / a.name
out.mkdir(parents=True, exist_ok=True)
onnx.save(m, out / "q.onnx")
onnx.save(with_front_end(out / "q.onnx", names), out / "model.onnx")
compile_engine(out / "model.onnx", out / "model.engine")
print("engine", out / "model.engine")
