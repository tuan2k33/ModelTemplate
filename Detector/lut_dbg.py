"""Debug quantize.py --silu lut: the same weight/<name>/q.onnx through onnxruntime (CPU, reference semantics) and TensorRT, every LUT layer's output marked as a graph output,
compared in quantization steps. python lut_dbg.py weight/lut/q.onnx"""
import sys
import cv2, numpy as np, onnx, onnxruntime as ort, tensorrt as trt, torch
from onnx import TensorProto as T, helper as h, numpy_helper as nh

m = onnx.load(sys.argv[1])
g = m.graph
init = {i.name: i for i in g.initializer}
outs = []  # (layer, tensor, scale)
for n in g.node:
    if n.op_type == "DequantizeLinear" and n.input[0].endswith("_zq"):
        k = n.input[0][:-3]
        outs.append((k, n.output[0], float(nh.to_array(init[k + "_sz"]))))
        outs.append((k + " pre-q", k + "_yq", float(nh.to_array(init[k + "_sy"]))))
for _, t, _ in outs:
    g.output.append(h.make_tensor_value_info(t, T.INT8 if t.endswith("_yq") else T.FLOAT16, None))
m.opset_import[0].version = 19
blob = m.SerializeToString()

im = cv2.cvtColor(cv2.imread("/home/tuannq/.dev/data/bus.jpg"), cv2.COLOR_BGR2RGB)
r = 640 / max(im.shape[:2]); im = cv2.resize(im, (round(im.shape[1] * r), round(im.shape[0] * r)))
c = np.full((640, 640, 3), 114, np.uint8); c[:im.shape[0], :im.shape[1]] = im
x = (c.transpose(2, 0, 1)[None] / 255).astype(np.float16)

so = ort.SessionOptions(); so.log_severity_level = 3; so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(blob, so, providers=["CPUExecutionProvider"]).run([t for _, t, _ in outs] + [g.output[0].name], {"images": x})

lg = trt.Logger(trt.Logger.WARNING)
b = trt.Builder(lg); net = b.create_network(0); p = trt.OnnxParser(net, lg)
assert p.parse(blob), [p.get_error(i) for i in range(p.num_errors)]
eng = trt.Runtime(lg).deserialize_cuda_engine(b.build_serialized_network(net, b.create_builder_config()))
ctx = eng.create_execution_context()
TD = {trt.float16: torch.float16, trt.float32: torch.float32, trt.int8: torch.int8, trt.int32: torch.int32}
bufs = {}
for i in range(eng.num_io_tensors):
    nm = eng.get_tensor_name(i)
    bufs[nm] = torch.from_numpy(np.ascontiguousarray(x)).cuda() if nm == "images" else torch.empty(tuple(eng.get_tensor_shape(nm)), dtype=TD[eng.get_tensor_dtype(nm)], device="cuda")
    ctx.set_tensor_address(nm, bufs[nm].data_ptr())
s = torch.cuda.Stream()
ctx.execute_async_v3(s.cuda_stream); s.synchronize()
for (k, t, sc), rv in zip(outs, ref):
    tv = bufs[t].float().cpu().numpy()
    d = np.abs(tv - rv.astype(np.float32)) / (1 if t.endswith("_yq") else sc)
    print(f"{k:28s} max diff {d.max():8.1f} steps  mean {d.mean():.3f}  frac>1 {np.mean(d > 1.01):.4f}")
o = bufs[g.output[0].name].float().cpu().numpy()[0]
print("final conf trt", np.sort(o[:, 4])[::-1][:4].round(3), "ort", np.sort(ref[-1][0][:, 4].astype(np.float32))[::-1][:4].round(3))
