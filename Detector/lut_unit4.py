"""TensorRT: fp16 input (no Q/DQ) -> Conv with per-channel DQ weights. Correct? python lut_unit4.py [Cout Cin]"""
import sys, numpy as np, onnx, tensorrt as trt, torch
import torch.nn.functional as F
from onnx import TensorProto as T, helper as h, numpy_helper as nh
co, ci = (int(v) for v in (sys.argv[1:3] or [128, 64]))
rng = np.random.default_rng(0)
wq = rng.integers(-127, 128, (co, ci, 3, 3)).astype(np.int8)
ws = (rng.uniform(0.5, 2.0, co) * 1e-2).astype(np.float16)
x = rng.standard_normal((1, ci, 40, 40)).astype(np.float16)
nodes = [h.make_node("DequantizeLinear", ["wq", "ws", "wz"], ["w"], axis=0), h.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2])]
g = h.make_graph(nodes, "t", [h.make_tensor_value_info("x", T.FLOAT16, list(x.shape))], [h.make_tensor_value_info("y", T.FLOAT16, None)],
                 [nh.from_array(wq, "wq"), nh.from_array(ws, "ws"), nh.from_array(np.zeros(co, np.int8), "wz")])
m = h.make_model(g, opset_imports=[h.make_opsetid("", 19)]); m.ir_version = 9
lg = trt.Logger(trt.Logger.WARNING); b = trt.Builder(lg); net = b.create_network(0); p = trt.OnnxParser(net, lg); assert p.parse(m.SerializeToString())
eng = trt.Runtime(lg).deserialize_cuda_engine(b.build_serialized_network(net, b.create_builder_config())); ctx = eng.create_execution_context()
xi = torch.from_numpy(x).cuda(); yo = torch.empty(tuple(eng.get_tensor_shape("y")), dtype=torch.float16, device="cuda")
ctx.set_tensor_address("x", xi.data_ptr()); ctx.set_tensor_address("y", yo.data_ptr())
s = torch.cuda.Stream(); torch.cuda.synchronize(); ctx.execute_async_v3(s.cuda_stream); s.synchronize()
w = torch.from_numpy(wq.astype(np.float32) * ws.astype(np.float32)[:, None, None, None])
ref = F.conv2d(torch.from_numpy(x).float(), w, stride=2, padding=1)
for name, wt in (("axis0 (correct)", w), ("ws along axis1", torch.from_numpy(wq.astype(np.float32) * np.resize(ws.astype(np.float32), ci)[None, :, None, None]))):
    r = F.conv2d(torch.from_numpy(x).float(), wt, stride=2, padding=1)
    print(f"{co}x{ci}: trt vs {name}: rel err {((yo.float().cpu() - r).abs().mean() / r.abs().mean()).item():.4f}")
yt = yo.float().cpu()
print("trt nan", int(torch.isnan(yt).sum()), "inf", int(torch.isinf(yt).sum()), "of", yt.numel(), "| absmax trt", yt[torch.isfinite(yt)].abs().max().item(), "ref", ref.abs().max().item())
f = torch.isfinite(yt)
print("finite-only rel err", ((yt[f] - ref[f]).abs().mean() / ref[f].abs().mean()).item())
