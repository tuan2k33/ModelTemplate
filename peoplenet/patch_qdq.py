import os, struct, sys
import numpy as np, onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T

src, cache, dst = sys.argv[1:4]
MODE = sys.argv[4] if len(sys.argv) > 4 else "both"
MULT = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
m = onnx.load(src)
g = m.graph
scale = {}
for l in open(cache).read().splitlines()[1:]:
    k, v = l.rsplit(": ", 1)
    scale[k] = struct.unpack(">f", bytes.fromhex(v))[0]
init = {i.name: i for i in g.initializer}
new_nodes, new_init, done = [], [], {}
out = []
for o in m.opset_import:
    if o.domain in ('', 'ai.onnx') and o.version < 13: o.version = 13

def qdq(name, s, axis=None, uid=""):
    name0, name = name, name + uid
    if name in done and not uid:
        return done[name]
    sn = name + "_qs"
    zp = name + "_qz"
    sv = np.asarray(s, np.float32)
    new_init.append(nh.from_array(sv, sn))
    new_init.append(nh.from_array(np.zeros(sv.shape, np.int8), zp))
    kw = {} if axis is None else {"axis": axis}
    new_nodes.append(h.make_node("QuantizeLinear", [name0, sn, zp], [name + "_q"], **kw))
    new_nodes.append(h.make_node("DequantizeLinear", [name + "_q", sn, zp], [name + "_dq"], **kw))
    done[name] = name + "_dq"
    return done[name]

miss = 0
LIMIT = int(os.environ.get("LIMIT", 999))
seen = 0
for n in g.node:
    new_nodes.clear()
    if n.op_type == "Conv":
        seen += 1
        if seen > LIMIT or (os.environ.get("ONLY") and str(seen) not in os.environ["ONLY"].split(",")):
            out.append(n); continue
        x, w = n.input[0], n.input[1]
        skip = os.environ.get("SKIP", "").split(",")
        if any(s and (s in n.name or s in x or s in n.output[0]) for s in skip):
            out.append(n); continue
        key = x if x in scale else x.removesuffix(":0")
        if key in scale and MODE != "w":
            n.input[0] = qdq(x, scale[key] * MULT * (1 + 1e-4 * seen), uid=f"_c{seen}")
        else:
            miss += 1
        if w in init and MODE != "a":
            wa = nh.to_array(init[w])
            s = np.abs(wa).reshape(wa.shape[0], -1).max(1) / 127.0
            n.input[1] = qdq(w, np.maximum(s, 1e-8), axis=0)
    out += new_nodes + [n]
g.initializer.extend(new_init)
del g.node[:]
g.node.extend(out)
onnx.checker.check_model(m)
onnx.save(m, dst)
print("convs without activation scale:", miss)
