"""Which quantized convs did TensorRT not fuse into int8 (weights dequantized by a Scale layer, FP16 conv)? python lut_layers.py q.onnx"""
import json, sys, onnx, tensorrt as trt
m = onnx.load(sys.argv[1])
lg = trt.Logger(trt.Logger.ERROR); b = trt.Builder(lg); net = b.create_network(0); p = trt.OnnxParser(net, lg); assert p.parse(m.SerializeToString())
cfg = b.create_builder_config(); cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
eng = trt.Runtime(lg).deserialize_cuda_engine(b.build_serialized_network(net, cfg))
L = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))["Layers"]
sc = [l["Name"] for l in L if l.get("LayerType") == "Scale"]
i8 = [l["Name"] for l in L if l.get("LayerType") == "CaskConvolution" and "_i8" in str(l.get("TacticName", ""))]
print("unfused weight DQ (Scale):", len(sc), sc[:12]); print("int8 convs:", len(i8))
for l in L:
    if l.get("LayerType") == "Scale":
        for c in L:
            if any(i["Name"] == l["Outputs"][0]["Name"] for i in c.get("Inputs", [])):
                print("  ", c["Name"][:70], "<-", [i["Name"][:50] for i in c["Inputs"]], str(c.get("TacticName", ""))[:50])
