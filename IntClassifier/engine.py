"""python engine.py weight/<run> [...]  : model.onnx of a train.py run -> model.engine (trtexec, 1-2 min), checked and timed.
The test set goes through the engine: its accuracy must match the run's test_acc (the int8 head computed exactly on the backbone engine's
embeddings; the final engine's own tactics can move a few embedding values by 1, hence a 1-point tolerance). metrics.json gets an `engine` entry."""
import json, sys, time
from pathlib import Path

import numpy as np
from cuda.bindings import runtime as cudart

from dataset import load
from model import BATCH, Engine, arr, compile_engine, infer


def bench(f, x, n):
    f(x)
    t0 = time.perf_counter()
    for _ in range(n):
        f(x)
    return (time.perf_counter() - t0) / n * 1e3


def gpu_mb():
    _, free, total = cudart.cudaMemGetInfo()
    return (total - free) / 2**20


for run in map(Path, sys.argv[1:] or ["weight/cifar10"]):
    m = json.loads((run / "metrics.json").read_text())
    t0 = time.perf_counter()
    compile_engine(run / "model.onnx", run / "model.engine")
    e = dict(compile_s=time.perf_counter() - t0)
    _, _, te, yt, _ = load(m["data"], m.get("per_class"), m.get("test_per_class"))
    cudart.cudaFree(0); g0 = gpu_mb()
    fe = Engine(run / "model.engine")
    x64 = np.stack([arr(i) for i in te[:BATCH]]); fe(x64); e["gpu_mb"] = gpu_mb() - g0
    e["batch1_ms"], e["batch64_ms"] = bench(fe, x64[:1], 200), bench(fe, x64, 30)
    t0 = time.perf_counter()
    e["test_acc"] = float(np.mean(np.concatenate([infer(fe, te[s:s + BATCH])[0] for s in range(0, len(te), BATCH)]) == yt))
    e["img_per_s"] = len(te) / (time.perf_counter() - t0)
    assert abs(e["test_acc"] - m["test_acc"]) < 0.01, f"{run}: engine accuracy {e['test_acc']} != int8 head {m['test_acc']}"
    m["engine"] = {k: round(v, 4) for k, v in e.items()}
    (run / "metrics.json").write_text(json.dumps(m, indent=1))
    print(run, m["engine"])
