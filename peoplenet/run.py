"""PeopleNet (DetectNet_v2) TensorRT engine on a video: boxes drawn, H264 out. usage: run.py engine in.mp4 out.mp4"""
import subprocess, sys, time
from pathlib import Path

import cv2, numpy as np, tensorrt as trt, torch
from torchvision.ops import batched_nms

W, H, STRIDE, NORM, THR = 960, 544, 16, 35.0, 0.4
NAMES = ["person", "bag", "face"]
COL = [(0, 255, 0), (0, 165, 255), (255, 0, 0)]


class Net:
    def __init__(self, path):
        e = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(Path(path).read_bytes())
        self.ctx = e.create_execution_context()
        io = [e.get_tensor_name(i) for i in range(e.num_io_tensors)]
        self.inp = next(n for n in io if e.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.x = torch.empty(1, 3, H, W, device="cuda")
        self.out = {n: torch.empty(tuple(e.get_tensor_shape(n)), device="cuda") for n in io if n != self.inp}
        self.ctx.set_tensor_address(self.inp, self.x.data_ptr())
        for n, t in self.out.items():
            self.ctx.set_tensor_address(n, t.data_ptr())

    def __call__(self, rgb):
        self.x.copy_(torch.from_numpy(rgb).cuda().permute(2, 0, 1)[None] / 255.0)
        assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        cov = next(t for n, t in self.out.items() if "cov" in n)[0]
        box = next(t for n, t in self.out.items() if "bbox" in n)[0]
        gy, gx = torch.meshgrid(torch.arange(cov.shape[1], device="cuda"), torch.arange(cov.shape[2], device="cuda"), indexing="ij")
        cx, cy = (gx * STRIDE + 0.5) / NORM, (gy * STRIDE + 0.5) / NORM
        b = box.view(3, 4, *cov.shape[1:])
        xyxy = torch.stack([-(b[:, 0] - cx) * NORM, -(b[:, 1] - cy) * NORM, (b[:, 2] + cx) * NORM, (b[:, 3] + cy) * NORM], -1)
        k = cov > THR
        cls = torch.nonzero(k)[:, 0]
        bx, sc = xyxy[k], cov[k]
        keep = batched_nms(bx, sc, cls, 0.4)
        return bx[keep].cpu().numpy(), sc[keep].cpu().numpy(), cls[keep].cpu().numpy()


if __name__ == "__main__":
    eng, src, dst = sys.argv[1:4]
    net, cap = Net(eng), cv2.VideoCapture(src)
    fps, w0, h0 = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w0}x{h0}", "-r", str(fps), "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", dst], stdin=subprocess.PIPE)
    counts, t = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(cv2.resize(fr, (W, H)), cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        bx, sc, cl = net(rgb)
        t.append(time.perf_counter() - t0)
        counts.append(int((cl == 0).sum()))
        for (x1, y1, x2, y2), s, c in zip(bx * [w0 / W, h0 / H, w0 / W, h0 / H], sc, cl):
            cv2.rectangle(fr, (int(x1), int(y1)), (int(x2), int(y2)), COL[c], 2)
            cv2.putText(fr, f"{NAMES[c]} {s:.2f}", (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL[c], 1)
        ff.stdin.write(fr.tobytes())
    ff.stdin.close(); ff.wait()
    print(f"{eng}: {len(t)} frames, {1000 * np.median(t[5:]):.1f} ms/frame (incl. decode+NMS), persons/frame mean {np.mean(counts):.2f}")
    np.save(Path(dst).with_suffix(".npy"), np.array(counts))
