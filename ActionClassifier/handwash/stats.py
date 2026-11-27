"""Share of frames with 0/1/2 hands found, per WHO step. python stats.py cache/kpt-video-0.5 [...]"""
import sys
from pathlib import Path

import numpy as np

for d in sys.argv[1:]:
    fs = sorted(Path(d).glob("*.npz"))
    P = np.concatenate([np.load(f)["present"] for f in fs]).sum(1)
    L = np.concatenate([np.load(f)["label"] for f in fs])
    print(f"{d} ({len(fs)} videos, {len(P)} frames) 0/1/2 hands %:")
    print("  " + "  ".join(f"{s}: " + "/".join(f"{(P[L == s] == k).mean() * 100:.0f}" for k in range(3)) for s in range(8) if (L == s).any()))
