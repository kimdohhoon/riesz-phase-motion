"""Decisive sanity: can a temporal CNN over monogenic maps recover MOTION
DIRECTION (Swiping L/R/U/D) from synthetic gratings?

L vs R use the SAME vertical stripes moving in OPPOSITE directions -> structure
orientation is identical, so they are separable ONLY by temporal order. If the
temporal CNN classifies 4 directions well above chance (0.25), then the
"global-pooling killed direction" diagnosis is right and the temporal-CNN fix
works. If it sits at chance, the phase-map representation itself is the problem.

Run:  python verify_motion_cnn.py
"""
from __future__ import annotations
import sys
import math
import torch
import torch.nn as nn

from monogenic import MonogenicExtractor
from motion_features import monogenic_maps
from motion_cnn import MotionCNNClassifier

S, OUT, T = 48, 24, 8
DIRS = ["L", "R", "U", "D"]


def grating(lam, angle, shift, contrast, size=S):
    yy, xx = torch.meshgrid(torch.arange(size).float(),
                            torch.arange(size).float(), indexing="ij")
    coord = (xx - shift * math.cos(angle)) * math.cos(angle) + \
            (yy - shift * math.sin(angle)) * math.sin(angle)
    return (contrast * torch.sin(2 * math.pi * coord / lam)).view(1, 1, size, size)


def make_clip(direction, g):
    angle = 0.0 if direction in ("L", "R") else math.pi / 2   # L/R vertical, U/D horizontal
    sign = 1.0 if direction in ("R", "D") else -1.0
    lam = 10 + torch.rand(1, generator=g).item() * 8
    contrast = 0.6 + torch.rand(1, generator=g).item() * 0.4
    phase0 = torch.rand(1, generator=g).item() * lam
    speed = sign * (0.8 + torch.rand(1, generator=g).item() * 0.6)
    frames = [grating(lam, angle, phase0 + t * speed, contrast) for t in range(T)]
    clip = torch.cat(frames, 0)                                # (T,1,S,S)
    clip = clip + 0.05 * torch.randn(clip.shape, generator=g)
    return clip


def build(n_per, ext, g):
    X, y = [], []
    for ci, d in enumerate(DIRS):
        for _ in range(n_per):
            maps = monogenic_maps(ext, make_clip(d, g), out_size=OUT)   # (T,4,OUT,OUT)
            X.append(maps.permute(1, 0, 2, 3))                          # (4,T,OUT,OUT)
            y.append(ci)
    return torch.stack(X), torch.tensor(y)


def main():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    ext = MonogenicExtractor(img_size=S, n_scales=5, trainable=False)
    print("building synthetic L/R/U/D clips...", flush=True)
    Xtr, ytr = build(80, ext, g)
    Xva, yva = build(20, ext, g)
    print(f"train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}", flush=True)

    model = MotionCNNClassifier(n_classes=4, in_ch=4, dim=64)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    n = Xtr.shape[0]
    for ep in range(40):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(Xtr[idx]), ytr[idx])
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xva).argmax(1)
        acc = (pred == yva).float().mean().item()
        # L-vs-R confusion specifically (the hard pair)
        lr_mask = (yva == 0) | (yva == 1)
        lr_acc = (pred[lr_mask] == yva[lr_mask]).float().mean().item()

    print("=" * 56)
    print(f"  chance (4-way)        : 0.250")
    print(f"  temporal-CNN val acc  : {acc:.3f}")
    print(f"  L-vs-R subset acc     : {lr_acc:.3f}  (the orientation-blind pair)")
    print("=" * 56)
    ok = acc > 0.5 and lr_acc > 0.6
    print(f"[{'PASS' if ok else 'FAIL'}] temporal CNN recovers motion direction "
          f"from phase maps" if ok else
          f"[FAIL] direction NOT recovered -- representation issue")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
