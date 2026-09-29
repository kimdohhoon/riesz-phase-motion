"""ARID regime-boundary analysis: split classes by MOTION MAGNITUDE.

ARID combines our strong regime (natural darkness) with our weak regime (large
whole-body motion). Aggregate ARID is therefore a wash (flow ~= phase). This
script DISENTANGLES the two by grouping the 11 ARID classes into small/local-motion
vs large/whole-body-motion and reporting phase vs flow per group, from CACHED ARID
features (no re-extraction). Hypothesis: phase wins on small-motion-dark classes;
flow wins on large-motion classes -> proves the regime claim on NATURAL dark data.

Run:  CUDA_VISIBLE_DEVICES=0 python arid_motion_split.py
"""
from __future__ import annotations
import os
import json
import numpy as np
import torch
import torch.nn as nn

from motion_cnn import TemporalMotionCNN

DEVICE = "cuda"
CACHE = "cache"
SEEDS = [0, 1, 2]

# ARID mapping_table order -> class id
ARID_CLASSES = ["Drink", "Jump", "Pick", "Pour", "Push", "Run", "Sit", "Stand",
                "Turn", "Walk", "Wave"]
SMALL = {"Drink", "Pick", "Pour", "Wave", "Sit", "Stand"}     # local / minimal motion
LARGE = {"Jump", "Run", "Turn", "Walk", "Push"}               # whole-body large motion


def load(motion, cond):
    b = torch.load(os.path.join(CACHE, f"arid_{motion}_{cond}.pt"), map_location="cpu")
    return b["app"].float(), b["maps"].float(), b["ys"]


class FusedModel(nn.Module):
    def __init__(self, app_dim, n_classes, in_ch, motion_dim=64, hidden=256):
        super().__init__()
        self.app_proj = nn.Sequential(nn.Linear(app_dim, 128), nn.LayerNorm(128))
        self.motion = TemporalMotionCNN(in_ch=in_ch, dim=motion_dim)
        self.motion_norm = nn.LayerNorm(motion_dim)
        self.head = nn.Sequential(nn.Linear(128 + motion_dim, hidden), nn.ReLU(True),
                                  nn.Dropout(0.3), nn.Linear(hidden, n_classes))

    def forward(self, app, maps):
        a = self.app_proj(app)
        m = self.motion_norm(self.motion(maps.permute(0, 2, 1, 3, 4)))
        return self.head(torch.cat([a, m], -1))


def train(tr, in_ch, nc, app_dim, seed, epochs=60, bs=32):
    torch.manual_seed(seed)
    A, M, y = tr
    model = FusedModel(app_dim, nc, in_ch).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n = A.shape[0]
    for _ in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = nn.functional.cross_entropy(
                model(A[idx].to(DEVICE), M[idx].to(DEVICE)), y[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def per_class_acc(model, ev, nc):
    A, M, y = ev
    model.eval()
    correct = np.zeros(nc); total = np.zeros(nc)
    with torch.no_grad():
        for i in range(0, len(y), 256):
            pred = model(A[i:i + 256].to(DEVICE), M[i:i + 256].to(DEVICE)).argmax(1).cpu()
            yb = y[i:i + 256]
            for c in range(nc):
                m = yb == c
                total[c] += m.sum().item()
                correct[c] += (pred[m] == c).sum().item()
    return correct / np.maximum(total, 1)


def group_acc(pc, group):
    ids = [i for i, name in enumerate(ARID_CLASSES) if name in group]
    return float(np.mean([pc[i] for i in ids]))


def main():
    res = {}
    for motion in ["phase", "flow"]:
        tr = load(motion, "train_clean")
        ev = load(motion, "val_clean")
        nc = int(tr[2].max()) + 1
        app_dim = tr[0].shape[1]
        in_ch = tr[1].shape[2]
        small, large, overall = [], [], []
        pcs = []
        for s in SEEDS:
            model = train(tr, in_ch, nc, app_dim, s)
            pc = per_class_acc(model, ev, nc)
            pcs.append(pc)
            small.append(group_acc(pc, SMALL))
            large.append(group_acc(pc, LARGE))
            overall.append(float(np.mean(pc)))
        res[motion] = {
            "small_motion": [float(np.mean(small)), float(np.std(small))],
            "large_motion": [float(np.mean(large)), float(np.std(large))],
            "overall": [float(np.mean(overall)), float(np.std(overall))],
            "per_class": np.mean(pcs, 0).tolist()}
        print(f"[split] {motion:6s} small {res[motion]['small_motion'][0]:.3f}  "
              f"large {res[motion]['large_motion'][0]:.3f}  "
              f"overall {res[motion]['overall'][0]:.3f}", flush=True)

    with open("runs/arid_motion_split.json", "w") as f:
        json.dump({"classes": ARID_CLASSES, "small": list(SMALL), "large": list(LARGE),
                   "results": res}, f, indent=2)
    print("\n=== ARID regime split (natural dark, fused, 3-seed) ===")
    print(f"{'group':16s}{'phase':>16}{'flow':>16}")
    for g in ["small_motion", "large_motion", "overall"]:
        p = res["phase"][g]; fl = res["flow"][g]
        win = "phase" if p[0] > fl[0] else "flow"
        print(f"{g:16s}{p[0]:>8.3f}±{p[1]:.2f}{fl[0]:>8.3f}±{fl[1]:.2f}   <- {win}")
    print("\nper-class (phase | flow):")
    for i, name in enumerate(ARID_CLASSES):
        tag = "small" if name in SMALL else "large"
        print(f"  {name:7s}({tag}) {res['phase']['per_class'][i]:.3f} | "
              f"{res['flow']['per_class'][i]:.3f}")
    print("[split] saved -> runs/arid_motion_split.json")


if __name__ == "__main__":
    main()
