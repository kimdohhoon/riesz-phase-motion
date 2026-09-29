"""Appearance-only temporal baseline vs. the fused model (paper Sec. 4.7, Supp. Tab. 8).

Three appearance-only heads over the per-frame CLIP features of extract_temporal_app.py:
  mid      : middle frame only (reproduces the appearance-only row of paper Tab. 2)
  meanpool : mean over the T frame embeddings
  tconv    : Conv1d 768->256->128 (kernel 3, BN, ReLU), temporal average pooling, LayerNorm
All use the head and training recipe of the main model (60 epochs, batch 32, AdamW 1e-3,
weight decay 1e-4) and 3 seeds. The fused model on the same subset is train_fused_12k.py.

  python train_temporal_app.py   -> runs/temporal_app_baseline.json
"""
from __future__ import annotations
import json, os
import numpy as np
import torch
import torch.nn as nn

CACHE, DEV, SEEDS = "cache", "cuda", [0, 1, 2]
EPOCHS, BS, PROJ, HID, DROP = 60, 32, 128, 256, 0.3
CONDS = ["clean", "ll5_none", "ll5_gamma", "ll5r_none"]


class AppOnly(nn.Module):
    def __init__(self, dim, n_classes, mode):
        super().__init__()
        self.mode = mode
        if mode == "tconv":
            self.tc = nn.Sequential(
                nn.Conv1d(dim, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(True),
                nn.Conv1d(256, PROJ, 3, padding=1), nn.BatchNorm1d(PROJ), nn.ReLU(True),
                nn.AdaptiveAvgPool1d(1))
            self.norm = nn.LayerNorm(PROJ)
        else:
            self.proj = nn.Sequential(nn.Linear(dim, PROJ), nn.LayerNorm(PROJ))
        self.head = nn.Sequential(nn.Linear(PROJ, HID), nn.ReLU(True),
                                  nn.Dropout(DROP), nn.Linear(HID, n_classes))

    def forward(self, a):                              # a: (B,T,dim)
        if self.mode == "mid":
            v = self.proj(a[:, a.shape[1] // 2])
        elif self.mode == "meanpool":
            v = self.proj(a.mean(1))
        else:
            v = self.norm(self.tc(a.transpose(1, 2)).squeeze(-1))
        return self.head(v)


def load(split, cond):
    b = torch.load(os.path.join(CACHE, f"jester_appT_{split}_{cond}.pt"),
                   map_location="cpu", weights_only=False)
    return b["app"].float(), b["ys"]


def run(mode, Atr, ytr, vals, nc, seed):
    torch.manual_seed(seed)
    m = AppOnly(Atr.shape[-1], nc, mode).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(EPOCHS):
        m.train(); perm = torch.randperm(Atr.shape[0])
        for i in range(0, len(perm), BS):
            idx = perm[i:i + BS]; opt.zero_grad()
            nn.functional.cross_entropy(m(Atr[idx].to(DEV)), ytr[idx].to(DEV)).backward()
            opt.step()
    m.eval(); out = {}
    with torch.no_grad():
        for c, (Av, yv) in vals.items():
            ok = sum((m(Av[i:i + 256].to(DEV)).argmax(1).cpu() == yv[i:i + 256]).sum().item()
                     for i in range(0, len(yv), 256))
            out[c] = ok / len(yv)
    return out


def main():
    Atr, ytr = load("train", "clean")
    vals = {c: load("val", c) for c in CONDS if os.path.exists(os.path.join(CACHE, f"jester_appT_val_{c}.pt"))}
    nc = int(ytr.max()) + 1
    res = {}
    for mode in ["mid", "meanpool", "tconv"]:
        runs = [run(mode, Atr, ytr, vals, nc, s) for s in SEEDS]
        res[mode] = {c: {"mean": float(np.mean([r[c] for r in runs])),
                         "std": float(np.std([r[c] for r in runs])),
                         "raw": [r[c] for r in runs]} for c in vals}
        print(f"[appT] {mode:9s} " + "  ".join(f"{c}={res[mode][c]['mean']:.4f}" for c in vals), flush=True)
    os.makedirs("runs", exist_ok=True)
    json.dump({"protocol": "12k clean-train subset (seed 0) / full val, 3 seeds", "results": res},
              open("runs/temporal_app_baseline.json", "w"), indent=1)
    print("saved runs/temporal_app_baseline.json")


if __name__ == "__main__":
    main()
