"""Design ablation on FULL Jester, computed entirely from cached features.

Each row isolates one design decision (paper Tab. 2). All variants reuse the
SAME cached frozen features (cache/jester_phase_{train,val}_*.pt), so no
re-extraction is needed -- the 7-channel maps are [phi,E,sin,cos,dphi,dphi*cos,
dphi*sin]; the static-only ablation just slices the first 4 channels.

Variants (clean + low-light s5, 3 seeds):
  A appearance only
  B + motion, static 4ch (no temporal Delta-phi), temporal CNN, balanced
  C + motion, 7ch, GLOBAL POOLING (no temporal order), balanced
  D + motion, 7ch, temporal CNN, NAIVE fusion (raw 768-d concat)
  E + motion, 7ch, temporal CNN, balanced fusion           <- ours (full)

Run:  CUDA_VISIBLE_DEVICES=0 python ablation_study.py
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
DS = "jester"


def load(cond):
    b = torch.load(os.path.join(CACHE, f"{DS}_phase_{cond}.pt"), map_location="cpu")
    return b["app"].float(), b["maps"].float(), b["ys"]


class GlobalPoolMotion(nn.Module):
    """Order-destroying baseline: global mean/std over (T,H,W) -> MLP."""
    def __init__(self, in_ch, dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * in_ch, 128), nn.ReLU(True),
                                 nn.Linear(128, dim))

    def forward(self, maps):                       # (B,T,C,H,W)
        mu = maps.mean(dim=(1, 3, 4)); sd = maps.std(dim=(1, 3, 4))
        return self.net(torch.cat([mu, sd], -1))


class AblationModel(nn.Module):
    def __init__(self, app_dim, n_classes, in_ch, use_app, use_motion,
                 motion_enc="cnn", fusion="balanced", motion_dim=64, hidden=256):
        super().__init__()
        self.use_app, self.use_motion, self.fusion = use_app, use_motion, fusion
        a_out = 0
        if use_app:
            if fusion == "balanced":
                self.app_proj = nn.Sequential(nn.Linear(app_dim, 128), nn.LayerNorm(128))
                a_out = 128
            else:                                   # naive: raw app, no proj/norm
                self.app_proj = nn.Identity(); a_out = app_dim
        m_out = 0
        if use_motion:
            self.motion = (TemporalMotionCNN(in_ch=in_ch, dim=motion_dim)
                           if motion_enc == "cnn" else GlobalPoolMotion(in_ch, motion_dim))
            self.motion_enc = motion_enc
            self.motion_norm = (nn.LayerNorm(motion_dim) if fusion == "balanced"
                                else nn.Identity())
            m_out = motion_dim
        self.head = nn.Sequential(nn.Linear(a_out + m_out, hidden), nn.ReLU(True),
                                  nn.Dropout(0.3), nn.Linear(hidden, n_classes))

    def forward(self, app, maps=None):
        parts = []
        if self.use_app:
            parts.append(self.app_proj(app))
        if self.use_motion:
            mv = self.motion(maps.permute(0, 2, 1, 3, 4)) if self.motion_enc == "cnn" \
                else self.motion(maps)
            parts.append(self.motion_norm(mv))
        return self.head(torch.cat(parts, -1))


def fit_eval(tr, evals, in_ch, use_app, use_motion, motion_enc, fusion, nc, app_dim,
             seed, epochs=60, bs=32):
    torch.manual_seed(seed)
    A, M, y = tr
    model = AblationModel(app_dim, nc, in_ch, use_app, use_motion, motion_enc,
                          fusion).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n = A.shape[0]
    for _ in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            a = A[idx].to(DEVICE)
            m = M[idx].to(DEVICE) if use_motion else None
            loss = nn.functional.cross_entropy(model(a, m), y[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    out = {}
    model.eval()
    with torch.no_grad():
        for name, (Ae, Me, ye) in evals.items():
            correct = 0
            for i in range(0, len(ye), 256):
                a = Ae[i:i + 256].to(DEVICE)
                m = Me[i:i + 256].to(DEVICE) if use_motion else None
                correct += (model(a, m).argmax(1).cpu() == ye[i:i + 256]).sum().item()
            out[name] = correct / len(ye)
    return out


def slice4(feat):
    a, m, y = feat
    return a, m[:, :, :4].contiguous(), y


TRAIN_CAP = 12000   # ablation on a fixed balanced-ish subset (full eval); shared-GPU


def main():
    tr7 = load("train_clean")
    ev7 = {"clean": load("val_clean"), "ll_s5": load("val_low_light_s5")}
    nc = int(tr7[2].max()) + 1
    app_dim = tr7[0].shape[1]
    if tr7[0].shape[0] > TRAIN_CAP:                       # subsample train for speed
        g = torch.Generator().manual_seed(0)
        sel = torch.randperm(tr7[0].shape[0], generator=g)[:TRAIN_CAP]
        tr7 = (tr7[0][sel], tr7[1][sel], tr7[2][sel])
    tr4 = slice4(tr7)
    ev4 = {k: slice4(v) for k, v in ev7.items()}
    print(f"[abl] {DS}: train={tr7[0].shape[0]} (cap {TRAIN_CAP}) val_full nc={nc}", flush=True)

    # (label, in_ch, use_app, use_motion, motion_enc, fusion, which_feats)
    VARIANTS = [
        ("A. appearance only",                 7, True,  False, "cnn",  "balanced", "7"),
        ("B. + motion static 4ch (no dphi)",   4, True,  True,  "cnn",  "balanced", "4"),
        ("C. + motion 7ch, global pooling",    7, True,  True,  "pool", "balanced", "7"),
        ("D. + motion 7ch, naive fusion",      7, True,  True,  "cnn",  "naive",    "7"),
        ("E. + motion 7ch, temporal CNN (ours)",7, True, True,  "cnn",  "balanced", "7"),
    ]
    rows = {}
    for lbl, in_ch, ua, um, me, fu, wf in VARIANTS:
        tr = tr4 if wf == "4" else tr7
        ev = ev4 if wf == "4" else ev7
        accs = {"clean": [], "ll_s5": []}
        for s in SEEDS:
            r = fit_eval(tr, ev, in_ch, ua, um, me, fu, nc, app_dim, s)
            accs["clean"].append(r["clean"]); accs["ll_s5"].append(r["ll_s5"])
        rows[lbl] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in accs.items()}
        print(f"[abl] {lbl:42s} clean {rows[lbl]['clean'][0]:.3f}±{rows[lbl]['clean'][1]:.3f}"
              f"  ll_s5 {rows[lbl]['ll_s5'][0]:.3f}±{rows[lbl]['ll_s5'][1]:.3f}", flush=True)

    with open("runs/ablation_jester.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\n=== ABLATION (full Jester, 3-seed) ===")
    print(f"{'variant':44s}{'clean':>14}{'low-light s5':>16}")
    for lbl, r in rows.items():
        print(f"{lbl:44s}{r['clean'][0]:>8.3f}±{r['clean'][1]:.2f}"
              f"{r['ll_s5'][0]:>10.3f}±{r['ll_s5'][1]:.2f}")
    print("[abl] saved -> runs/ablation_jester.json")


if __name__ == "__main__":
    main()
