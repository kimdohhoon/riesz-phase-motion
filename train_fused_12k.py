"""Fused model (phase) re-trained on the 12k clean-train subset of the temporal-appearance
baseline, evaluated under the same four conditions (paper Sec. 4.7, Supp. Tab. 8).

The subset is drawn exactly as in extract_temporal_app.py (randperm, seed 0), so both
models see the same training clips. Needs the full_study.py clean caches and the
extract_controls.py caches ll5_none, ll5_gamma (main corruption, s5) and ll5r_none
(non-invertible control, s5).

  python train_fused_12k.py   -> runs/fused_12k_phase.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_smoke import fit, evaluate

CACHE, DEV, CAP, SEEDS, REP = "cache", "cuda", 12000, [0, 1, 2], "phase"
CONDS = ["clean", "ll5_none", "ll5_gamma", "ll5r_none"]


def load(split, cond, sel=None):
    """maps are cached as fp16; subset BEFORE widening to fp32 to bound RAM."""
    b = torch.load(os.path.join(CACHE, f"jester_{REP}_{split}_{cond}.pt"),
                   map_location="cpu", weights_only=False)
    A, M, y = b["app"], b["maps"], b["ys"]
    if sel is not None:
        A, M, y = A[sel], M[sel], y[sel]
    return A.float(), M.float(), y


def main():
    n = torch.load(os.path.join(CACHE, f"jester_{REP}_train_clean.pt"),
                   map_location="cpu", weights_only=False)["ys"].shape[0]
    sel = torch.randperm(n, generator=torch.Generator().manual_seed(0))[:CAP]
    tr = load("train", "clean", sel)
    nc = int(tr[2].max()) + 1
    models = [fit(tr, True, nc, tr[0].shape[-1], DEV, seed=s) for s in SEEDS]
    del tr
    res = {}
    for c in CONDS:                                   # one validation condition in RAM at a time
        va = load("val", c)
        res[c] = [float(evaluate(m, va, True)) for m in models]
        del va
        print(f"[fused12k] {c:10s} " + " ".join(f"{x:.4f}" for x in res[c]) + f"  mean={np.mean(res[c]):.4f}", flush=True)
    os.makedirs("runs", exist_ok=True)
    json.dump({"protocol": "12k clean-train subset (seed 0) / full val, 3 seeds", "rep": REP,
               "results": {c: {"mean": float(np.mean(v)), "std": float(np.std(v)), "raw": v}
                           for c, v in res.items()}},
              open(f"runs/fused_12k_{REP}.json", "w"), indent=1)
    print(f"saved runs/fused_12k_{REP}.json")


if __name__ == "__main__":
    main()
