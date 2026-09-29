"""Cheap, zero-re-extraction analysis: does phase's advantage CONCENTRATE on
motion-direction pairs, and does it PERSIST under low-light where flow collapses?

Reuses the cached fused features (train_clean + val clean/low_light s1/s3/s5) for
phase and flow. Trains the SAME fused TwoStream head (3 seeds) and slices per-clip
predictions into:
  - DIR  : the 10 direction-sensitive pairs (remapped classes 0..19)
  - NONDIR: the 7 non-directional classes (20..26)
  - ALL  : full val

The cell we care about: DIR accuracy under low-light, phase vs flow.

Run:  CUDA_VISIBLE_DEVICES=1 python -m experiments.dir_subset
"""
from __future__ import annotations
import os, sys, json, numpy as np, torch
from experiments.full_study import CACHE
from rieszmotion.train import fit

DEVICE = "cuda"
SEEDS = [0, 1, 2]
FULL = len(sys.argv) > 1 and sys.argv[1] == "full"
TRAIN_CAP = None if FULL else 12000     # None -> use ALL 118k train clips
SUBSET_SEED = 0
OUT = "runs/dir_subset_full.json" if FULL else "runs/dir_subset.json"
REPS = ["phase", "flow"]
CONDS = ["clean", "low_light_s1", "low_light_s3", "low_light_s5"]
DIR_CLASSES = set(range(20))      # classes 0..19 = the 10 direction pairs


def load(rep, split, cond):
    b = torch.load(os.path.join(CACHE, f"jester_{rep}_{split}_{cond}.pt"),
                   map_location="cpu")
    return b["app"].float(), b["maps"].float(), b["ys"]


def predict(model, A, M, bs=256):
    dev = next(model.parameters()).device
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(A), bs):
            out.append(model(A[i:i+bs].to(dev), M[i:i+bs].to(dev)).argmax(1).cpu())
    return torch.cat(out)


def subset_acc(pred, y, mask):
    m = mask
    return (pred[m] == y[m]).float().mean().item() if m.sum() > 0 else float("nan")


def run_rep(rep):
    app_tr, m_tr, ys_tr = load(rep, "train", "clean")
    nc = int(ys_tr.max()) + 1
    app_dim = app_tr.shape[1]
    if TRAIN_CAP is not None:
        g = torch.Generator().manual_seed(SUBSET_SEED)
        sel = torch.randperm(app_tr.shape[0], generator=g)[:TRAIN_CAP]
        app_tr, m_tr, ys_tr = app_tr[sel], m_tr[sel], ys_tr[sel]
    va = {c: load(rep, "val", c) for c in CONDS}
    dir_mask = {c: torch.tensor([int(l) in DIR_CLASSES for l in va[c][2]]) for c in CONDS}
    rows = {c: {"all": [], "dir": [], "nondir": []} for c in CONDS}
    for s in SEEDS:
        model = fit((app_tr, m_tr, ys_tr), use_motion=True, n_classes=nc,
                    app_dim=app_dim, device=DEVICE, use_app=True, seed=s)
        for c in CONDS:
            A, M, y = va[c]
            pred = predict(model, A, M)
            dm = dir_mask[c]
            rows[c]["all"].append((pred == y).float().mean().item())
            rows[c]["dir"].append(subset_acc(pred, y, dm))
            rows[c]["nondir"].append(subset_acc(pred, y, ~dm))
        print(f"[dir] {rep} seed={s} done", flush=True)
    return {c: {k: [float(np.mean(v)), float(np.std(v))] for k, v in d.items()}
            for c, d in rows.items()}


def main():
    print(f"[dir] MODE={'FULL(118k)' if FULL else '12k'} out={OUT}", flush=True)
    res = {r: run_rep(r) for r in REPS}
    json.dump(res, open(OUT, "w"), indent=2)
    print("\n=== DIRECTION-PAIR vs NON-DIRECTION accuracy (Jester, fused, 3-seed) ===")
    hdr = f"{'cond':14s}" + "".join(f"{r+'/'+s:>14}" for r in REPS for s in ("dir", "all"))
    print(hdr)
    for c in CONDS:
        line = f"{c:14s}"
        for r in REPS:
            line += f"{res[r][c]['dir'][0]:>10.3f}±{res[r][c]['dir'][1]:.2f}"
            line += f"{res[r][c]['all'][0]:>10.3f}±{res[r][c]['all'][1]:.2f}"
        print(line)
    print("\n--- KEY CELL: DIR accuracy, phase vs flow ---")
    for c in CONDS:
        p, f = res["phase"][c]["dir"][0], res["flow"][c]["dir"][0]
        print(f"  {c:14s} phase_dir={p:.3f}  flow_dir={f:.3f}  gap={p-f:+.3f}")
    print("\n--- NON-DIR accuracy, phase vs flow ---")
    for c in CONDS:
        p, f = res["phase"][c]["nondir"][0], res["flow"][c]["nondir"][0]
        print(f"  {c:14s} phase_nd={p:.3f}   flow_nd={f:.3f}   gap={p-f:+.3f}")
    print(f"[dir] saved -> {OUT}")


if __name__ == "__main__":
    main()
