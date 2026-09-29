"""Channel-matched control (paper Sec. 4.4): does phase's low-light advantage survive when
flow / frame-difference are given the SAME 7-channel budget (explicit orientation +
temporal-difference) that phase enjoys?

Compares, under the identical pipeline (12k train / full val, 3 seeds, cached ViT-L
appearance), clean + low-light sweep:
  phase (7ch)         [cached]
  flow (2ch)          [cached]   flow-aug (7ch)        [extracted here]
  framediff (3ch)     [cached]   framediff-aug (7ch)   [extracted here]

If phase still leads flow-aug/framediff-aug under low light, the advantage is the
representation, not the hand-engineered channels.

Run:  CUDA_VISIBLE_DEVICES=0 python channel_match.py
"""
from __future__ import annotations
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from train_smoke import make_dataset
from full_study import full_items, CACHE, OUT, SIZE, NF
from motion_features import flow_maps_aug, framediff_maps_aug
from corruptions import apply as apply_corrupt
from arid_motion_split import FusedModel

DEVICE = "cuda"
SEEDS = [0, 1, 2]
TRAIN_CAP = 12000
SUBSET_SEED = 0
CONDS = [("clean", None), ("low_light_s1", ("low_light", 1)),
         ("low_light_s3", ("low_light", 3)), ("low_light_s5", ("low_light", 5))]
NUM_WORKERS = 24


def load_cache(motion, cond):
    b = torch.load(os.path.join(CACHE, f"jester_{motion}_{cond}.pt"), map_location="cpu")
    return b["app"].float(), b["maps"].float(), b["ys"]


class AugDS(Dataset):
    """worker computes flow-aug(7) + framediff-aug(7) on CPU (parallel)."""
    def __init__(self, base, di_list, corrupt):
        self.base, self.di, self.corrupt = base, di_list, corrupt

    def __len__(self):
        return len(self.di)

    def __getitem__(self, i):
        clip, _ = self.base[self.di[i]]
        if self.corrupt is not None:
            clip = apply_corrupt(self.corrupt[0], clip, self.corrupt[1]).clamp(0, 1)
        return (flow_maps_aug(clip, OUT).clone(), framediff_maps_aug(clip, OUT).clone())


def get_aug(split, base, chosen, sel, cond_name, corrupt):
    """cached (flow7, framediff7) maps for one (split,cond); train uses sel subset."""
    fp = os.path.join(CACHE, f"jester_flow7_{split}_{cond_name}.pt")
    dp = os.path.join(CACHE, f"jester_framediff7_{split}_{cond_name}.pt")
    di = [chosen[i][0] for i in sel.tolist()] if split == "train" else [c[0] for c in chosen]
    if os.path.exists(fp) and os.path.exists(dp):
        return (torch.load(fp, map_location="cpu")["maps"].float(),
                torch.load(dp, map_location="cpu")["maps"].float())
    loader = DataLoader(AugDS(base, di, corrupt), batch_size=8, num_workers=NUM_WORKERS,
                        collate_fn=lambda b: b)
    fl, fd, n, t0 = [], [], 0, time.time()
    for batch in loader:
        for f7, d7 in batch:
            fl.append(f7); fd.append(d7); n += 1
        if n % 5000 < 8:
            print(f"[chan] {split}/{cond_name}: {n}/{len(di)} "
                  f"({n/(time.time()-t0+1e-9):.1f}/s)", flush=True)
    fl, fd = torch.stack(fl), torch.stack(fd)
    torch.save({"maps": fl.half()}, fp); torch.save({"maps": fd.half()}, dp)
    return fl, fd


def fit(tr, in_ch, nc, app_dim, seed, epochs=60, bs=32):
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


def ev(model, va, bs=256):
    A, M, y = va
    model.eval(); c = 0
    with torch.no_grad():
        for i in range(0, len(y), bs):
            c += (model(A[i:i+bs].to(DEVICE), M[i:i+bs].to(DEVICE)).argmax(1).cpu()
                  == y[i:i+bs]).sum().item()
    return c / len(y)


def main():
    tr_ds = make_dataset("jester", "train", NF, SIZE)
    va_ds = make_dataset("jester", "val", NF, SIZE)
    chosen_tr, remap = full_items(tr_ds)
    chosen_va, _ = full_items(va_ds, remap)
    nc = len(remap)
    g = torch.Generator().manual_seed(SUBSET_SEED)
    sel = torch.randperm(len(chosen_tr), generator=g)[:TRAIN_CAP]
    print(f"[chan] train_sub={len(sel)} val={len(chosen_va)} nc={nc}", flush=True)

    # --- aug features (extract+cache) ---
    f7_tr, d7_tr = get_aug("train", tr_ds, chosen_tr, sel, "clean", None)
    f7_va, d7_va = {}, {}
    for c, corr in CONDS:
        f7_va[c], d7_va[c] = get_aug("val", va_ds, chosen_va, sel, c, corr)

    # --- appearance + ys from phase cache (same ViT-L, same order) ---
    app_tr, _, ys_tr = load_cache("phase", "train_clean")
    app_tr, ys_tr = app_tr[sel], ys_tr[sel]
    app_va = {c: load_cache("phase", f"val_{c}")[0] for c, _ in CONDS}
    ys_va = {c: load_cache("phase", f"val_{c}")[2] for c, _ in CONDS}
    app_dim = app_tr.shape[1]

    # assemble all motions: (train_maps, {cond: val_maps})
    feats = {
        "flow7": (f7_tr, f7_va),
        "framediff7": (d7_tr, d7_va),
    }
    for m in ["phase", "flow", "framediff"]:
        mtr = load_cache(m, "train_clean")[1][sel]
        mva = {c: load_cache(m, f"val_{c}")[1] for c, _ in CONDS}
        feats[m] = (mtr, mva)

    res = {}
    for m, (mtr, mva) in feats.items():
        in_ch = mtr.shape[2]
        row = {c: [] for c, _ in CONDS}
        for s in SEEDS:
            model = fit((app_tr, mtr, ys_tr), in_ch, nc, app_dim, s)
            for c, _ in CONDS:
                row[c].append(ev(model, (app_va[c], mva[c], ys_va[c])))
        res[m] = {c: [float(np.mean(v)), float(np.std(v))] for c, v in row.items()}
        print(f"[chan] {m:11s}(ch{in_ch}) " +
              "  ".join(f"{c}={res[m][c][0]:.3f}" for c, _ in CONDS), flush=True)

    with open("runs/channel_match.json", "w") as f:
        json.dump({"n_train": int(len(sel)), "results": res}, f, indent=2)
    print("\n=== CHANNEL-MATCHED CONTROL (Jester, 12k/full val, 3-seed) ===")
    order = ["phase", "flow", "flow7", "framediff", "framediff7"]
    print(f"{'motion':12s}" + "".join(f"{c:>13}" for c, _ in CONDS))
    for m in order:
        print(f"{m:12s}" + "".join(f"{res[m][c][0]:.3f}".rjust(13) for c, _ in CONDS))
    print("[chan] saved -> runs/channel_match.json")


if __name__ == "__main__":
    main()
