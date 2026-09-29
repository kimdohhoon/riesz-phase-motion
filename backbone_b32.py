"""Backbone-generalization / MoCLIP-Lite-backbone comparison: ViT-B/32.

MoCLIP-Lite uses a frozen CLIP ViT-B/32. We re-run our controlled phase-vs-flow
comparison at ViT-B/32 (re-extracting only the appearance; the Riesz/flow motion maps
are backbone-independent and reused from cache) to show (a) the low-light advantage is
not specific to ViT-L/14, and (b) it holds at MoCLIP-Lite's backbone. Same controlled
protocol: 12k-clip train subset, full val, clean + low-light sweep, 3 seeds.

Run:  CUDA_VISIBLE_DEVICES=0 python backbone_b32.py
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
from clip_features import CLIPAppearance
from corruptions import apply as apply_corrupt
from arid_motion_split import FusedModel

DEVICE = "cuda"
SEEDS = [0, 1, 2]
TRAIN_CAP = 12000
SUBSET_SEED = 0
B32_ID = "openai/clip-vit-base-patch32"
CONDS = [("clean", None), ("low_light_s1", ("low_light", 1)),
         ("low_light_s3", ("low_light", 3)), ("low_light_s5", ("low_light", 5))]
NUM_WORKERS = 16


def load_maps(motion, cond):
    b = torch.load(os.path.join(CACHE, f"jester_{motion}_{cond}.pt"), map_location="cpu")
    return b["maps"].float(), b["ys"]


class DecodeDS(Dataset):
    def __init__(self, base, di_list, corrupt):
        self.base, self.di, self.corrupt = base, di_list, corrupt

    def __len__(self):
        return len(self.di)

    def __getitem__(self, i):
        clip, _ = self.base[self.di[i]]
        if self.corrupt is not None:
            clip = apply_corrupt(self.corrupt[0], clip, self.corrupt[1]).clamp(0, 1)
        return clip


def extract_b32_app(enc, base, di_list, corrupt, tag):
    loader = DataLoader(DecodeDS(base, di_list, corrupt), batch_size=8,
                        num_workers=NUM_WORKERS, collate_fn=lambda b: b)
    out, n, t0 = [], 0, time.time()
    for batch in loader:
        for clip in batch:
            with torch.no_grad():
                out.append(enc.mid_frame(clip.to(DEVICE)).cpu())
            n += 1
        if n % 5000 < 8:
            r = n / (time.time() - t0 + 1e-9)
            print(f"[b32] {tag}: {n}/{len(di_list)} ({r:.1f}/s)", flush=True)
    return torch.stack(out)


def get_app(enc, split, base, chosen, sel, cond_name, corrupt):
    path = os.path.join(CACHE, f"jester_b32app_{split}_{cond_name}.pt")
    if split == "train":
        di = [chosen[i][0] for i in sel.tolist()]
    else:
        di = [c[0] for c in chosen]
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")["app"].float()
    app = extract_b32_app(enc, base, di, corrupt, f"{split}/{cond_name}")
    torch.save({"app": app.half()}, path)
    return app


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
    enc = CLIPAppearance(model_id=B32_ID, device=DEVICE, dtype="float32")
    print(f"[b32] CLIP {B32_ID} dim={enc.dim}", flush=True)
    tr_ds = make_dataset("jester", "train", NF, SIZE)
    va_ds = make_dataset("jester", "val", NF, SIZE)
    chosen_tr, remap = full_items(tr_ds)
    chosen_va, _ = full_items(va_ds, remap)
    nc = len(remap)
    g = torch.Generator().manual_seed(SUBSET_SEED)
    sel = torch.randperm(len(chosen_tr), generator=g)[:TRAIN_CAP]
    print(f"[b32] train_sub={len(sel)} val={len(chosen_va)} nc={nc}", flush=True)

    app_tr = get_app(enc, "train", tr_ds, chosen_tr, sel, "clean", None)
    app_va = {c: get_app(enc, "val", va_ds, chosen_va, sel, c, corr) for c, corr in CONDS}

    res = {}
    for m in ["phase", "flow"]:
        mtr, ytr = load_maps(m, "train_clean")
        tr = (app_tr, mtr[sel], ytr[sel])
        in_ch = mtr.shape[2]
        row = {c: [] for c, _ in CONDS}
        for s in SEEDS:
            model = fit(tr, in_ch, nc, enc.dim, s)
            for c, _ in CONDS:
                mv, yv = load_maps(m, f"val_{c}")
                row[c].append(ev(model, (app_va[c], mv, yv)))
        res[m] = {c: [float(np.mean(v)), float(np.std(v))] for c, v in row.items()}
        print(f"[b32] {m:6s} " + "  ".join(f"{c}={res[m][c][0]:.3f}" for c, _ in CONDS), flush=True)

    with open("runs/backbone_b32.json", "w") as f:
        json.dump({"backbone": B32_ID, "n_train": int(len(sel)),
                   "n_val": len(chosen_va), "results": res}, f, indent=2)
    print("\n=== ViT-B/32 BACKBONE (Jester, train 12k / full val, 3-seed) ===")
    print(f"{'motion':8s}" + "".join(f"{c:>14}" for c, _ in CONDS))
    for m in ["phase", "flow"]:
        print(f"{m:8s}" + "".join(f"{res[m][c][0]:.3f}±{res[m][c][1]:.2f}".rjust(14)
                                  for c, _ in CONDS))
    print("[b32] saved -> runs/backbone_b32.json")


if __name__ == "__main__":
    main()
