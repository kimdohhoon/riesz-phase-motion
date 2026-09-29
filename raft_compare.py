"""Modern-flow (RAFT) baseline comparison on Jester, controlled.

Adds RAFT-large as a 4th motion representation and compares it to Riesz-phase and
Farneback flow under an IDENTICAL pipeline: same frozen ViT-L appearance (reused from
cache), same temporal CNN + balanced fusion, same train subset (12k) and full val,
clean + low-light sweep. Tests whether phase's low-light advantage holds against SOTA
learned flow. RAFT maps are extracted here (decode parallelized via DataLoader workers,
RAFT on GPU) and cached; phase/flow features are reused from the full_study caches.

Run:  CUDA_VISIBLE_DEVICES=0 python raft_compare.py
"""
from __future__ import annotations
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from train_smoke import make_dataset
from full_study import full_items, CACHE, OUT, SIZE, NF
from motion_features import raft_maps
from corruptions import apply as apply_corrupt
from arid_motion_split import FusedModel

DEVICE = "cuda"
SEEDS = [0, 1, 2]
TRAIN_CAP = 12000
SUBSET_SEED = 0
CONDS = [("clean", None), ("low_light_s1", ("low_light", 1)),
         ("low_light_s3", ("low_light", 3)), ("low_light_s5", ("low_light", 5))]
NUM_WORKERS = 16


def load_cache(motion, cond):
    b = torch.load(os.path.join(CACHE, f"jester_{motion}_{cond}.pt"), map_location="cpu")
    return b["app"].float(), b["maps"].float(), b["ys"]


class DecodeDS(Dataset):
    """yields decoded+corrupted clip for a fixed ordered index list (shuffle=False)."""
    def __init__(self, base, di_list, corrupt):
        self.base, self.di, self.corrupt = base, di_list, corrupt

    def __len__(self):
        return len(self.di)

    def __getitem__(self, i):
        clip, _ = self.base[self.di[i]]
        if self.corrupt is not None:
            clip = apply_corrupt(self.corrupt[0], clip, self.corrupt[1]).clamp(0, 1)
        return clip


def extract_raft(base, di_list, corrupt, tag):
    """RAFT maps for di_list in order; cache to disk."""
    loader = DataLoader(DecodeDS(base, di_list, corrupt), batch_size=4,
                        num_workers=NUM_WORKERS, collate_fn=lambda b: b)
    out, n = [], 0
    import time
    t0 = time.time()
    for batch in loader:
        for clip in batch:
            out.append(raft_maps(clip.to(DEVICE), out_size=OUT).cpu())
            n += 1
        if n % 4000 < 4:
            r = n / (time.time() - t0 + 1e-9)
            print(f"[raft] {tag}: {n}/{len(di_list)} ({r:.1f}/s, "
                  f"eta {(len(di_list)-n)/r/60:.0f}min)", flush=True)
    return torch.stack(out)


def get_raft(dataset_split, base, chosen, sel, cond_name, corrupt):
    """cached RAFT (app+maps+ys) for one (split,cond); train uses `sel` subset."""
    path = os.path.join(CACHE, f"jester_raft_{dataset_split}_{cond_name}.pt")
    # pair with phase-cache app/ys (same ViT-L appearance, same clip order)
    app, _, ys = load_cache("phase", f"{dataset_split}_{cond_name}"
                            if dataset_split == "val" else "train_clean")
    if dataset_split == "train":
        app, ys = app[sel], ys[sel]
        di = [chosen[i][0] for i in sel.tolist()]
    else:
        di = [c[0] for c in chosen]
    if os.path.exists(path):
        maps = torch.load(path, map_location="cpu")["maps"].float()
    else:
        maps = extract_raft(base, di, corrupt, f"{dataset_split}/{cond_name}")
        torch.save({"maps": maps.half()}, path)
    return app, maps, ys


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
    print(f"[raft] train_sub={len(sel)} val={len(chosen_va)} nc={nc}", flush=True)

    # ---- RAFT features (extract+cache) ----
    raft_tr = get_raft("train", tr_ds, chosen_tr, sel, "clean", None)
    raft_va = {c: get_raft("val", va_ds, chosen_va, sel, c, corr) for c, corr in CONDS}

    # ---- phase / flow features (reuse cache; same subset) ----
    feats = {"raft": (raft_tr, raft_va)}
    for m in ["phase", "flow"]:
        a, M, y = load_cache(m, "train_clean")
        tr = (a[sel], M[sel], y[sel])
        va = {c: load_cache(m, f"val_{c}") for c, _ in CONDS}
        feats[m] = (tr, va)

    app_dim = raft_tr[0].shape[1]
    res = {}
    for m, (tr, va) in feats.items():
        in_ch = tr[1].shape[2]
        row = {c: [] for c, _ in CONDS}
        for s in SEEDS:
            model = fit(tr, in_ch, nc, app_dim, s)
            for c, _ in CONDS:
                row[c].append(ev(model, va[c]))
        res[m] = {c: [float(np.mean(v)), float(np.std(v))] for c, v in row.items()}
        print(f"[raft] {m:6s} " + "  ".join(
            f"{c}={res[m][c][0]:.3f}" for c, _ in CONDS), flush=True)

    with open("runs/raft_compare.json", "w") as f:
        json.dump({"n_train": int(len(sel)), "n_val": len(chosen_va), "results": res}, f, indent=2)
    print("\n=== MODERN-FLOW COMPARISON (Jester, train 12k / full val, 3-seed) ===")
    print(f"{'motion':8s}" + "".join(f"{c:>14}" for c, _ in CONDS))
    for m in ["phase", "flow", "raft"]:
        print(f"{m:8s}" + "".join(f"{res[m][c][0]:.3f}±{res[m][c][1]:.2f}".rjust(14)
                                  for c, _ in CONDS))
    print("[raft] saved -> runs/raft_compare.json")


if __name__ == "__main__":
    main()
