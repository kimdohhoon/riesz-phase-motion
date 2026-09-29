"""FULL-scale single-benchmark study (no per-class subsetting).

Runs ONE motion type over the ENTIRE dataset split (e.g. full Jester:
118,562 train / 14,787 val) and produces, from a single extraction pass:
  - contribution    : appearance-only / motion-only / fused (clean, 3 seeds)
  - clean controlled: fused clean acc (this motion)
  - low-light sweep : fused acc at low_light s1/s3/s5 (3 seeds)  [multiseed]
  - corruption sweep: fused acc at noise / blur s1/s3/s5 (3 seeds) [corruption]

Features are cached to disk (fp16) per (motion, split, condition) so a crash
or shared-GPU eviction resumes instantly. Compute is fp32.

Run (per motion, per GPU):
  CUDA_VISIBLE_DEVICES=0 python full_study.py jester phase
  CUDA_VISIBLE_DEVICES=1 python full_study.py jester flow
  CUDA_VISIBLE_DEVICES=1 python full_study.py jester framediff

Then aggregate the 3 json outputs with aggregate_full.py.
"""
from __future__ import annotations
import os
import sys
import json
import time
import numpy as np
import torch

from train_smoke import (make_dataset, build_features, fit, evaluate)
from clip_features import CLIPAppearance
from monogenic import MonogenicExtractor

DEVICE = "cuda"
SIZE, NF, OUT, NS = 112, 12, 32, 6
SEEDS = [0, 1, 2]
CACHE = "cache"
os.makedirs(CACHE, exist_ok=True)
os.makedirs("runs", exist_ok=True)

# all eval conditions from one extraction pass (train is clean only)
_DS0 = sys.argv[1] if len(sys.argv) > 1 else "jester"
CONDS = [("clean", None)]
if _DS0 == "arid":
    # ARID is ALREADY naturally dark -> "clean" IS the natural low-light test.
    # add a further-darkening sweep to match the Jester/IPN robustness-curve format.
    for _s in (1, 3, 5):
        CONDS.append((f"low_light_s{_s}", ("low_light", _s)))
else:
    for _lbl, _cn in [("low_light", "low_light"), ("noise", "gaussian_noise"),
                      ("blur", "motion_blur")]:
        for _s in (1, 3, 5):
            CONDS.append((f"{_lbl}_s{_s}", (_cn, _s)))


def full_items(ds, remap=None):
    """Every clip of the split, labels remapped to 0..C-1 (label at items[i][1])."""
    if remap is None:
        labels = sorted({it[1] for it in ds.items})
        remap = {o: i for i, o in enumerate(labels)}
    chosen = [(i, remap[it[1]]) for i, it in enumerate(ds.items) if it[1] in remap]
    return chosen, remap


def cached_features(ds, chosen, clip_enc, motion, mono, dataset, split, cond_name,
                    corrupt):
    """Extract (app, maps, ys) for one (motion, split, condition); cache fp16 to disk."""
    path = os.path.join(CACHE, f"{dataset}_{motion}_{split}_{cond_name}.pt")
    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu")
        return (blob["app"].float(), blob["maps"].float(), blob["ys"])
    t = time.time()
    app, maps, ys, fails = build_features(ds, chosen, clip_enc, motion, mono, OUT,
                                          DEVICE, f"{split}/{cond_name}", corrupt=corrupt)
    torch.save({"app": app.half(), "maps": maps.half(), "ys": ys}, path)
    print(f"[full]   extracted {split}/{cond_name}: {tuple(maps.shape)} "
          f"fails={fails} ({(time.time()-t)/60:.1f}min) -> cached", flush=True)
    return app.float(), maps.float(), ys


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "jester"
    motion = sys.argv[2] if len(sys.argv) > 2 else "phase"
    print(f"[full] dataset={dataset} motion={motion} seeds={SEEDS}", flush=True)

    clip_enc = CLIPAppearance(device=DEVICE, dtype="float32")
    mono = MonogenicExtractor(img_size=SIZE, n_scales=NS, trainable=False).to(DEVICE)
    tr_ds = make_dataset(dataset, "train", NF, SIZE)
    va_ds = make_dataset(dataset, "val", NF, SIZE)
    chosen_tr, remap = full_items(tr_ds)
    chosen_va, _ = full_items(va_ds, remap)
    nc = len(remap)
    print(f"[full] classes={nc} train={len(chosen_tr)} val={len(chosen_va)}", flush=True)

    # ---- extract (cached) ----
    tr = cached_features(tr_ds, chosen_tr, clip_enc, motion, mono, dataset, "train",
                         "clean", None)
    va_feats = {}
    for cname, corr in CONDS:
        va_feats[cname] = cached_features(va_ds, chosen_va, clip_enc, motion, mono,
                                          dataset, "val", cname, corr)

    # ---- train + eval (3 seeds) ----
    out = {"dataset": dataset, "motion": motion, "n_classes": nc,
           "n_train": len(chosen_tr), "n_val": len(chosen_va),
           "chance": 1.0 / nc, "seeds": SEEDS}

    # contribution (clean): app-only / motion-only / fused
    contrib = {"app": [], "motion": [], "fused": []}
    cond = {c[0]: [] for c in CONDS}
    for s in SEEDS:
        m_app = fit(tr[:3], use_motion=False, n_classes=nc, app_dim=clip_enc.dim,
                    device=DEVICE, use_app=True, seed=s)
        contrib["app"].append(evaluate(m_app, va_feats["clean"][:3], use_motion=False))
        m_mot = fit(tr[:3], use_motion=True, n_classes=nc, app_dim=clip_enc.dim,
                    device=DEVICE, use_app=False, seed=s)
        contrib["motion"].append(evaluate(m_mot, va_feats["clean"][:3], use_motion=True))
        m_fus = fit(tr[:3], use_motion=True, n_classes=nc, app_dim=clip_enc.dim,
                    device=DEVICE, use_app=True, seed=s)
        contrib["fused"].append(evaluate(m_fus, va_feats["clean"][:3], use_motion=True))
        for cname, _ in CONDS:
            cond[cname].append(evaluate(m_fus, va_feats[cname][:3], use_motion=True))
        print(f"[full] seed={s} app={contrib['app'][-1]:.3f} "
              f"motion={contrib['motion'][-1]:.3f} fused={contrib['fused'][-1]:.3f} "
              f"| ll_s5={cond['low_light_s5'][-1]:.3f}", flush=True)

    def ms(x):
        return {"mean": float(np.mean(x)), "std": float(np.std(x)), "raw": x}
    out["contribution"] = {k: ms(v) for k, v in contrib.items()}
    out["conditions"] = {k: ms(v) for k, v in cond.items()}

    jpath = f"runs/full_{dataset}_{motion}.json"
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[full] === {dataset}/{motion} (full: {out['n_train']}tr/{out['n_val']}va) ===")
    print(f"  app={out['contribution']['app']['mean']:.3f} "
          f"motion={out['contribution']['motion']['mean']:.3f} "
          f"fused={out['contribution']['fused']['mean']:.3f}")
    for cname, _ in CONDS:
        c = out["conditions"][cname]
        print(f"  {cname:14s} {c['mean']:.3f} +/- {c['std']:.3f}")
    print(f"[full] saved -> {jpath}", flush=True)


if __name__ == "__main__":
    main()
