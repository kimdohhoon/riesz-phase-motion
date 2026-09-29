"""Multi-seed low-light robustness + figure (the main result, stats + plot).

Extract features once, then train with multiple seeds and evaluate on
clean + low_light s1/3/5. Reports mean+/-std per (motion, condition) and saves a
robustness curve. Confirms the single-seed low-light win is real.

Run:  CUDA_VISIBLE_DEVICES=1 python multiseed_study.py
"""
from __future__ import annotations
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_smoke import (make_dataset, pick_class_subset, val_subset,
                         build_features, fit, evaluate)
from clip_features import CLIPAppearance
from monogenic import MonogenicExtractor

DEVICE = "cuda"
SIZE, NF, OUT, NS = 112, 12, 32, 6
DATASET = sys.argv[1] if len(sys.argv) > 1 else "jester"   # jester | ipn | ssv2
CLASSES, TR_PER, VA_PER = 27, 150, 40
MOTIONS = ["phase", "flow", "framediff"]
SEEDS = [0, 1, 2]
# severity axis: clean=0, then low_light 1/3/5
CONDS = [("clean", None, 0), ("ll_s1", ("low_light", 1), 1),
         ("ll_s3", ("low_light", 3), 3), ("ll_s5", ("low_light", 5), 5)]
OUTPNG = f"runs/lowlight_{DATASET}.png"


def main():
    clip_enc = CLIPAppearance(device=DEVICE, dtype="float32")
    mono = MonogenicExtractor(img_size=SIZE, n_scales=NS, trainable=False).to(DEVICE)
    tr_ds = make_dataset(DATASET, "train", NF, SIZE)
    va_ds = make_dataset(DATASET, "val", NF, SIZE)
    chosen_tr, remap = pick_class_subset(tr_ds.items, CLASSES, TR_PER)
    chosen_va = val_subset(va_ds.items, remap, VA_PER)
    nc = len(remap)
    print(f"[ms] classes={nc} train={len(chosen_tr)} val={len(chosen_va)} seeds={SEEDS}", flush=True)

    mean = {mt: {} for mt in MOTIONS}
    std = {mt: {} for mt in MOTIONS}
    for mt in MOTIONS:
        print(f"\n[ms] === motion={mt} ===", flush=True)
        tr = build_features(tr_ds, chosen_tr, clip_enc, mt, mono, OUT, DEVICE, "tr")
        # extract each eval condition once (seed-independent)
        va_feats = {}
        for cname, corr, _ in CONDS:
            va_feats[cname] = build_features(va_ds, chosen_va, clip_enc, mt, mono,
                                             OUT, DEVICE, cname, corrupt=corr)
        for cname, _, _ in CONDS:
            accs = []
            for s in SEEDS:
                model = fit(tr[:3], use_motion=True, n_classes=nc,
                            app_dim=clip_enc.dim, device=DEVICE, seed=s)
                accs.append(evaluate(model, va_feats[cname][:3], use_motion=True))
            mean[mt][cname] = float(np.mean(accs))
            std[mt][cname] = float(np.std(accs))
            print(f"[ms]   {mt:10s} {cname:8s} {mean[mt][cname]:.3f} +/- {std[mt][cname]:.3f}", flush=True)

    # table
    labels = [c[0] for c in CONDS]
    print("\n" + "=" * 64)
    print(f"{'motion':<11}" + "".join(f"{l:>13}" for l in labels))
    for mt in MOTIONS:
        print(f"{mt:<11}" + "".join(f"{mean[mt][l]:.3f}±{std[mt][l]:.2f}".rjust(13) for l in labels))
    print("=" * 64)

    # figure: robustness curve
    sev = [c[2] for c in CONDS]
    plt.figure(figsize=(6, 4.2))
    for mt in MOTIONS:
        m = [mean[mt][c[0]] for c in CONDS]
        e = [std[mt][c[0]] for c in CONDS]
        plt.errorbar(sev, m, yerr=e, marker="o", capsize=3, label=mt)
    plt.xlabel("low-light severity (0 = clean)")
    plt.ylabel("accuracy (appearance + motion)")
    plt.title(f"{DATASET.upper()} {nc}-class: low-light robustness ({len(SEEDS)} seeds)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    import os
    os.makedirs("runs", exist_ok=True)
    plt.savefig(OUTPNG, dpi=130)
    print(f"[ms] saved figure -> {OUTPNG}")


if __name__ == "__main__":
    main()
