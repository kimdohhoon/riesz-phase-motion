"""Corruption robustness study (paper Sec. 4.4).

Train each motion stream on CLEAN Jester, then evaluate on clean + corrupted
val (low-light / noise / blur). Compare absolute acc AND relative drop across
phase vs flow vs framediff. Hypothesis: Riesz-phase (illumination-invariant)
drops LESS under low-light/noise; motion_blur is phase's weak regime (kept
honestly). This is where phase's value should show even if clean is a tie.

Run:  CUDA_VISIBLE_DEVICES=0 python corruption_study.py
"""
from __future__ import annotations
import sys
import torch

from train_smoke import (make_dataset, pick_class_subset, val_subset,
                         build_features, fit, evaluate)
from clip_features import CLIPAppearance
from monogenic import MonogenicExtractor

DEVICE = "cuda"
SIZE, NF, OUT, NS = 112, 12, 32, 6
DATASET = sys.argv[1] if len(sys.argv) > 1 else "jester"   # jester | ipn | ssv2
CLASSES, TR_PER, VA_PER = 27, 150, 40
MOTIONS = ["phase", "flow", "framediff"]
SEVS = [1, 3, 5]
# (label, (corruption_name, severity) or None) -- severity sweep
CONDS = [("clean", None)]
for _lbl, _cn in [("low_light", "low_light"), ("noise", "gaussian_noise"),
                  ("blur", "motion_blur")]:
    for _s in SEVS:
        CONDS.append((f"{_lbl}_s{_s}", (_cn, _s)))


def main():
    clip_enc = CLIPAppearance(device=DEVICE, dtype="float32")
    mono = MonogenicExtractor(img_size=SIZE, n_scales=NS, trainable=False).to(DEVICE)
    tr_ds = make_dataset(DATASET, "train", NF, SIZE)
    va_ds = make_dataset(DATASET, "val", NF, SIZE)
    chosen_tr, remap = pick_class_subset(tr_ds.items, CLASSES, TR_PER)
    chosen_va = val_subset(va_ds.items, remap, VA_PER)
    nc = len(remap)
    print(f"[corr] dataset={DATASET} classes={nc} train={len(chosen_tr)} val={len(chosen_va)}", flush=True)

    results = {}
    for mt in MOTIONS:
        print(f"\n[corr] === motion={mt} ===", flush=True)
        tr = build_features(tr_ds, chosen_tr, clip_enc, mt, mono, OUT, DEVICE, "tr")
        model = fit(tr[:3], use_motion=True, n_classes=nc, app_dim=clip_enc.dim, device=DEVICE)
        row = {}
        for cname, corr in CONDS:
            va = build_features(va_ds, chosen_va, clip_enc, mt, mono, OUT, DEVICE,
                                cname, corrupt=corr)
            row[cname] = evaluate(model, va[:3], use_motion=True)
            print(f"[corr]   {mt:10s} {cname:10s} acc={row[cname]:.3f}", flush=True)
        results[mt] = row

    labels = [l for l, _ in CONDS]
    print("\n=== absolute acc ===")
    print(f"{'motion':<11}" + "".join(f"{l:>13}" for l in labels))
    for mt in MOTIONS:
        r = results[mt]
        print(f"{mt:<11}" + "".join(f"{r[l]:>13.3f}" for l in labels))
    print("\n=== relative drop vs clean (%) ===")
    nonclean = [l for l in labels if l != "clean"]
    print(f"{'motion':<11}" + "".join(f"{l:>13}" for l in nonclean))
    for mt in MOTIONS:
        r = results[mt]; c = r["clean"]
        print(f"{mt:<11}" + "".join(f"{(c-r[l])/c*100:>12.0f}%" for l in nonclean))
    print("\nphase should win under low_light across severities (illumination-invariant);")
    print("blur is phase's expected weak regime.")


if __name__ == "__main__":
    main()
