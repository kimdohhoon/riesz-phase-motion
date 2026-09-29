"""Per-frame CLIP features for the appearance-only temporal baseline (paper Sec. 4.7,
Supp. Tab. 8).

The main model's appearance stream sees only the middle frame, so its motion stream is
also its only temporal pathway. To separate the two, this extracts frozen-CLIP features
for ALL T=12 frames. Protocol: a fixed 12k-clip clean training subset (seed 0), the full
validation split, as in the design ablation (paper Tab. 2).

  --cond clean | ll5_none | ll5_gamma | ll5r_none
        (main low light s5 uncorrected / + mean-based gamma / non-invertible control s5)

  python -m experiments.temporal_baseline.extract_temporal_app --split train --cond clean
  python -m experiments.temporal_baseline.extract_temporal_app --split val   --cond ll5_gamma
  -> cache/jester_appT_{split}_{cond}.pt  with {app: (N,T,768), ys: (N,)}
"""
from __future__ import annotations
import argparse, os, sys, time
import torch
from rieszmotion.train import make_dataset
from experiments.full_study import full_items
from rieszmotion.clip_features import CLIPAppearance
from rieszmotion.corruptions import low_light
from rieszmotion.corruptions_noninvertible import low_light_noninvertible
from rieszmotion.corrections import gamma_mean_based, CLEAN_MEAN

SIZE, NF, TRAIN_CAP = 112, 12, 12000
CLIP_ID = os.environ.get("CLIP_MODEL", "openai/clip-vit-large-patch14-336")


def corrupt(clip, cond):
    if cond == "clean":
        return clip
    if cond == "ll5_none":
        return low_light(clip, 5).clamp(0, 1)
    if cond == "ll5_gamma":
        return gamma_mean_based(low_light(clip, 5).clamp(0, 1), CLEAN_MEAN["jester"])
    if cond == "ll5r_none":
        return low_light_noninvertible(clip, 5)
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--cond", default="clean", choices=["clean", "ll5_none", "ll5_gamma", "ll5r_none"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default="cache")
    a = ap.parse_args()
    assert a.split == "val" or a.cond == "clean", "training is always on clean data"
    os.makedirs(a.cache, exist_ok=True)

    tr = make_dataset("jester", "train", NF, SIZE)
    ds = tr if a.split == "train" else make_dataset("jester", "val", NF, SIZE)
    _, remap = full_items(tr)
    chosen, _ = full_items(ds, remap)
    if a.split == "train":
        g = torch.Generator().manual_seed(0)
        chosen = [chosen[i] for i in torch.randperm(len(chosen), generator=g)[:TRAIN_CAP].tolist()]
    if a.limit:
        chosen = chosen[:a.limit]
    print(f"[appT] {a.split} {a.cond} clips={len(chosen)}", flush=True)

    enc = CLIPAppearance(model_id=CLIP_ID, device=a.device)
    feats, ys, t0 = [], [], time.time()
    for n, (di, nl) in enumerate(chosen):
        clip = corrupt(ds[di][0].to(a.device), a.cond)
        with torch.no_grad():
            feats.append(enc(clip).cpu())            # (T,3,H,W) -> (T,768)
        ys.append(nl)
        if (n + 1) % 1000 == 0:
            print(f"  {n+1}/{len(chosen)}  {(time.time()-t0)/60:.1f} min", flush=True)
    p = os.path.join(a.cache, f"jester_appT_{a.split}_{a.cond}.pt")
    torch.save({"app": torch.stack(feats).half(), "ys": torch.tensor(ys, dtype=torch.long)}, p)
    print("saved", p)


if __name__ == "__main__":
    main()
