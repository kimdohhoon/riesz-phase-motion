"""Mechanism metrics on the full Jester validation set (paper Sec. 4.8, Supp. Tab. 9).

From the cached 32x32 maps the temporal CNN consumes, form a per-frame motion-MAGNITUDE
map for each clip -- |dphi| for phase (frame 0 skipped: its dphi is zero padding),
sqrt(u^2+v^2) for flow, and the L2 norm over R,G,B for frame difference -- and compare
the clean map a with the corrupted map b of the same clip:

  pattern   : cosine similarity of the flattened maps (scale-invariant), mean over clips
  amplitude : mean(|b|) / mean(|a|)
  per-class : pattern preservation per Jester class

  python -m experiments.mechanism.mechanism_metrics [--cond low_light_s5]   -> runs/mechanism_metrics.json
Needs the full_study.py caches jester_{rep}_val_{clean,<cond>}.pt.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
from rieszmotion.paths import JESTER_LABELS

REPS = ["phase", "flow", "framediff"]


def magnitude(rep, m):
    if rep == "phase":
        return m[:, 1:, 4].float().abs()
    if rep == "flow":
        return torch.sqrt(m[:, :, 0].float() ** 2 + m[:, :, 1].float() ** 2)
    return m.float().pow(2).sum(dim=2).sqrt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="low_light_s5")
    ap.add_argument("--cache", default="cache")
    a = ap.parse_args()
    # labels are indices in jester-v1-labels.csv FILE order (train_smoke.load_jester_label_map),
    # not alphabetical order
    names = [l.strip() for l in open(JESTER_LABELS) if l.strip()]
    out = {"cond": a.cond, "classes": names, "reps": {}}
    for rep in REPS:
        ld = lambda c: torch.load(os.path.join(a.cache, f"jester_{rep}_val_{c}.pt"),
                                  map_location="cpu", weights_only=False)
        dc, dd = ld("clean"), ld(a.cond)
        assert torch.equal(dc["ys"], dd["ys"])
        cos, ac, ad, N = [], 0.0, 0.0, dc["maps"].shape[0]
        for i in range(0, N, 512):
            x, y = magnitude(rep, dc["maps"][i:i + 512]), magnitude(rep, dd["maps"][i:i + 512])
            xf, yf = x.reshape(len(x), -1), y.reshape(len(y), -1)
            cos.append(((xf * yf).sum(1) / (xf.norm(dim=1) * yf.norm(dim=1) + 1e-8)).numpy())
            ac += x.abs().mean().item() * len(x); ad += y.abs().mean().item() * len(y)
        cos = np.concatenate(cos); ys = dc["ys"].numpy()
        out["reps"][rep] = {
            "n": int(N), "pattern_cos": float(cos.mean()), "pattern_cos_std": float(cos.std()),
            "amplitude_ratio": ad / (ac + 1e-12),
            "per_class": {names[c]: {"n": int((ys == c).sum()), "mean": float(cos[ys == c].mean())}
                          for c in range(len(names)) if (ys == c).any()}}
        r = out["reps"][rep]
        print(f"[{rep:9s}] pattern cos {r['pattern_cos']:.3f}  amplitude {r['amplitude_ratio']:.3f}", flush=True)
    os.makedirs("runs", exist_ok=True)
    json.dump(out, open("runs/mechanism_metrics.json", "w"), indent=1)
    print("saved runs/mechanism_metrics.json")


if __name__ == "__main__":
    main()
