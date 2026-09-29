"""Accuracy vs. motion SNR over the 27 (cue x corruption x severity) cells of the
full-Jester sweep (paper Sec. 4.8, Supp. Sec. 3).

Motion SNR is computed on the motion CHANNELS the temporal CNN receives (dphi and its two
orientation-projected components for phase; u, v for flow; the R, G, B differences for
frame difference), with signal and error power pooled over all validation clips:
    SNR = 10 log10( mean(a^2) / mean((b - a)^2) ),  a clean map, b corrupted map.
Reports Pearson r, Spearman rho, and the partial correlation with severity regressed
out of both variables. Cells share models and clips, so treat it as descriptive.

  python -m experiments.mechanism.mechanism_correlation   -> runs/motion_snr.json, runs/motion_snr_stats.json
Needs the full_study.py caches for all corruptions and runs/full_jester_{rep}.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch
from experiments.full_study import CACHE

REPS = ["phase", "flow", "framediff"]
MOTION_CH = {"phase": slice(4, 7), "flow": slice(0, None), "framediff": slice(0, None)}
CORR, SEVS = ["low_light", "noise", "blur"], [1, 3, 5]


def maps(rep, cond):
    return torch.load(os.path.join(CACHE, f"jester_{rep}_val_{cond}.pt"),
                      map_location="cpu", weights_only=False)["maps"].float()


def snr_db(a, b):
    return 10 * np.log10((a ** 2).mean().item() / (((b - a) ** 2).mean().item() + 1e-8))


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def main():
    acc = {r: json.load(open(f"runs/full_jester_{r}.json"))["conditions"] for r in REPS}
    pts = []
    for r in REPS:
        clean = maps(r, "clean")[:, :, MOTION_CH[r]]
        for c in CORR:
            for s in SEVS:
                cond = f"{c}_s{s}"
                pts.append({"rep": r, "corr": c, "sev": s,
                            "snr_db": snr_db(clean, maps(r, cond)[:, :, MOTION_CH[r]]),
                            "acc": acc[r][cond]["mean"]})
                print(f"[snr] {r:9s} {cond:12s} SNR={pts[-1]['snr_db']:6.2f} dB  acc={pts[-1]['acc']:.3f}", flush=True)
        del clean
    x = np.array([p["snr_db"] for p in pts]); y = np.array([p["acc"] for p in pts])
    sev = np.array([p["sev"] for p in pts], float)
    res = lambda v: v - np.polyval(np.polyfit(sev, v, 1), sev)
    stats = {"n_cells": len(pts), "pearson_r": float(np.corrcoef(x, y)[0, 1]),
             "spearman_rho": float(np.corrcoef(rank(x), rank(y))[0, 1]),
             "partial_r_given_severity": float(np.corrcoef(res(x), res(y))[0, 1])}
    os.makedirs("runs", exist_ok=True)
    json.dump(pts, open("runs/motion_snr.json", "w"), indent=1)
    json.dump(stats, open("runs/motion_snr_stats.json", "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    main()
