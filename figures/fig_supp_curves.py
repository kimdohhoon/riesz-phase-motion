"""Supplementary Fig. 2 (accuracy heatmap over all corruptions) and Supplementary Fig. 3
(corruption-severity curves for Jester and IPN Hand), from the full_study.py results
runs/full_{jester,ipn}_{phase,flow,framediff}.json.

  python -m figures.fig_supp_curves
  -> runs/gallery/F6_heatmap.{png,pdf}, runs/gallery/F2_corruption_grid_{jester,ipn}.{png,pdf}
"""
from __future__ import annotations
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figures import save_png_pdf

OUT = "runs/gallery"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 140})

DSETS = ["jester", "ipn"]
DNAME = {"jester": "Jester", "ipn": "IPN Hand"}
MOTIONS = ["phase", "flow", "framediff"]
LABEL = {"phase": "Riesz-phase (ours)", "flow": "optical flow", "framediff": "frame-diff"}
COLOR = {"phase": "#2ca02c", "flow": "#1f77b4", "framediff": "#ff7f0e"}
MARK = {"phase": "o", "flow": "s", "framediff": "^"}
CORR = ["low_light", "noise", "blur"]
CORRNAME = {"low_light": "low-light", "noise": "noise", "blur": "blur"}
SEV = [1, 3, 5]


def load():
    d = {}
    for ds in DSETS:
        d[ds] = {}
        for m in MOTIONS:
            p = f"runs/full_{ds}_{m}.json"
            if os.path.exists(p):
                d[ds][m] = json.load(open(p))
    return d


def mean(d, ds, m, cond):
    return d[ds][m]["conditions"][cond]["mean"]


def std(d, ds, m, cond):
    return d[ds][m]["conditions"][cond]["std"]


def save(fig, name):
    fig.tight_layout()
    p = os.path.join(OUT, name)
    save_png_pdf(fig, p, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def fig_corruption_grid(d, ds):
    """3-panel severity curves (low-light / noise / blur) for one dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, corr in zip(axes, CORR):
        for m in MOTIONS:
            if m not in d[ds]:
                continue
            ys = [mean(d, ds, m, "clean")] + [mean(d, ds, m, f"{corr}_s{s}") for s in SEV]
            es = [std(d, ds, m, "clean")] + [std(d, ds, m, f"{corr}_s{s}") for s in SEV]
            ax.errorbar([0] + SEV, ys, yerr=es, marker=MARK[m], color=COLOR[m],
                        capsize=3, lw=2, label=LABEL[m])
        ax.set_title(CORRNAME[corr])
        ax.set_xlabel("severity (0 = clean)")
        ax.set_xticks([0] + SEV)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Top-1 accuracy")
    fig.suptitle(f"{DNAME[ds]}: full corruption sweep "
                 f"(phase wins low-light & noise; frame-diff best under blur)",
                 fontweight="bold")
    save(fig, f"F2_corruption_grid_{ds}.png")


def fig_heatmap(d):
    """Accuracy heatmap: methods x conditions, one row-block per dataset."""
    conds = ["clean", "low_light_s1", "low_light_s3", "low_light_s5",
             "noise_s1", "noise_s3", "noise_s5", "blur_s1", "blur_s3", "blur_s5"]
    clab = ["clean", "ll1", "ll3", "ll5", "no1", "no3", "no5", "bl1", "bl3", "bl5"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2))
    for ax, ds in zip(axes, DSETS):
        if not d[ds]:
            continue
        M = np.array([[mean(d, ds, m, c) for c in conds] for m in MOTIONS])
        im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=0.9)
        ax.set_xticks(range(len(conds))); ax.set_xticklabels(clab, fontsize=9)
        ax.set_yticks(range(len(MOTIONS)))
        ax.set_yticklabels([LABEL[m] for m in MOTIONS], fontsize=9)
        ax.set_title(DNAME[ds], fontsize=11)
        for i in range(len(MOTIONS)):
            for j in range(len(conds)):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.5 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.suptitle("Accuracy across all corruptions (full scale, 3-seed mean)",
                 fontweight="bold")
    save(fig, "F6_heatmap.png")


def main():
    d = load()
    for ds in DSETS:
        if d[ds]:
            fig_corruption_grid(d, ds)
    fig_heatmap(d)


if __name__ == "__main__":
    main()
