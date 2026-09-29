"""Supplementary Fig. 1: mechanism grid over six Jester gestures (clean RGB, low-light RGB,
low-light flow magnitude, low-light Riesz |dphi|), using the central-motion clip and frame
selection of visualize_mechanism_v2.

  python -m figures.fig_supp_mechanism_grid   -> runs/gallery/mech_combined_grid_v2.png
"""
import os
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rieszmotion.monogenic import MonogenicExtractor
from rieszmotion.corruptions import low_light
from figures.visualize_mechanism_v2 import (jester_ids, load_jester, flow_mag, phase_dmap,
                                    best_clip, SIZE, NF)

GESTURES = ["Swiping Right", "Zooming In With Full Hand", "Sliding Two Fingers Down",
            "Turning Hand Clockwise", "Drumming Fingers", "Swiping Left"]
ext = MonogenicExtractor(img_size=SIZE, n_scales=6, trainable=False)

rows = []
for g in GESTURES:
    clip, fr = best_clip(load_jester, jester_ids(g, 8))   # central-motion clip+frame
    rows.append((g, clip, fr))

n = len(rows)
fig, ax = plt.subplots(n, 4, figsize=(9, 2.3 * n))
titles = ["RGB (clean)", "RGB (low-light)", "flow |v| (low-light)", "Riesz |Δphase| (low-light)"]
for r, (g, clip, t) in enumerate(rows):
    dark = low_light(clip, 5).clamp(0, 1)
    fm_c = flow_mag(clip)[t]; fm_d = flow_mag(dark)[t]
    dp_c = phase_dmap(ext, clip)[t]; dp_d = phase_dmap(ext, dark)[t]
    imgs = [clip[t].permute(1, 2, 0).numpy(), dark[t].permute(1, 2, 0).numpy(),
            fm_d.numpy(), dp_d.numpy()]
    cmaps = [None, None, "inferno", "viridis"]
    vmax = [None, None, float(fm_c.max()) + 1e-6, float(dp_c.max()) + 1e-6]
    for c in range(4):
        if cmaps[c]:
            ax[r, c].imshow(imgs[c], cmap=cmaps[c], vmin=0, vmax=vmax[c])
        else:
            ax[r, c].imshow(imgs[c])
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        if r == 0:
            ax[r, c].set_title(titles[c], fontsize=10)
    ax[r, 0].set_ylabel(g, fontsize=9)
fig.suptitle("Under low-light, RGB & optical flow collapse while Riesz Δphase persists",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs("runs/gallery", exist_ok=True)
fig.savefig("runs/gallery/mech_combined_grid_v2.png", dpi=140); plt.close(fig)
print("saved runs/gallery/mech_combined_grid_v2.png")
