"""Panels of paper Fig. 2: monogenic (Riesz) decomposition of one Jester frame into local
phase, orientation, and energy.

  python -m figures.fig_decomposition   -> runs/fig_riesz_decomp.png
"""
from __future__ import annotations
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from rieszmotion.monogenic import MonogenicExtractor
from rieszmotion.paths import JESTER_ROOT as ROOT, JESTER_CSV

TRAIN_CSV = JESTER_CSV["train"]
SIZE, NF = 112, 12
os.makedirs("runs", exist_ok=True)


def ids_for(gesture, n):
    out = []
    with open(TRAIN_CSV) as f:
        for line in f:
            vid, lab = (line.strip().split(";") + [""])[:2]
            if lab == gesture:
                out.append(vid)
                if len(out) >= n:
                    break
    return out


def load_clip(vid):
    d = os.path.join(ROOT, vid)
    jpgs = sorted(x for x in os.listdir(d) if x.endswith(".jpg"))
    idx = [min(len(jpgs) - 1, int(i * len(jpgs) / NF)) for i in range(NF)]
    fr = [torch.from_numpy(np.asarray(Image.open(os.path.join(d, jpgs[i])).convert("RGB")))
          .permute(2, 0, 1).float() / 255.0 for i in idx]
    clip = torch.stack(fr)
    return torch.nn.functional.interpolate(clip, size=(SIZE, SIZE), mode="bilinear",
                                           align_corners=False)


def fig_decomp(ext):
    clip = load_clip(ids_for("Swiping Left", 1)[0])
    t = NF // 2
    o = ext(clip[t:t + 1])
    rgb = clip[t].permute(1, 2, 0).numpy()
    phase = o["phase"][0, 0].numpy()
    orient = o["orientation"][0, 0].numpy()
    energy = o["energy"][0, 0].numpy()
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
    ax[0].imshow(rgb); ax[0].set_title("RGB frame")
    ax[1].imshow(phase, cmap="viridis"); ax[1].set_title("local phase  φ")
    ax[2].imshow(orient, cmap="hsv"); ax[2].set_title("orientation  θ")
    ax[3].imshow(energy, cmap="magma"); ax[3].set_title("local energy  E")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Riesz / monogenic decomposition of one frame\n"
                 "phase = local structure TYPE (edge vs flat), brightness-invariant"
                 "  ·  energy = structure strength", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("runs/fig_riesz_decomp.png", dpi=140); plt.close(fig)
    print("saved fig_riesz_decomp.png")


if __name__ == "__main__":
    fig_decomp(MonogenicExtractor(img_size=SIZE, n_scales=6, trainable=False))
