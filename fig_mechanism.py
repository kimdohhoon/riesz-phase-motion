"""Paper Fig. 4: mechanism on a Swiping Left clip, as one horizontal strip of
6 panels grouped by modality (clean | low-light):
  [RGB clean | RGB dark] [flow clean | flow dark] [Riesz d-phase clean | dark]
RGB & flow collapse under low light; Riesz |dphi| stays intact.

Run:  python fig_mechanism.py   -> runs/mechanism_wide.png
"""
from __future__ import annotations
import os, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from monogenic import MonogenicExtractor
from motion_features import flow_maps, monogenic_maps
from corruptions import low_light

from paths import JESTER_ROOT as ROOT, JESTER_CSV
TRAIN_CSV = JESTER_CSV["train"]
GESTURE = "Swiping Left"
SIZE, NF, SEV = 112, 12, 5
OUTPNG = "runs/mechanism_wide.png"


def list_ids(gesture, n):
    ids = []
    with open(TRAIN_CSV) as f:
        for line in f:
            vid, lab = (line.strip().split(";") + [""])[:2]
            if lab == gesture:
                ids.append(vid)
                if len(ids) >= n:
                    break
    return ids


def load_clip(vid):
    d = os.path.join(ROOT, vid)
    jpgs = sorted(x for x in os.listdir(d) if x.endswith(".jpg"))
    idx = [min(len(jpgs) - 1, int(i * len(jpgs) / NF)) for i in range(NF)]
    fr = [torch.from_numpy(np.asarray(Image.open(os.path.join(d, jpgs[i])).convert("RGB")))
          .permute(2, 0, 1).float() / 255.0 for i in idx]
    clip = torch.stack(fr)
    return torch.nn.functional.interpolate(clip, size=(SIZE, SIZE), mode="bilinear",
                                           align_corners=False)


def flow_mag(clip):
    f = flow_maps(clip, out_size=SIZE)
    return torch.sqrt(f[:, 0] ** 2 + f[:, 1] ** 2)


def phase_dmap(ext, clip):
    m = monogenic_maps(ext, clip, out_size=SIZE, temporal=True)
    return m[1:, 4].abs()


def cos_preserve(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))


def main():
    ext = MonogenicExtractor(img_size=SIZE, n_scales=6, trainable=False)
    ids = list_ids(GESTURE, 12)
    flow_pres, phase_pres = [], []
    for vid in ids:
        clip = load_clip(vid); dark = low_light(clip, SEV).clamp(0, 1)
        flow_pres.append(cos_preserve(flow_mag(clip), flow_mag(dark)))
        phase_pres.append(cos_preserve(phase_dmap(ext, clip), phase_dmap(ext, dark)))
    fp, pp = np.mean(flow_pres), np.mean(phase_pres)
    print(f"[viz] preservation flow {fp:.3f} vs phase {pp:.3f}", flush=True)

    clip = load_clip(ids[0]); dark = low_light(clip, SEV).clamp(0, 1)
    t = NF // 2
    rgb_c, rgb_d = clip[t].permute(1, 2, 0).numpy(), dark[t].permute(1, 2, 0).numpy()
    fmag_c, fmag_d = flow_mag(clip)[t].numpy(), flow_mag(dark)[t].numpy()
    dph_c, dph_d = phase_dmap(ext, clip)[t].numpy(), phase_dmap(ext, dark)[t].numpy()

    # 1 row x 6 cols, grouped by modality (clean | low-light)
    fig, ax = plt.subplots(1, 6, figsize=(12, 2.5))
    panels = [
        (rgb_c, None, None, "RGB\nclean"),
        (rgb_d, None, None, "RGB\nlow-light"),
        (fmag_c, "inferno", float(fmag_c.max()), "flow $|v|$\nclean"),
        (fmag_d, "inferno", float(fmag_c.max()), "flow $|v|$\nlow-light"),
        (dph_c, "viridis", float(dph_c.max()), r"Riesz $|\Delta\phi|$" + "\nclean"),
        (dph_d, "viridis", float(dph_c.max()), r"Riesz $|\Delta\phi|$" + "\nlow-light"),
    ]
    for i, (img, cmap, vmax, title) in enumerate(panels):
        if cmap is None:
            ax[i].imshow(img)
        else:
            ax[i].imshow(img, cmap=cmap, vmin=0, vmax=vmax)
        ax[i].set_title(title, fontsize=10)
        ax[i].set_xticks([]); ax[i].set_yticks([])
        # thin separators between modality groups
        for s in ax[i].spines.values():
            s.set_visible(True); s.set_color("0.5"); s.set_linewidth(0.6)
    fig.tight_layout()
    os.makedirs("runs", exist_ok=True)
    fig.savefig(OUTPNG, dpi=150, bbox_inches="tight")
    print(f"[viz] saved -> {OUTPNG}  (flow {fp:.2f} / phase {pp:.2f})")


if __name__ == "__main__":
    main()
