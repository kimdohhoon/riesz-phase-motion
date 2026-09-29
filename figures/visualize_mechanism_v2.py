"""M2-style mechanism panels, v2: picks the MAX-MOTION frame (so the moving hand
is visible, not a static face) and supports BOTH Jester and IPN clips.

  python -m figures.visualize_mechanism_v2 jester   # ~10 panels, varied gestures
  python -m figures.visualize_mechanism_v2 ipn      # ~10 panels, varied IPN classes

Each panel: rows RGB / optical-flow|v| / Riesz|Δphase|, cols clean / low-light(s5),
title shows per-clip motion preservation (cosine) flow vs phase.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from rieszmotion.monogenic import MonogenicExtractor
from rieszmotion.motion_features import flow_maps, monogenic_maps
from rieszmotion.corruptions import low_light

from rieszmotion.paths import JESTER_ROOT as JROOT, JESTER_CSV
SIZE, NF, SEV = 112, 12, 5
JCSV = JESTER_CSV["train"]

JESTER_GESTURES = ["Swiping Left", "Swiping Right", "Swiping Up", "Swiping Down",
                   "Zooming In With Full Hand", "Zooming Out With Full Hand",
                   "Sliding Two Fingers Down", "Drumming Fingers", "Stop Sign",
                   "Turning Hand Clockwise", "Pushing Hand Away", "Thumb Up"]
# best-effort IPN code -> readable
IPN_NAME = {"B0A": "point 1 finger", "B0B": "point 2 fingers", "G01": "click 1 finger",
            "G02": "click 2 fingers", "G03": "throw up", "G04": "throw down",
            "G05": "throw left", "G06": "throw right", "G07": "open twice",
            "G08": "2x click 1 finger", "G09": "2x click 2 fingers",
            "G10": "zoom in", "G11": "zoom out"}


def jester_ids(gesture, n):
    ids = []
    with open(JCSV) as f:
        for line in f:
            vid, lab = (line.strip().split(";") + [""])[:2]
            if lab == gesture:
                ids.append(vid)
                if len(ids) >= n:
                    break
    return ids


def load_jester(vid):
    d = os.path.join(JROOT, vid)
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


def _center(fm):
    """per-frame mean flow in the central 50% box (moving hand, not edge/face)."""
    lo, hi = int(SIZE * 0.22), int(SIZE * 0.82)
    return fm[:, lo:hi, lo:hi].mean(dim=(1, 2))


def peak_frame(fm):
    """frame with the most CENTRAL motion (so the hand is in-frame, not at edge)."""
    return int(_center(fm).argmax())


def panel(ext, label, clip, path, frame=None):
    dark = low_light(clip, SEV).clamp(0, 1)
    fm_c, fm_d = flow_mag(clip), flow_mag(dark)
    dp_c, dp_d = phase_dmap(ext, clip), phase_dmap(ext, dark)
    fp = cos_preserve(fm_c, fm_d); pp = cos_preserve(dp_c, dp_d)
    t = frame if frame is not None else peak_frame(fm_c)  # central-motion frame
    fig, ax = plt.subplots(3, 2, figsize=(6.4, 9))
    cols = [(clip[t], fm_c[t], dp_c[t], "clean"),
            (dark[t], fm_d[t], dp_d[t], f"low-light s{SEV}")]
    for col, (rgb, fm, dp, tag) in enumerate(cols):
        ax[0, col].imshow(rgb.permute(1, 2, 0).numpy()); ax[0, col].set_title(f"RGB ({tag})")
        ax[1, col].imshow(fm.numpy(), cmap="inferno", vmin=0, vmax=float(fm_c[t].max()) + 1e-6)
        ax[1, col].set_title("optical flow |v|")
        ax[2, col].imshow(dp.numpy(), cmap="viridis", vmin=0, vmax=float(dp_c[t].max()) + 1e-6)
        ax[2, col].set_title("Riesz |Δphase| (ours)")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{label}  (frame {t})  —  motion preserved:\n"
                 f"flow {fp:.2f}  vs  phase {pp:.2f}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=140); plt.close(fig)
    return fp, pp


def best_clip(loader, ids):
    """pick the (clip, frame) with the largest CENTRAL clean motion -> hand in view."""
    best, bestframe, bestscore = None, None, -1
    for vid in ids:
        try:
            clip = loader(vid)
        except Exception:
            continue
        cen = _center(flow_mag(clip))
        sc = float(cen.max()); fr = int(cen.argmax())
        if sc > bestscore:
            best, bestframe, bestscore = clip, fr, sc
    return best, bestframe


def run_jester(ext, outdir):
    rows = []
    for g in JESTER_GESTURES:
        ids = jester_ids(g, 8)
        if not ids:
            print(f"  [skip] {g}"); continue
        clip, fr = best_clip(load_jester, ids)
        if clip is None:
            continue
        p = os.path.join(outdir, f"jester_{g.replace(' ', '_')}.png")
        fp, pp = panel(ext, g, clip, p, frame=fr)
        rows.append((g, fp, pp))
        print(f"  saved {p}  (flow {fp:.2f} / phase {pp:.2f})")
    return rows


def run_ipn(ext, outdir):
    from rieszmotion.datasets import IPNDataset
    from rieszmotion.paths import IPN_ROOT, IPN_ANNOT
    ds = IPNDataset(IPN_ROOT, IPN_ANNOT["train"], n_frames=NF, size=SIZE)
    inv = {v: k for k, v in ds.label_map.items()}          # label-id -> code
    # one clip per class (first ~10 distinct labels)
    seen, rows = {}, []
    for i, (vid, lab, s, e) in enumerate(ds.items):
        if lab in seen:
            continue
        seen[lab] = i
        if len(seen) >= 12:
            break
    for lab, i in sorted(seen.items()):
        clip, _ = ds[i]
        code = inv[lab]
        name = f"IPN {code} ({IPN_NAME.get(code, '?')})"
        p = os.path.join(outdir, f"ipn_{code}.png")
        fp, pp = panel(ext, name, clip, p)
        rows.append((name, fp, pp))
        print(f"  saved {p}  (flow {fp:.2f} / phase {pp:.2f})")
    return rows


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "jester"
    outdir = f"runs/gallery/mech2_{which}"
    os.makedirs(outdir, exist_ok=True)
    ext = MonogenicExtractor(img_size=SIZE, n_scales=6, trainable=False)
    rows = run_jester(ext, outdir) if which == "jester" else run_ipn(ext, outdir)
    fps = [r[1] for r in rows]; pps = [r[2] for r in rows]
    print(f"[v2/{which}] {len(rows)} panels. mean preserve: "
          f"flow {np.mean(fps):.2f} vs phase {np.mean(pps):.2f} -> {outdir}")


if __name__ == "__main__":
    main()
