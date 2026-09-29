"""Synthetic verification of the clip-level Riesz-phase motion descriptor.

No data, no model. Builds moving/static gratings with known motion & contrast
and asserts the clip descriptor behaves:
  M1 fixed output dim
  M2 moving clip has motion energy >> static clip
  M3 amplitude invariance: same motion at half contrast -> motion stats ~equal
                           (only the energy channel scales)
  M4 direction sensitivity: horizontal vs vertical motion -> different
                            orientation histograms

Run:  python -m tests.verify_motion_features
"""
from __future__ import annotations
import sys
import math
import torch

from rieszmotion.motion_features import MonogenicMotionFeature

PASS, FAIL = "PASS", "FAIL"
_res = []
S = 128


def check(name, cond, detail=""):
    cond = bool(cond)
    _res.append(cond)
    print(f"[{PASS if cond else FAIL}] {name}" + (f"  --  {detail}" if detail else ""))


def grating(size=S, wavelength=16.0, angle=0.0, shift=0.0, contrast=1.0):
    yy, xx = torch.meshgrid(torch.arange(size).float(),
                            torch.arange(size).float(), indexing="ij")
    ct, st = math.cos(angle), math.sin(angle)
    coord = (xx - shift * ct) * ct + (yy - shift * st) * st
    return (contrast * torch.sin(2 * math.pi * coord / wavelength)).view(1, 1, size, size)


def clip(T=8, angle=0.0, speed=1.0, contrast=1.0, wavelength=16.0):
    return torch.cat([grating(wavelength=wavelength, angle=angle,
                              shift=t * speed, contrast=contrast)
                      for t in range(T)], 0)        # (T,1,H,W)


def main():
    print("=" * 64)
    print("Clip-level monogenic motion descriptor verification")
    print("=" * 64)
    mf = MonogenicMotionFeature(img_size=S, n_scales=8, n_orient_bins=8)
    print(f"  out_dim={mf.out_dim}  (pair_dim={mf.pair_dim})")

    moving = mf(clip(angle=0.0, speed=1.0))
    static = mf(clip(angle=0.0, speed=0.0))
    half = mf(clip(angle=0.0, speed=1.0, contrast=0.5))
    vert = mf(clip(angle=math.pi / 2, speed=1.0))

    # M1
    check("M1 fixed output dim", moving.shape == (mf.out_dim,),
          f"shape={tuple(moving.shape)}")

    # M2  index0 = mean|dphi| (clip-mean over pairs)
    check("M2 moving >> static motion energy",
          moving[0] > 10 * (static[0] + 1e-6),
          f"moving|dphi|={moving[0]:.4f} static={static[0]:.4f}")

    # M3 amplitude invariance: motion stats (mean,std + orient hist) ~equal;
    # energy channel (index 2 and its pooled std at pair_dim+2) differs.
    motion_idx = [0, 1] + list(range(3, mf.pair_dim))     # exclude energy ch
    diff = (moving[motion_idx] - half[motion_idx]).abs().max()
    e_full, e_half = moving[2], half[2]
    check("M3a contrast-invariant motion stats", diff < 0.05,
          f"max|Δ|={diff:.4f}")
    check("M3b energy channel scales with contrast",
          e_half < e_full and e_half > 0,
          f"energy full={e_full:.4f} half={e_half:.4f}")

    # M4 direction: orientation histogram occupies bins 3..3+8
    h_idx = list(range(3, 3 + 8))
    hist_h = moving[h_idx]
    hist_v = vert[h_idx]
    peak_h = int(hist_h.argmax())
    peak_v = int(hist_v.argmax())
    check("M4 horizontal vs vertical -> different orientation peak",
          peak_h != peak_v, f"peak_h={peak_h} peak_v={peak_v}")

    print("=" * 64)
    n = sum(_res)
    print(f"{n}/{len(_res)} checks passed")
    print("=" * 64)
    sys.exit(0 if all(_res) else 1)


if __name__ == "__main__":
    main()
