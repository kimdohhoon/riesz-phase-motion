"""Synthetic, data-free verification of the monogenic motion cue.

No dataset, no model. Builds gratings/blobs with KNOWN motion and contrast and
asserts the two claims our whole thesis rests on:

  C1  phase constancy : temporal phase-difference grows ~monotonically with
                        displacement, and is ~0 for a static pair.
  C2  amplitude invariance : changing contrast/illumination (NOT motion) leaves
                        the phase ~unchanged while energy scales accordingly.
  C3  structure localization : energy is high on textured regions, low on flat.

If C1 and C2 hold, the core premise ("phase encodes motion, invariant to
illumination") is sound on controlled inputs before any video is touched.

Run:  python verify_monogenic.py
"""
from __future__ import annotations
import sys
import math
import torch

from monogenic import MonogenicExtractor, temporal_phase_difference, motion_descriptor

PASS, FAIL = "PASS", "FAIL"
_results = []
S = 128                      # image size
EPS = 1e-6


def check(name, cond, detail=""):
    cond = bool(cond)
    _results.append(cond)
    print(f"[{PASS if cond else FAIL}] {name}" + (f"  --  {detail}" if detail else ""))


def emean(x, w):
    """energy-weighted mean of |x| (focus on textured, reliable regions)."""
    return float((w * x.abs()).sum() / (w.sum() + EPS))


def grating(size=S, wavelength=16.0, angle=0.0, shift=0.0, contrast=1.0):
    """Sinusoidal grating; `shift` (px) moves it along its wave direction."""
    yy, xx = torch.meshgrid(torch.arange(size).float(),
                            torch.arange(size).float(), indexing="ij")
    ct, st = math.cos(angle), math.sin(angle)
    coord = (xx - shift * ct) * ct + (yy - shift * st) * st
    img = torch.sin(2 * math.pi * coord / wavelength)
    return (contrast * img).view(1, 1, size, size)


def gaussian_blob(size=S, cx=None, cy=None, sigma=10.0):
    cx = size / 2 if cx is None else cx
    cy = size / 2 if cy is None else cy
    yy, xx = torch.meshgrid(torch.arange(size).float(),
                            torch.arange(size).float(), indexing="ij")
    g = torch.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
    return g.view(1, 1, size, size)


def main():
    print("=" * 64)
    print("Monogenic motion-cue verification (synthetic, data-free)")
    print("=" * 64)
    ext = MonogenicExtractor(img_size=S, n_scales=8, min_wavelength=4.0,
                             trainable=False)
    lam = 16.0

    # ---- C1: phase constancy -- dphi grows with displacement -------------
    base = grating(wavelength=lam, shift=0.0)
    dvals = [0.0, 1.0, 2.0, 3.0]
    dphis = []
    for d in dvals:
        moved = grating(wavelength=lam, shift=d)
        dphi, w, _ = motion_descriptor(ext, base, moved)
        dphis.append(emean(dphi, w))
    print("  displacement(px) -> energy-wtd mean|dphi|(rad):",
          {d: round(v, 4) for d, v in zip(dvals, dphis)})
    check("C1a static pair: dphi ~ 0", dphis[0] < 0.05, f"dphi={dphis[0]:.4f}")
    check("C1b dphi monotonically increases with displacement",
          dphis[1] < dphis[2] < dphis[3] and dphis[1] > 0.05,
          f"{[round(v,3) for v in dphis]}")
    # expected dphi for d=1 is ~2*pi/lam
    exp = 2 * math.pi / lam
    check("C1c d=1 dphi near 2*pi/wavelength", abs(dphis[1] - exp) < 0.5 * exp,
          f"got={dphis[1]:.3f} expected~{exp:.3f}")

    # ---- C2: amplitude invariance -- contrast change != motion -----------
    full = grating(wavelength=lam, shift=0.0, contrast=1.0)
    half = grating(wavelength=lam, shift=0.0, contrast=0.5)
    dphi_c, w_c, _ = motion_descriptor(ext, full, half)
    a, b = ext(full), ext(half)
    e_ratio = float(b["energy"].mean() / (a["energy"].mean() + EPS))
    dphi_contrast = emean(dphi_c, w_c)
    # compare against an actual 1px move (should be MUCH larger)
    dphi_move = emean(*motion_descriptor(ext, full, grating(wavelength=lam, shift=1.0))[:2])
    check("C2a contrast change leaves phase ~unchanged",
          dphi_contrast < 0.05, f"dphi(contrast)={dphi_contrast:.4f}")
    check("C2b a real 1px move shifts phase >> contrast change",
          dphi_move > 5 * (dphi_contrast + EPS),
          f"move={dphi_move:.3f} vs contrast={dphi_contrast:.4f}")
    check("C2c energy scales with contrast (~0.5)", abs(e_ratio - 0.5) < 0.1,
          f"energy_ratio={e_ratio:.3f}")

    # ---- C3: structure localization --------------------------------------
    blob = gaussian_blob(sigma=8.0)
    out = ext(blob)
    en = out["energy"][0, 0]
    edge_band = en[40:88, 40:88].mean()          # near blob edges/center
    corner = en[:16, :16].mean()                 # flat background corner
    check("C3 energy high on structure, low on flat bg",
          edge_band > 3 * corner, f"struct={edge_band:.4f} flat={corner:.4f}")

    print("=" * 64)
    n = sum(_results)
    print(f"{n}/{len(_results)} checks passed")
    print("=" * 64)
    sys.exit(0 if all(_results) else 1)


if __name__ == "__main__":
    main()
