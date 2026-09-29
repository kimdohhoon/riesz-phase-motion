"""Test-time pixel corrections for the low-light study (paper Sec. 4.6, Supp. Sec. 3).

Both operate on a clip tensor (T,3,H,W) in [0,1] and return the same shape and range.

  gamma_mean_based : the "mean-based gamma correction" of the paper. For a clip of mean
                     Rec.601 luminance m, return x**g with g = ln(t) / ln(m), the exponent
                     that maps m itself to the target t. t is a FIXED constant, the mean
                     luminance of the clean training set. The correction uses only the
                     test clip and this constant -- no corruption parameters, clean
                     counterpart, or statistics of other test clips -- so it is deployable.
                     It is an approximation: mean(x**g) != m**g, so the corrected mean does
                     not equal t (see measure_gamma_mean.py).
  gamma_oracle     : the exact analytic inverse of corruptions.low_light, given the true
                     (gamma, gain). Used only to show that the main corruption is
                     invertible (and the non-invertible control is not).
"""
from __future__ import annotations
import numpy as np
import torch

from .corruptions import _LOWLIGHT_GAMMA, _LOWLIGHT_GAIN

# mean Rec.601 luminance of the clean TRAINING split (measured once, see measure_gamma_mean.py)
CLEAN_MEAN = {"jester": 0.4089, "ipn": 0.4700}
_EPS = 1e-6


def luma(x: torch.Tensor) -> torch.Tensor:
    """(T,3,H,W) -> (T,H,W) Rec.601 luminance."""
    w = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype)
    return (x * w.view(1, 3, 1, 1)).sum(1)


def gamma_mean_based(clip: torch.Tensor, target_mean: float) -> torch.Tensor:
    m = float(luma(clip).mean().clamp(_EPS, 1 - _EPS))
    t = min(max(target_mean, _EPS), 1 - _EPS)
    g = np.log(t) / np.log(m)                      # m ** g == t
    return clip.clamp(0, 1).pow(float(g)).clamp(0, 1)


def gamma_oracle(clip: torch.Tensor, severity: int) -> torch.Tensor:
    g, gain = _LOWLIGHT_GAMMA[severity - 1], _LOWLIGHT_GAIN[severity - 1]
    return (clip.clamp(0, 1) / gain).clamp(0, 1).pow(1.0 / g).clamp(0, 1)


if __name__ == "__main__":
    from .corruptions import low_light
    torch.manual_seed(0)
    x = torch.rand(4, 3, 64, 64) * 0.6 + 0.2
    for s in (1, 3, 5):
        y = low_light(x, s)
        c = gamma_mean_based(y, CLEAN_MEAN["jester"])
        o = gamma_oracle(y, s)
        print(f"s{s}: dark mean {float(luma(y).mean()):.3f} -> corrected {float(luma(c).mean()):.3f} "
              f"(target {CLEAN_MEAN['jester']}) | oracle inverse error {float((o - x).abs().mean()):.1e}")
