"""Non-invertible low-light control (paper Sec. 4.6, Supp. Sec. 3).

The main corruption, corruptions.low_light, is the pointwise map y0 = x**gamma * gain with
no noise; its exact inverse recovers the input to < 1e-8. This control keeps y0 (same
gamma/gain ladder) and adds what a pointwise correction cannot undo:

    y = round(255 * clip(y0 + sqrt(SHOT_K * y0) * e1 + READ_NOISE * e2)) / 255,
    e1, e2 ~ N(0, 1)

  * shot noise: variance proportional to the darkened signal, SHOT_K = 1e-4
  * read noise: constant across severities, READ_NOISE = 0.005 (the value used in the
    paper; a first setting of 0.02 drove every cue to chance and was abandoned)
  * clipping to [0, 1] and 8-bit requantization
"""
from __future__ import annotations
import torch

from .corruptions import _LOWLIGHT_GAMMA, _LOWLIGHT_GAIN, _check

READ_NOISE = 0.005
SHOT_K = 1.0e-4
QUANT_LEVELS = 255


def low_light_noninvertible(x: torch.Tensor, severity: int,
                            read_noise: float = READ_NOISE, shot_k: float = SHOT_K,
                            generator: torch.Generator | None = None) -> torch.Tensor:
    _check(severity)
    g, gain = _LOWLIGHT_GAMMA[severity - 1], _LOWLIGHT_GAIN[severity - 1]
    y = (x.clamp(0, 1) ** g) * gain
    y = y + torch.randn(y.shape, generator=generator, device=y.device,
                        dtype=y.dtype) * (shot_k * y).clamp_min(0).sqrt()
    y = y + torch.randn(y.shape, generator=generator, device=y.device,
                        dtype=y.dtype) * read_noise
    return torch.round(y.clamp(0, 1) * QUANT_LEVELS) / QUANT_LEVELS


if __name__ == "__main__":
    from .corruptions import low_light
    from .corrections import gamma_oracle
    torch.manual_seed(0)
    x = torch.rand(4, 3, 64, 64) * 0.6 + 0.2
    for s in (1, 3, 5):
        e_main = float((gamma_oracle(low_light(x, s), s) - x).abs().mean())
        y = low_light_noninvertible(x, s)
        e_ctrl = float((gamma_oracle(y, s) - x).abs().mean())
        print(f"s{s}: oracle-inverse error  main {e_main:.1e}  |  non-invertible {e_ctrl:.1e}"
              f"  | occupied 8-bit levels {len(torch.unique(y))}")
