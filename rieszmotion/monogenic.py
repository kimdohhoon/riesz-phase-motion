"""Monogenic signal extractor (PyTorch) for the Riesz-phase motion study.

Implements the Mono2D recipe (Kimbowa et al., 2025, arXiv:2503.09050, §2.1):
    FFT -> Butterworth low-pass -> multi-scale Log-Gabor band-pass
        -> Riesz kernels (i*f_x - f_y)/|f|  -> IFFT
        -> local phase + phase asymmetry + local energy (amplitude).

Reference monogenic signal: Felsberg & Sommer 2001; Kovesi.
Frozen extraction by default (training-free). Trainable f0/sigma_r (Mono2D
eq. 8-9, sigmoid reparam) is an OPTIONAL extension, off by default.

Our use differs from Mono2D: Mono2D uses *spatial* phase of a single image for
domain-generalization segmentation. We take the **temporal phase difference**
between frames as a motion cue (see `temporal_phase_difference`).

Convention: grayscale input, unshifted FFT grid (matches torch.fft.fft2).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _freq_grid(h: int, w: int):
    """Unshifted radial frequency grid matching torch.fft.fft2 layout."""
    fy = torch.fft.fftfreq(h).view(h, 1).expand(h, w)   # v (rows)
    fx = torch.fft.fftfreq(w).view(1, w).expand(h, w)   # u (cols)
    radius = torch.sqrt(fx * fx + fy * fy)
    radius[0, 0] = 1.0                                   # avoid /0 at DC
    return fx, fy, radius


class MonogenicExtractor(nn.Module):
    """Frozen-by-default monogenic feature extractor.

    forward(x) -> dict with per-pixel maps (B,1,H,W):
        phase       local phase    atan2(odd_mag, even)  in [0, pi]
        orientation atan2(R2, R1)  in (-pi, pi]
        energy      sqrt(even^2 + R1^2 + R2^2)  (local amplitude)
        asym        phase asymmetry (Mono2D eq.1), in [0, ~1]
    """

    def __init__(self, img_size, n_scales: int = 8, min_wavelength: float = 4.0,
                 mult: float = 2.0, sigma_onf: float = 0.55,
                 lp_cutoff: float = 0.5, lp_order: int = 10,
                 trainable: bool = False, eps: float = 1e-6):
        super().__init__()
        h, w = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.h, self.w, self.n_scales, self.eps = h, w, n_scales, eps
        self.trainable = trainable

        fx, fy, radius = _freq_grid(h, w)
        self.register_buffer("radius", radius)
        # Riesz transform in frequency domain: (i*f_x - f_y)/|f|  (Mono2D eq.3)
        riesz = torch.complex(-fy / radius, fx / radius)     # = (i*fx - fy)/r
        riesz[0, 0] = 0
        self.register_buffer("riesz", riesz)
        # Butterworth low-pass (Mono2D: cutoff 0.5, order 10)
        lp = 1.0 / (1.0 + (radius / lp_cutoff) ** (2 * lp_order))
        self.register_buffer("lp", lp)
        self.register_buffer("log_radius", torch.log(radius))

        # Log-Gabor center frequencies (one per scale)
        wavelengths = [min_wavelength * (mult ** s) for s in range(n_scales)]
        f0 = torch.tensor([1.0 / wl for wl in wavelengths])   # (n_scales,)
        if trainable:
            # Mono2D eq.8-9: optimize unbounded params, map to valid range.
            f0_min, f0_max = 1.0 / max(h, w), 0.5
            self._f0_min, self._f0_max = f0_min, f0_max
            f0_star = torch.log((f0 - f0_min) / (f0_max - f0 + eps) + eps)  # inverse-sigmoid init
            self.f0_star = nn.Parameter(f0_star)
            self.sigma_star = nn.Parameter(torch.zeros(()))   # sigmoid(0)=0.5
        else:
            band = self._build_band(f0, sigma_onf)
            self.register_buffer("band", band)

    def _build_band(self, f0: torch.Tensor, sigma_onf: float):
        """Sum of n Log-Gabor band-pass filters x low-pass (Mono2D: responses summed)."""
        log_r = self.log_radius
        lg_sum = torch.zeros_like(self.radius)
        denom = 2.0 * (math.log(sigma_onf) ** 2)
        for fo in f0:
            lg = torch.exp(-((log_r - math.log(float(fo))) ** 2) / denom)
            lg_sum = lg_sum + lg
        band = self.lp * lg_sum
        band[0, 0] = 0
        return band

    def _current_band(self):
        if not self.trainable:
            return self.band
        f0 = self._f0_min + torch.sigmoid(self.f0_star) * (self._f0_max - self._f0_min)
        sigma_onf = torch.sigmoid(self.sigma_star).clamp(1e-3, 0.999)
        # differentiable rebuild
        log_r = self.log_radius
        denom = 2.0 * (torch.log(sigma_onf) ** 2)
        lg_sum = torch.zeros_like(self.radius)
        for fo in f0:
            lg = torch.exp(-((log_r - torch.log(fo)) ** 2) / denom)
            lg_sum = lg_sum + lg
        band = self.lp * lg_sum
        band = band.clone()
        band[0, 0] = 0
        return band

    @staticmethod
    def _to_gray(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:                      # (B,H,W)
            return x.unsqueeze(1)
        if x.size(1) == 1:
            return x
        # luminance
        w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (x * w).sum(1, keepdim=True)

    def forward(self, x: torch.Tensor) -> dict:
        x = self._to_gray(x)                  # (B,1,H,W)
        band = self._current_band()
        Fx = torch.fft.fft2(x)
        Ff = Fx * band                        # band-pass spectrum
        even = torch.fft.ifft2(Ff).real       # I_f
        odd = torch.fft.ifft2(Ff * self.riesz)   # R1 + i R2
        R1, R2 = odd.real, odd.imag
        odd_mag = torch.sqrt(R1 * R1 + R2 * R2 + self.eps)
        energy = torch.sqrt(even * even + R1 * R1 + R2 * R2 + self.eps)
        phase = torch.atan2(odd_mag, even)            # [0, pi]
        orientation = torch.atan2(R2, R1)             # (-pi, pi]
        asym = F.relu(odd_mag - even.abs()) / (energy + self.eps)
        return {"phase": phase, "orientation": orientation,
                "energy": energy, "asym": asym}


def temporal_phase_difference(phase_t: torch.Tensor, phase_t1: torch.Tensor) -> torch.Tensor:
    """Wrapped local-phase difference between two frames (phase-constancy motion cue).

    Returns dphi in (-pi, pi]; |dphi| grows ~linearly with small displacement.
    """
    d = phase_t1 - phase_t
    return torch.atan2(torch.sin(d), torch.cos(d))


def motion_descriptor(extractor: MonogenicExtractor, frame_t: torch.Tensor,
                      frame_t1: torch.Tensor):
    """Energy-weighted temporal phase-difference cue for a frame pair.

    Returns (dphi_map, weight, signed_orientation) for downstream pooling.
    weight = mean local energy of the pair (confidence on textured regions).
    """
    a = extractor(frame_t)
    b = extractor(frame_t1)
    dphi = temporal_phase_difference(a["phase"], b["phase"])
    weight = 0.5 * (a["energy"] + b["energy"])
    return dphi, weight, b["orientation"]
