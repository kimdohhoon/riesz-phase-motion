"""Video corruption suite for the robustness axis (paper Sec. 4.1, 4.4).

Three corruptions, 5 severities each, applied per-frame to a clip tensor
(T,C,H,W) or (C,H,W) in [0,1]:
  low_light       -- gamma darkening + gain (illumination drop)
  gaussian_noise  -- additive Gaussian
  motion_blur     -- directional (horizontal) blur kernel

low_light/noise are regimes where phase SHOULD stay robust (amplitude-invariant);
motion_blur is phase's WEAK regime (local phase breaks) -- both kept on purpose
so the regime analysis can draw the boundary.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

_LOWLIGHT_GAMMA = [1.5, 2.0, 3.0, 4.0, 5.0]
_LOWLIGHT_GAIN = [0.6, 0.45, 0.3, 0.2, 0.12]
_NOISE_STD = [0.04, 0.08, 0.12, 0.18, 0.26]
_BLUR_K = [3, 5, 9, 13, 17]


def _check(severity):
    if not 1 <= severity <= 5:
        raise ValueError(f"severity must be 1..5, got {severity}")


def low_light(x: torch.Tensor, severity: int) -> torch.Tensor:
    _check(severity)
    g, gain = _LOWLIGHT_GAMMA[severity - 1], _LOWLIGHT_GAIN[severity - 1]
    return (x.clamp(0, 1) ** g) * gain


def gaussian_noise(x: torch.Tensor, severity: int,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    _check(severity)
    std = _NOISE_STD[severity - 1]
    noise = torch.randn(x.shape, generator=generator, device=x.device,
                        dtype=x.dtype) * std
    return (x + noise).clamp(0, 1)


def motion_blur(x: torch.Tensor, severity: int) -> torch.Tensor:
    _check(severity)
    k = _BLUR_K[severity - 1]
    kernel = torch.zeros(k, k, device=x.device, dtype=x.dtype)
    kernel[k // 2, :] = 1.0 / k                  # horizontal motion
    squeeze = x.dim() == 3
    if squeeze:
        x = x.unsqueeze(0)                       # (1,C,H,W)
    flat = x.reshape(-1, 1, x.shape[-2], x.shape[-1])   # (N*C,1,H,W)
    w = kernel.view(1, 1, k, k)
    out = F.conv2d(flat, w, padding=k // 2).reshape(x.shape)
    return out.squeeze(0) if squeeze else out


CORRUPTIONS = {"low_light": low_light, "gaussian_noise": gaussian_noise,
               "motion_blur": motion_blur}


def apply(name: str, x: torch.Tensor, severity: int, **kw) -> torch.Tensor:
    return CORRUPTIONS[name](x, severity, **kw)


if __name__ == "__main__":
    torch.manual_seed(0)
    res = []
    img = torch.rand(4, 3, 64, 64)               # clip (T,C,H,W)

    # low_light darkens monotonically
    means = [img.mean().item()] + [low_light(img, s).mean().item() for s in range(1, 6)]
    ok = all(means[i] > means[i + 1] for i in range(len(means) - 1))
    res.append(("low_light darkens with severity", ok, f"means={[round(m,3) for m in means]}"))

    # noise increases variance
    base_v = img.var().item()
    vs = [gaussian_noise(img, s).var().item() for s in range(1, 6)]
    res.append(("noise variance grows", vs[0] > base_v and vs[-1] > vs[0],
                f"base={base_v:.4f} s1={vs[0]:.4f} s5={vs[-1]:.4f}"))

    # motion blur reduces sharpness (laplacian variance) & preserves shape
    def lap_var(t):
        lap = t[..., 1:, :] - t[..., :-1, :]
        return lap.var().item()
    sharp = lap_var(img)
    blurred = motion_blur(img, 5)
    res.append(("motion_blur reduces sharpness", lap_var(blurred) < sharp,
                f"sharp={sharp:.4f} blurred={lap_var(blurred):.4f}"))
    res.append(("motion_blur preserves shape", blurred.shape == img.shape,
                f"{tuple(blurred.shape)}"))

    n = sum(r[1] for r in res)
    print("=" * 56)
    for name, ok, d in res:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  --  {d}")
    print(f"{n}/{len(res)} checks passed")
    print("=" * 56)
    import sys
    sys.exit(0 if n == len(res) else 1)
