"""Training-free motion representations (paper Sec. 3.2 and 3.4).

Map sequences fed to the temporal CNN (the model of the paper):
  monogenic_maps  Riesz phase: [phi, E, sin(theta), cos(theta), dphi, dphi*cos, dphi*sin],
                  T maps (the first frame's dphi channels are zero-padded)
  flow_maps       Farneback optical flow (u, v), T-1 frame-pair maps
  raft_maps       RAFT-large optical flow (u, v), T-1 frame-pair maps
  framediff_maps  RGB frame difference, T-1 frame-pair maps
  *_aug           7-channel flow / frame-difference inputs for the channel-matched control

MonogenicMotionFeature is the earlier globally pooled phase descriptor, kept for the
"global pooling" row of the design ablation (paper Tab. 2).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from monogenic import MonogenicExtractor, temporal_phase_difference

EPS = 1e-6


@torch.no_grad()
def monogenic_maps(extractor: "MonogenicExtractor", frames: torch.Tensor,
                   out_size: int = 32, temporal: bool = False) -> torch.Tensor:
    """frames (T,C,H,W) -> (T, Cout, out, out) channel maps WITHOUT pooling.

    Static channels (always): [phase, energy, sin(orient), cos(orient)] -> 4ch.
    temporal=True adds the EXPLICIT motion cue (phase constancy) so phase is
    compared fairly vs flow/framediff (which hand the CNN motion directly):
      + [dphi (wrapped temporal phase diff), dphi*cos(orient), dphi*sin(orient)]
      -> 7ch.  (first frame's temporal channels are zero-padded.)
    """
    import torch.nn.functional as F
    feats = [extractor(frames[t:t + 1]) for t in range(frames.shape[0])]
    phase = torch.cat([o["phase"] for o in feats], 0)          # (T,1,H,W)
    energy = torch.cat([o["energy"] for o in feats], 0)
    orient = torch.cat([o["orientation"] for o in feats], 0)
    chans = [phase, energy, torch.sin(orient), torch.cos(orient)]
    if temporal:
        d = phase[1:] - phase[:-1]
        dphi = torch.atan2(torch.sin(d), torch.cos(d))        # wrapped (T-1,1,H,W)
        dphi = torch.cat([torch.zeros_like(dphi[:1]), dphi], 0)   # pad -> (T,1,H,W)
        chans += [dphi, dphi * torch.cos(orient), dphi * torch.sin(orient)]
    full = torch.cat(chans, dim=1)                            # (T,Cout,H,W)
    return F.interpolate(full, size=(out_size, out_size), mode="bilinear",
                         align_corners=False)


@torch.no_grad()
def flow_maps(frames: torch.Tensor, out_size: int = 32) -> torch.Tensor:
    """Optical-flow baseline. frames (T,C,H,W)[0,1] -> (T-1, 2, out, out) [u,v].

    Farneback dense flow (OpenCV). Same controlled slot as monogenic_maps so the
    temporal CNN compares motion REPRESENTATIONS under identical everything-else.
    """
    import cv2
    import numpy as np
    import torch.nn.functional as F
    g = (frames.mean(1).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()   # (T,H,W)
    outs = []
    for t in range(len(g) - 1):
        flow = cv2.calcOpticalFlowFarneback(g[t], g[t + 1], None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)       # (H,W,2)
        m = torch.from_numpy(flow).permute(2, 0, 1).float().unsqueeze(0)    # (1,2,H,W)
        outs.append(F.interpolate(m, size=(out_size, out_size),
                                  mode="bilinear", align_corners=False)[0])
    return torch.stack(outs)                                               # (T-1,2,out,out)


_RAFT = {}


@torch.no_grad()
def raft_maps(frames: torch.Tensor, out_size: int = 32) -> torch.Tensor:
    """Modern LEARNED optical-flow baseline (RAFT-large, torchvision pretrained).
    frames (T,C,H,W)[0,1] -> (T-1, 2, out, out). Same controlled slot as flow_maps;
    answers 'why not a modern flow baseline?'. RAFT is still brightness-based, so
    this tests whether phase's low-light advantage holds against SOTA flow."""
    import torch.nn.functional as F
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    dev = frames.device
    if "m" not in _RAFT:
        w = Raft_Large_Weights.DEFAULT
        _RAFT["m"] = raft_large(weights=w, progress=False).to(dev).eval()
        _RAFT["tf"] = w.transforms()
    model, tf = _RAFT["m"], _RAFT["tf"]
    # RAFT needs >=128px (feature maps /8 must be >=16); our clips are 112 -> upsample
    fr = F.interpolate(frames, size=(128, 128), mode="bilinear", align_corners=False)
    a, b = fr[:-1].contiguous(), fr[1:].contiguous()            # [0,1]
    a, b = tf(a, b)                                              # normalize + /8 resize
    flow = model(a.to(dev), b.to(dev))[-1]                      # (T-1,2,H',W')
    return F.interpolate(flow, size=(out_size, out_size), mode="bilinear",
                         align_corners=False)


@torch.no_grad()
def framediff_maps(frames: torch.Tensor, out_size: int = 32) -> torch.Tensor:
    """Simplest motion baseline: RGB frame difference. -> (T-1, C, out, out)."""
    import torch.nn.functional as F
    d = frames[1:] - frames[:-1]                                           # (T-1,C,H,W)
    return F.interpolate(d, size=(out_size, out_size), mode="bilinear",
                         align_corners=False)


def _wrap0(x):
    """temporal difference with zero-padded first frame (matches Delta-phi padding)."""
    return torch.cat([torch.zeros_like(x[:1]), x[1:] - x[:-1]], 0)


@torch.no_grad()
def flow_maps_aug(frames: torch.Tensor, out_size: int = 32) -> torch.Tensor:
    """Channel-matched flow control: give flow the SAME explicit
    orientation + temporal-difference treatment phase gets. 7 channels:
    [u, v, |flow|, sin(theta), cos(theta), du, dv]."""
    f = flow_maps(frames, out_size)                       # (T-1,2,o,o)
    u, v = f[:, 0], f[:, 1]
    mag = torch.sqrt(u ** 2 + v ** 2 + 1e-8)
    th = torch.atan2(v, u)
    return torch.stack([u, v, mag, torch.sin(th), torch.cos(th),
                        _wrap0(u), _wrap0(v)], dim=1)      # (T-1,7,o,o)


@torch.no_grad()
def framediff_maps_aug(frames: torch.Tensor, out_size: int = 32) -> torch.Tensor:
    """CHANNEL-MATCHED frame-difference control: 7 channels
    [dR, dG, dB, |gray|, sin(grad), cos(grad), d(gray)/dt]."""
    import torch.nn.functional as F
    d = framediff_maps(frames, out_size)                  # (T-1,3,o,o)
    g = d.mean(1)                                         # grayscale diff (T-1,o,o)
    gx = F.pad(g[:, :, 1:] - g[:, :, :-1], (0, 1))        # spatial gradient x
    gy = F.pad(g[:, 1:, :] - g[:, :-1, :], (0, 0, 0, 1))  # spatial gradient y
    th = torch.atan2(gy, gx)
    return torch.stack([d[:, 0], d[:, 1], d[:, 2], g.abs(),
                        torch.sin(th), torch.cos(th), _wrap0(g)], dim=1)  # (T-1,7,o,o)


class MonogenicMotionFeature(nn.Module):
    """Frame sequence -> fixed-length motion descriptor (no trainable params)."""

    def __init__(self, img_size, n_scales: int = 8, n_orient_bins: int = 8,
                 stride: int = 1, **mono_kwargs):
        super().__init__()
        self.ext = MonogenicExtractor(img_size=img_size, n_scales=n_scales,
                                      trainable=False, **mono_kwargs)
        self.n_orient_bins = n_orient_bins
        self.stride = stride
        # per-pair feature dim: [mean|dphi|, std dphi, mean energy] + orient hist
        self.pair_dim = 3 + n_orient_bins
        # clip pooling = mean + std over pairs
        self.out_dim = 2 * self.pair_dim

    def _pair_stats(self, dphi, weight, orient):
        """(1,1,H,W) maps -> (pair_dim,) energy-weighted statistics."""
        w = weight.flatten()
        d = dphi.flatten().abs()
        o = orient.flatten()
        wsum = w.sum() + EPS
        mean = (w * d).sum() / wsum
        var = (w * (d - mean) ** 2).sum() / wsum
        energy = w.mean()
        # motion-weighted orientation histogram (weight = energy * |dphi|)
        mw = w * d
        bins = ((o + math.pi) / (2 * math.pi) * self.n_orient_bins)
        bins = bins.clamp(0, self.n_orient_bins - 1e-3).long()
        hist = torch.zeros(self.n_orient_bins, device=dphi.device, dtype=mean.dtype)
        hist = hist.scatter_add(0, bins, mw)
        hist = hist / (mw.sum() + EPS)
        return torch.cat([mean.view(1), var.clamp_min(0).sqrt().view(1),
                          energy.view(1), hist])

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (T,C,H,W) or (B,T,C,H,W) -> (out_dim,) or (B,out_dim)."""
        if frames.dim() == 4:
            return self._one_clip(frames)
        return torch.stack([self._one_clip(c) for c in frames], 0)

    def _one_clip(self, frames: torch.Tensor) -> torch.Tensor:
        T = frames.shape[0]
        pair_feats = []
        for t in range(0, T - self.stride, self.stride):
            a = self.ext(frames[t:t + 1])
            b = self.ext(frames[t + self.stride:t + self.stride + 1])
            dphi = temporal_phase_difference(a["phase"], b["phase"])
            weight = 0.5 * (a["energy"] + b["energy"])
            pair_feats.append(self._pair_stats(dphi, weight, b["orientation"]))
        if not pair_feats:
            return torch.zeros(self.out_dim, device=frames.device)
        P = torch.stack(pair_feats, 0)                 # (n_pairs, pair_dim)
        return torch.cat([P.mean(0), P.std(0, unbiased=False)], 0)   # (out_dim,)
