"""Lightweight temporal CNN over monogenic channel-map sequences.

Input : (B, C, T, H, W)  monogenic maps over time (C=4: phase,energy,sin,cos)
Output: (B, dim)         motion feature capturing temporal dynamics (direction,
                         order) -- the info that global pooling threw away.

This is the trained motion encoder (phase extraction stays frozen). Small
enough for 2xA100 and CPU smoke. 3D conv = joint spatio-temporal so motion
direction (Swiping L vs R) can be learned.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TemporalMotionCNN(nn.Module):
    def __init__(self, in_ch: int = 4, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),                       # keep time, halve space
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),                               # halve time+space
            nn.Conv3d(64, dim, 3, padding=1), nn.BatchNorm3d(dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) -> (B, dim)."""
        return self.net(x).flatten(1)


class MotionCNNClassifier(nn.Module):
    """Motion-only classifier (for the synthetic direction sanity check)."""

    def __init__(self, n_classes: int, in_ch: int = 4, dim: int = 128):
        super().__init__()
        self.enc = TemporalMotionCNN(in_ch, dim)
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, x):
        return self.fc(self.enc(x))
