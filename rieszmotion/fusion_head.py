"""Two-stream late-fusion head (the ONLY trained part).

Frozen CLIP appearance vector + (training-free) motion descriptor are
concatenated and classified by a small MLP -- mirrors MoCLIP-Lite's protocol
so the comparison stays controlled (only the motion cue changes, the head
stays identical across motion variants).

motion_dim=0 gives the appearance-only baseline.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TwoStreamHead(nn.Module):
    def __init__(self, appearance_dim: int, motion_dim: int, n_classes: int,
                 hidden: int = 512, dropout: float = 0.5):
        super().__init__()
        self.appearance_dim = appearance_dim
        self.motion_dim = motion_dim
        in_dim = appearance_dim + motion_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, appearance: torch.Tensor,
                motion: torch.Tensor | None = None) -> torch.Tensor:
        if self.motion_dim == 0 or motion is None:
            x = appearance
        else:
            x = torch.cat([appearance, motion], dim=-1)
        return self.net(x)

    @property
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    B, APP, MOT, NC = 4, 768, 22, 27
    res = []

    head = TwoStreamHead(APP, MOT, NC)
    out = head(torch.randn(B, APP), torch.randn(B, MOT))
    res.append(("fusion forward -> (B, n_classes)", out.shape == (B, NC),
                f"{tuple(out.shape)}, trainable={head.n_trainable/1e6:.2f}M"))

    # appearance-only baseline
    head0 = TwoStreamHead(APP, 0, NC)
    out0 = head0(torch.randn(B, APP))
    res.append(("appearance-only baseline works", out0.shape == (B, NC),
                f"{tuple(out0.shape)}"))

    # a gradient step actually updates the head
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    before = head.net[0].weight.detach().clone()
    loss = nn.functional.cross_entropy(
        head(torch.randn(B, APP), torch.randn(B, MOT)),
        torch.randint(0, NC, (B,)))
    loss.backward(); opt.step()
    moved = not torch.allclose(before, head.net[0].weight)
    res.append(("one optimizer step updates weights", moved, f"loss={loss.item():.3f}"))

    n = sum(r[1] for r in res)
    print("=" * 56)
    for name, ok, d in res:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  --  {d}")
    print(f"{n}/{len(res)} checks passed")
    print("=" * 56)
    import sys
    sys.exit(0 if n == len(res) else 1)
