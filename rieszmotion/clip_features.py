"""Frozen CLIP appearance stream.

The appearance pathway of the two-stream design: a frozen CLIP image encoder turns
the middle frame of the clip into a semantic vector, which is concatenated with the
motion embedding and fed to a small head. Default model: ViT-L/14-336; the backbone
study (backbone_b32.py) uses ViT-B/32. Nothing here is trained.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPAppearance(nn.Module):
    def __init__(self, model_id: str = "openai/clip-vit-large-patch14-336",
                 device: str = "cuda", dtype: str = "float32"):
        super().__init__()
        from transformers import CLIPModel
        td = getattr(torch, dtype)
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=td).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.image_size = self.model.config.vision_config.image_size
        self.dim = self.model.config.projection_dim
        self.device, self.tdtype = device, td
        self.register_buffer("mean", torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_CLIP_STD).view(1, 3, 1, 1))
        self.to(device)        # move buffers (mean/std) onto the device too

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B,3,H,W) in [0,1] -> (B, dim) image embeddings."""
        x = imgs.to(self.device)
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bicubic", align_corners=False)
        x = (x - self.mean) / self.std
        out = self.model.get_image_features(pixel_values=x.to(self.tdtype))
        if torch.is_tensor(out):
            return out
        # transformers 5.x may wrap the output in an object
        emb = getattr(out, "image_embeds", None)
        return emb if emb is not None else out.pooler_output

    @torch.no_grad()
    def mid_frame(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (T,3,H,W) -> (dim,) embedding of the middle frame."""
        return self.forward(frames[len(frames) // 2:len(frames) // 2 + 1])[0]


if __name__ == "__main__":
    # sanity: same image -> cosine 1.0; different content -> lower
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = CLIPAppearance(device=dev)
    print(f"loaded CLIP dim={enc.dim} image_size={enc.image_size} on {dev}")
    torch.manual_seed(0)
    a = torch.rand(1, 3, 256, 256)
    b = torch.rand(1, 3, 256, 256)
    ea, eb = enc(a), enc(b)
    cos_same = F.cosine_similarity(ea, enc(a)).item()
    cos_diff = F.cosine_similarity(ea, eb).item()
    print(f"cos(same)={cos_same:.4f}  cos(diff)={cos_diff:.4f}")
    ok = cos_same > 0.999 and cos_diff < cos_same
    print(f"[{'PASS' if ok else 'FAIL'}] CLIP appearance stream sane")
