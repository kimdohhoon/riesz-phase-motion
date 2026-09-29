"""Clip dataset loaders for Jester (jpg frames) and SSv2 (webm).

Each __getitem__ returns (clip, label):
  clip  : (T, 3, H, W) float in [0,1], T frames evenly sampled
  label : int class id

SSv2 (video files) is decoded with OpenCV (FFmpeg backend).
Jester is not downloaded yet -> the loader follows the standard 20bn-jester
layout (root/<id>/<frame>.jpg) and will be sanity-checked once data arrives.
"""
from __future__ import annotations
import os
import csv
import torch
from torch.utils.data import Dataset


def _even_idx(total: int, n: int):
    if total <= 0:
        return [0] * n
    if total <= n:
        return list(range(total)) + [total - 1] * (n - total)
    step = total / n
    return [min(total - 1, int(i * step)) for i in range(n)]


def _resize_chw(t: torch.Tensor, size: int) -> torch.Tensor:
    import torch.nn.functional as F
    return F.interpolate(t.unsqueeze(0), size=(size, size),
                         mode="bilinear", align_corners=False).squeeze(0)


class SSv2Dataset(Dataset):
    """Something-Something v2 (webm). csv rows: '<id>.webm <label_id>'."""

    def __init__(self, video_dir: str, csv_path: str, n_frames: int = 8,
                 size: int = 224):
        self.video_dir, self.n_frames, self.size = video_dir, n_frames, size
        self.items = []
        with open(csv_path) as f:
            for row in f:
                parts = row.split()
                if len(parts) >= 2:
                    self.items.append((parts[0], int(parts[1])))

    def __len__(self):
        return len(self.items)

    def _decode(self, path):
        import cv2
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr[:, :, ::-1].copy())          # BGR->RGB
        cap.release()
        if not frames:
            raise RuntimeError(f"no frames decoded from {path}")
        idx = _even_idx(len(frames), self.n_frames)
        clip = torch.stack([
            torch.from_numpy(frames[i]).permute(2, 0, 1).float() / 255.0
            for i in idx])                                 # (T,3,h,w)
        return torch.stack([_resize_chw(c, self.size) for c in clip])

    def __getitem__(self, i):
        fname, label = self.items[i]
        return self._decode(os.path.join(self.video_dir, fname)), label


class JesterDataset(Dataset):
    """20bn-jester (jpg frames). csv rows: '<id>;<label_name>' or '<id> <label_id>'.

    """

    def __init__(self, frames_root: str, csv_path: str,
                 label_map: dict | None = None, n_frames: int = 8, size: int = 112):
        self.root, self.n_frames, self.size = frames_root, n_frames, size
        self.label_map = label_map or {}
        self.items = []
        with open(csv_path) as f:
            for row in f:
                row = row.strip()
                if not row:
                    continue
                vid, lab = (row.split(";", 1) + [""])[:2] if ";" in row else row.split()[:2]
                label = self.label_map.get(lab, lab)
                self.items.append((vid, int(label) if str(label).isdigit() else label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        import numpy as np
        vid, label = self.items[i]
        d = os.path.join(self.root, str(vid))
        jpgs = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
        idx = _even_idx(len(jpgs), self.n_frames)
        clip = torch.stack([
            torch.from_numpy(np.asarray(Image.open(os.path.join(d, jpgs[i])).convert("RGB")))
                 .permute(2, 0, 1).float() / 255.0
            for i in idx])
        return torch.stack([_resize_chw(c, self.size) for c in clip]), label


class IPNDataset(Dataset):
    """IPN Hand (RGB jpg frames). Continuous videos sliced into isolated gesture
    clips via the annotation's start/end frame ranges.

    Annot list rows: '<video>,<label>,<class_id>,<start>,<end>,<num_frames>'.
    Frames live at  <root>/<video>/<video>_<frame:06d>.jpg  (1-indexed; adjust
    `frame_fmt` after inspecting the extracted layout).

    drop_labels: e.g. {"D0X"} to exclude the no-gesture class. class ids are
    remapped to a contiguous 0..C-1 over the kept labels.
    """

    def __init__(self, frames_root, annot_path, n_frames=8, size=112,
                 drop_labels=("D0X",), frame_fmt="{vid}_{idx:06d}.jpg"):
        self.root, self.n_frames, self.size = frames_root, n_frames, size
        self.frame_fmt = frame_fmt
        rows = []
        with open(annot_path) as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) < 5:
                    continue
                vid, label, cid, start, end = p[0], p[1], p[2], int(p[3]), int(p[4])
                if drop_labels and label in drop_labels:
                    continue
                rows.append((vid, label, start, end))
        # remap kept labels -> 0..C-1 (stable by first appearance order, sorted)
        labels = sorted({r[1] for r in rows})
        self.label_map = {l: i for i, l in enumerate(labels)}
        self.items = [(vid, self.label_map[label], start, end)
                      for vid, label, start, end in rows]

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _fnum(fname):
        # '<video>_000123.jpg' -> 123
        return int(fname.rsplit("_", 1)[1].split(".")[0])

    def _load_range(self, vid, start, end):
        from PIL import Image
        import numpy as np
        d = os.path.join(self.root, vid)
        jpgs = sorted((f for f in os.listdir(d) if f.endswith(".jpg")),
                      key=self._fnum)
        # frames are SPARSE: keep only existing files within [start, end]
        inrange = [f for f in jpgs if start <= self._fnum(f) <= end]
        if not inrange:
            inrange = jpgs                          # fallback: whole video
        idx = _even_idx(len(inrange), self.n_frames)
        fr = [torch.from_numpy(np.asarray(
                  Image.open(os.path.join(d, inrange[i])).convert("RGB")))
              .permute(2, 0, 1).float() / 255.0 for i in idx]
        clip = torch.stack(fr)
        return torch.stack([_resize_chw(c, self.size) for c in clip])

    def __getitem__(self, i):
        vid, label, start, end = self.items[i]
        return self._load_range(vid, start, end), label


class ARIDDataset(Dataset):
    """ARID -- Action Recognition in the Dark (naturally low-light .avi clips).

    List rows: '<idx>\\t<label>\\t<Class/clip.avi>' (the official list_cvt format).
    ARID is INHERENTLY dark, so no synthetic low-light is applied -- it is the
    natural-illumination test of the phase-vs-flow regime claim. Whole-body actions
    (walk/run/jump) also probe phase's large-motion (wrapping) weak regime.
    """

    def __init__(self, video_root, list_path, n_frames=8, size=112):
        self.root, self.n_frames, self.size = video_root, n_frames, size
        self.items = []
        with open(list_path) as f:
            for row in f:
                p = row.split()
                if len(p) >= 3:
                    rel = p[2].replace(".avi", ".mp4")       # v1.5 clips are .mp4
                    self.items.append((rel, int(p[1])))      # (relpath, label)

    def __len__(self):
        return len(self.items)

    def _decode(self, path):
        import cv2
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr[:, :, ::-1].copy())            # BGR->RGB
        cap.release()
        if not frames:
            raise RuntimeError(f"no frames decoded from {path}")
        idx = _even_idx(len(frames), self.n_frames)
        clip = torch.stack([
            torch.from_numpy(frames[i]).permute(2, 0, 1).float() / 255.0
            for i in idx])
        return torch.stack([_resize_chw(c, self.size) for c in clip])

    def __getitem__(self, i):
        rel, label = self.items[i]
        return self._decode(os.path.join(self.root, rel)), label


if __name__ == "__main__":
    # smoke test: decode one SSv2 clip (paths from paths.py / env vars)
    import sys
    from .paths import SSV2_DIR, SSV2_CSV
    vid_dir = SSV2_DIR
    csv_path = SSV2_CSV["val"]
    res = []
    try:
        ds = SSv2Dataset(vid_dir, csv_path, n_frames=8, size=224)
        res.append(("SSv2 csv parsed", len(ds) > 0, f"{len(ds)} clips"))
        clip, label = ds[0]
        res.append(("SSv2 webm decodes to (T,3,H,W)",
                    clip.shape == (8, 3, 224, 224), f"shape={tuple(clip.shape)} label={label}"))
        res.append(("clip values in [0,1]", 0.0 <= clip.min() and clip.max() <= 1.0,
                    f"[{clip.min():.3f},{clip.max():.3f}]"))
    except Exception as e:
        res.append((f"SSv2 decode FAILED: {type(e).__name__}: {e}", False, ""))
    print("=" * 56)
    for name, ok, d in res:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {d}" if d else ""))
    print("=" * 56)
    sys.exit(0 if all(r[1] for r in res) else 1)
