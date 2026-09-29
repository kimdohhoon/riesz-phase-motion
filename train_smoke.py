"""Generalized smoke / mini-train with a TEMPORAL-CNN motion stream (SSv2/Jester).

Decode -> frozen CLIP appearance (mid frame) + monogenic phase MAPS over time
-> trained temporal CNN motion encoder -> concat -> head. Compares
appearance-only vs appearance+temporal-motion on a small class subset.

  python train_smoke.py --dataset jester --device cuda \
      --class-names "Swiping Left,Swiping Right,...,Zooming Out With Full Hand" \
      --train-per 60 --val-per 30 --size 112 --n-frames 12

Phase extraction is frozen/training-free; only the temporal CNN + head train.
Tiny subsets -> plumbing/first-signal, not final numbers.
"""
from __future__ import annotations
import argparse
import time
import collections
import torch
import torch.nn as nn

from datasets import SSv2Dataset, JesterDataset, IPNDataset, ARIDDataset
from clip_features import CLIPAppearance
from monogenic import MonogenicExtractor
from motion_features import monogenic_maps, flow_maps, framediff_maps
from motion_cnn import TemporalMotionCNN

# dataset paths come from paths.py (override via environment variables)
from paths import (SSV2_DIR, SSV2_CSV, JESTER_ROOT, JESTER_LABELS, JESTER_CSV,
                   IPN_ROOT, IPN_ANNOT, ARID_ROOT, ARID_LIST)


def load_jester_label_map(path):
    with open(path) as f:
        names = [l.strip() for l in f if l.strip()]
    return {n: i for i, n in enumerate(names)}


def make_dataset(name, split, n_frames, size):
    if name == "ssv2":
        return SSv2Dataset(SSV2_DIR, SSV2_CSV[split], n_frames=n_frames, size=size)
    if name == "jester":
        lm = load_jester_label_map(JESTER_LABELS)
        return JesterDataset(JESTER_ROOT, JESTER_CSV[split], label_map=lm,
                             n_frames=n_frames, size=size)
    if name == "ipn":
        return IPNDataset(IPN_ROOT, IPN_ANNOT[split], n_frames=n_frames, size=size)
    if name == "arid":
        return ARIDDataset(ARID_ROOT, ARID_LIST[split], n_frames=n_frames, size=size)
    raise ValueError(name)


def pick_class_subset(items, n_classes, per_class, seed=0, only_labels=None):
    by = collections.defaultdict(list)
    for idx, it in enumerate(items):
        by[it[1]].append(idx)          # it=(vid,label[,start,end]); label at [1]
    if only_labels:
        labels = [l for l in only_labels if l in by]
    else:
        labels = sorted(by, key=lambda k: (-len(by[k]), k))[:n_classes]
    remap = {o: i for i, o in enumerate(labels)}
    rng = torch.Generator().manual_seed(seed)
    chosen = []
    for o in labels:
        ix = by[o]
        sel = torch.randperm(len(ix), generator=rng)[:per_class].tolist()
        chosen += [(ix[s], remap[o]) for s in sel]
    return chosen, remap


def val_subset(items, remap, per_class):
    by = collections.defaultdict(list)
    for idx, it in enumerate(items):
        lab = it[1]
        if lab in remap:
            by[remap[lab]].append(idx)
    out = []
    for c, ix in by.items():
        out += [(i, c) for i in ix[:per_class]]
    return out


class TwoStreamModel(nn.Module):
    """frozen-appearance vec + trained temporal-CNN over phase maps."""

    def __init__(self, app_dim, n_classes, use_motion, use_app=True, in_ch=4,
                 motion_dim=64, hidden=256, dropout=0.3):
        super().__init__()
        self.use_motion = use_motion
        self.use_app = use_app
        proj = 128
        if use_app:
            # project + normalize so a 768-d appearance vec doesn't drown the
            # 64-d motion vec when the two streams are concatenated.
            self.app_proj = nn.Sequential(nn.Linear(app_dim, proj), nn.LayerNorm(proj))
        if use_motion:
            self.motion = TemporalMotionCNN(in_ch=in_ch, dim=motion_dim)
            self.motion_norm = nn.LayerNorm(motion_dim)
        in_dim = (proj if use_app else 0) + (motion_dim if use_motion else 0)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, app, maps=None):
        parts = []
        if self.use_app:
            parts.append(self.app_proj(app))
        if self.use_motion:
            mv = self.motion(maps.permute(0, 2, 1, 3, 4))   # (B,T,C,H,W)->(B,C,T,H,W)
            parts.append(self.motion_norm(mv))
        return self.head(torch.cat(parts, dim=-1))


def extract_motion(clip, motion_type, mono_ext, out_size):
    if motion_type == "phase":
        return monogenic_maps(mono_ext, clip, out_size=out_size, temporal=True)
    if motion_type == "flow":
        return flow_maps(clip, out_size=out_size)
    if motion_type == "framediff":
        return framediff_maps(clip, out_size=out_size)
    raise ValueError(motion_type)


def build_features(ds, chosen, clip_enc, motion_type, mono_ext, out_size, device,
                   tag="", corrupt=None):
    """corrupt=(name,severity) applies a corruption to each clip before feature
    extraction (for robustness eval). None = clean."""
    from corruptions import apply as apply_corrupt
    apps, mapss, ys, fails = [], [], [], 0
    for di, nl in chosen:
        try:
            clip, _ = ds[di]
        except Exception as e:
            fails += 1
            print(f"  [skip {tag}] idx={di}: {type(e).__name__}: {e}", flush=True)
            continue
        clip = clip.to(device)
        if corrupt is not None:
            clip = apply_corrupt(corrupt[0], clip, corrupt[1]).clamp(0, 1)
        with torch.no_grad():
            app = clip_enc.mid_frame(clip)
            maps = extract_motion(clip, motion_type, mono_ext, out_size)
        apps.append(app.cpu()); mapss.append(maps.cpu()); ys.append(nl)
    if not apps:
        raise RuntimeError(f"no clips decoded ({tag})")
    return torch.stack(apps), torch.stack(mapss), torch.tensor(ys, dtype=torch.long), fails


def fit(tr, use_motion, n_classes, app_dim, device, use_app=True,
        epochs=60, bs=32, seed=0):
    """Keeps features on CPU; moves only per-batch to GPU (shared-GPU safe)."""
    torch.manual_seed(seed)
    Atr, Mtr, ytr = tr                                  # CPU tensors
    model = TwoStreamModel(app_dim, n_classes, use_motion, use_app=use_app,
                           in_ch=Mtr.shape[2]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n = Atr.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            a = Atr[idx].to(device)
            m = Mtr[idx].to(device) if use_motion else None
            y = ytr[idx].to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(a, m), y)
            loss.backward(); opt.step()
    return model


def evaluate(model, va, use_motion, bs=64):
    dev = next(model.parameters()).device
    Ava, Mva, yva = va                                  # CPU tensors
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(yva), bs):
            a = Ava[i:i + bs].to(dev)
            m = Mva[i:i + bs].to(dev) if use_motion else None
            correct += (model(a, m).argmax(1).cpu() == yva[i:i + bs]).sum().item()
    return correct / len(yva)


def train_eval(tr, va, use_motion, n_classes, app_dim, device, use_app=True,
               epochs=60, bs=32, seed=0):
    model = fit(tr, use_motion, n_classes, app_dim, device, use_app, epochs, bs, seed)
    return evaluate(model, va, use_motion)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ssv2", "jester"], default="jester")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--train-per", type=int, default=50)
    ap.add_argument("--val-per", type=int, default=25)
    ap.add_argument("--n-frames", type=int, default=12)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--out-size", type=int, default=32)
    ap.add_argument("--n-scales", type=int, default=6)
    ap.add_argument("--class-names", default="")
    ap.add_argument("--motion-type", choices=["phase", "flow", "framediff"], default="phase")
    args = ap.parse_args()
    print(f"[run] dataset={args.dataset} device={args.device} frames={args.n_frames} "
          f"size={args.size} out={args.out_size}", flush=True)

    t0 = time.time()
    clip_enc = CLIPAppearance(device=args.device, dtype="float32")
    mono_ext = MonogenicExtractor(img_size=args.size, n_scales=args.n_scales,
                                  trainable=False).to(args.device)
    print(f"[run] CLIP dim={clip_enc.dim} (load {time.time()-t0:.1f}s)", flush=True)

    tr_ds = make_dataset(args.dataset, "train", args.n_frames, args.size)
    va_ds = make_dataset(args.dataset, "val", args.n_frames, args.size)
    only = None
    if args.class_names and args.dataset == "jester":
        lm = load_jester_label_map(JESTER_LABELS)
        only = [lm[n.strip()] for n in args.class_names.split(",") if n.strip() in lm]
        args.classes = len(only)
        print(f"[run] explicit classes ({len(only)}): {args.class_names}", flush=True)
    chosen_tr, remap = pick_class_subset(tr_ds.items, args.classes, args.train_per, only_labels=only)
    chosen_va = val_subset(va_ds.items, remap, args.val_per)
    print(f"[run] train={len(chosen_tr)} val={len(chosen_va)} ({len(remap)} classes)", flush=True)

    t1 = time.time()
    print(f"[run] motion_type={args.motion_type}", flush=True)
    tr = build_features(tr_ds, chosen_tr, clip_enc, args.motion_type, mono_ext, args.out_size, args.device, "train")
    va = build_features(va_ds, chosen_va, clip_enc, args.motion_type, mono_ext, args.out_size, args.device, "val")
    n_clips = len(chosen_tr) + len(chosen_va)
    print(f"[run] extracted ~{(time.time()-t1)/max(1,n_clips):.2f}s/clip fails={tr[3]+va[3]} "
          f"| App{tuple(tr[0].shape)} Maps{tuple(tr[1].shape)} "
          f"nan={bool(torch.isnan(tr[1]).any() or torch.isnan(tr[0]).any())}", flush=True)

    nc = len(remap)
    acc_app = train_eval(tr[:3], va[:3], False, nc, clip_enc.dim, args.device, use_app=True)
    acc_mot = train_eval(tr[:3], va[:3], True, nc, clip_enc.dim, args.device, use_app=False)
    acc_two = train_eval(tr[:3], va[:3], True, nc, clip_enc.dim, args.device, use_app=True)
    print("=" * 56)
    print(f"  dataset            : {args.dataset}  ({nc} classes)")
    print(f"  chance             : {1.0/nc:.3f}")
    print(f"  motion-only (CNN)  : {acc_mot:.3f}   <- key diagnostic")
    print(f"  appearance-only    : {acc_app:.3f}")
    print(f"  appearance+motion  : {acc_two:.3f}")
    print(f"  motion delta       : {acc_two-acc_app:+.3f}")
    print("=" * 56)
    print("NOTE: tiny subset -> plumbing/first-signal, not statistically final.")


if __name__ == "__main__":
    main()
