"""PARALLEL feature extractor — fixes the CPU bottleneck of full_study.py.

The serial loop decoded each clip 3x (once per motion) on a single core. Here a
DataLoader with many workers decodes each clip ONCE (in parallel across cores) and
computes the CPU motions (flow, framediff) in the worker; the main process computes
the GPU work (CLIP appearance + monogenic phase). It writes the SAME cache files
`full_study.py` reads (cache/<ds>_<motion>_<split>_<cond>.pt = {app,maps,ys}) and
skips any that already exist, so it composes with in-flight full_study caches.

Job groups (run one per GPU):
  CUDA_VISIBLE_DEVICES=0 python -m experiments.extract_fast jester A   # train/clean + val/clean
  CUDA_VISIBLE_DEVICES=1 python -m experiments.extract_fast jester B   # 9 corrupted val conditions
"""
from __future__ import annotations
import os
import sys
import time
import torch
from torch.utils.data import Dataset, DataLoader

from rieszmotion.train import make_dataset
from experiments.full_study import full_items, CACHE, OUT, SIZE, NF, NS
from rieszmotion.clip_features import CLIPAppearance
from rieszmotion.monogenic import MonogenicExtractor
from rieszmotion.motion_features import monogenic_maps, flow_maps, framediff_maps
from rieszmotion.corruptions import apply as apply_corrupt

DEVICE = "cuda"
NUM_WORKERS = 24                      # of 72 cores; 2 GPUs -> 48 workers total
MOTIONS = ("phase", "flow", "framediff")

# eval conditions per group
GROUPS = {
    "A": [("train", "clean", None), ("val", "clean", None)],
    "B": [("val", f"{lbl}_s{s}", (cn, s))
          for lbl, cn in [("low_light", "low_light"), ("noise", "gaussian_noise"),
                          ("blur", "motion_blur")]
          for s in (1, 3, 5)],
}


class DecodeDS(Dataset):
    """Worker-side: decode once, corrupt, compute CPU motions (flow, framediff)."""

    def __init__(self, base_ds, chosen, corrupt):
        self.base, self.chosen, self.corrupt = base_ds, chosen, corrupt

    def __len__(self):
        return len(self.chosen)

    def __getitem__(self, i):
        di, label = self.chosen[i]
        clip, _ = self.base[di]                          # CPU decode (parallel)
        if self.corrupt is not None:
            clip = apply_corrupt(self.corrupt[0], clip, self.corrupt[1]).clamp(0, 1)
        flow = flow_maps(clip, out_size=OUT)             # CPU (Farneback)
        fd = framediff_maps(clip, out_size=OUT)          # CPU
        return clip, flow, fd, label


def _collate(batch):
    return batch                                          # list of tuples, no stacking


def need(ds, split, cond):
    return any(not os.path.exists(os.path.join(CACHE, f"{ds}_{m}_{split}_{cond}.pt"))
               for m in MOTIONS)


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "jester"
    group = sys.argv[2] if len(sys.argv) > 2 else "A"
    jobs = GROUPS[group]
    print(f"[fast] dataset={dataset} group={group} jobs={[ (s,c) for s,c,_ in jobs]}",
          flush=True)

    clip_enc = CLIPAppearance(device=DEVICE, dtype="float32")
    mono = MonogenicExtractor(img_size=SIZE, n_scales=NS, trainable=False).to(DEVICE)
    ds_cache = {}

    for split, cond, corrupt in jobs:
        if not need(dataset, split, cond):
            print(f"[fast] skip {split}/{cond} (all motions cached)", flush=True)
            continue
        if split not in ds_cache:
            base = make_dataset(dataset, split, NF, SIZE)
            chosen, _ = full_items(base) if split == "train" else \
                full_items(base, full_items(make_dataset(dataset, "train", NF, SIZE))[1])
            ds_cache[split] = (base, chosen)
        base, chosen = ds_cache[split]
        loader = DataLoader(DecodeDS(base, chosen, corrupt), batch_size=8,
                            num_workers=NUM_WORKERS, collate_fn=_collate,
                            prefetch_factor=2, persistent_workers=False)
        apps, ph, fl, fd, ys = [], [], [], [], []
        t0, n = time.time(), 0
        for batch in loader:
            for clip, flow, framed, label in batch:
                clip = clip.to(DEVICE)
                with torch.no_grad():
                    apps.append(clip_enc.mid_frame(clip).cpu())
                    ph.append(monogenic_maps(mono, clip, out_size=OUT,
                                             temporal=True).cpu())
                # clone worker tensors off shared memory: keeping 100k+ shared-mem
                # storages alive would exceed vm.max_map_count (65530).
                fl.append(flow.clone()); fd.append(framed.clone()); ys.append(label)
                n += 1
            if n % 4000 < 8:
                rate = n / (time.time() - t0 + 1e-9)
                print(f"[fast] {split}/{cond}: {n}/{len(chosen)} "
                      f"({rate:.1f} clip/s, eta {(len(chosen)-n)/rate/60:.0f}min)",
                      flush=True)
        app_t = torch.stack(apps); ys_t = torch.tensor(ys, dtype=torch.long)
        maps = {"phase": torch.stack(ph), "flow": torch.stack(fl),
                "framediff": torch.stack(fd)}
        for m in MOTIONS:
            p = os.path.join(CACHE, f"{dataset}_{m}_{split}_{cond}.pt")
            if not os.path.exists(p):
                torch.save({"app": app_t.half(), "maps": maps[m].half(), "ys": ys_t}, p)
        print(f"[fast] DONE {split}/{cond}: {n} clips in "
              f"{(time.time()-t0)/60:.1f}min -> cached 3 motions", flush=True)


if __name__ == "__main__":
    main()
