"""Training-free probe used to choose the read-noise level of the non-invertible control
(Supp. Sec. 3). A first setting (read noise 0.02) drove every cue to chance accuracy, so
the level was re-chosen from motion-map quality alone -- pattern (cosine to the clean map)
and amplitude retention -- before any model was evaluated on it. 0.005 was selected.

  python calibrate_noise.py --n 300 --sev 5   -> runs/calibrate_noise.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_smoke import make_dataset
from full_study import full_items
from monogenic import MonogenicExtractor
from motion_features import monogenic_maps, flow_maps, framediff_maps
from corruptions import low_light
from corruptions_noninvertible import low_light_noninvertible

SIZE, NF, OUT, NS = 112, 12, 32, 6
READ_LEVELS = [0.0, 0.002, 0.005, 0.010, 0.020]     # 0.0 = the main (noise-free) corruption
REPS = ["phase", "flow", "framediff"]


def mag(rep, m):
    if rep == "phase":
        return m[1:, 4].abs().float()
    if rep == "flow":
        return torch.sqrt(m[:, 0].float() ** 2 + m[:, 1].float() ** 2)
    return m.float().pow(2).sum(dim=1).sqrt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--sev", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    va, tr = make_dataset("jester", "val", NF, SIZE), make_dataset("jester", "train", NF, SIZE)
    _, remap = full_items(tr)
    chosen = full_items(va, remap)[0][:a.n]
    mono = MonogenicExtractor(img_size=SIZE, n_scales=NS, trainable=False).to(a.device)
    fn = {"phase": lambda c: monogenic_maps(mono, c, out_size=OUT, temporal=True),
          "flow": lambda c: flow_maps(c, out_size=OUT), "framediff": lambda c: framediff_maps(c, out_size=OUT)}
    acc = {(r, lv): {"cos": [], "amp": []} for r in REPS for lv in READ_LEVELS}
    for di, _ in chosen:
        clip = va[di][0].to(a.device)
        clean = {r: fn[r](clip) for r in REPS}
        for lv in READ_LEVELS:
            g = torch.Generator(device=a.device).manual_seed(1000 + di)
            dark = (low_light(clip, a.sev).clamp(0, 1) if lv == 0.0 else
                    low_light_noninvertible(clip, a.sev, read_noise=lv, generator=g))
            for r in REPS:
                x, y = mag(r, clean[r]).flatten(), mag(r, fn[r](dark)).flatten()
                acc[(r, lv)]["cos"].append(float(x @ y / (x.norm() * y.norm() + 1e-8)))
                acc[(r, lv)]["amp"].append(float(y.abs().mean() / (x.abs().mean() + 1e-12)))
    out = {f"{r}_{lv}": {k: float(np.mean(v)) for k, v in acc[(r, lv)].items()}
           for r in REPS for lv in READ_LEVELS}
    for lv in READ_LEVELS:
        print(f"read {lv:.3f}: " + "  ".join(f"{r} cos={out[f'{r}_{lv}']['cos']:.3f} amp={out[f'{r}_{lv}']['amp']:.3f}" for r in REPS))
    os.makedirs("runs", exist_ok=True)
    json.dump(out, open("runs/calibrate_noise.json", "w"), indent=1)


if __name__ == "__main__":
    main()
