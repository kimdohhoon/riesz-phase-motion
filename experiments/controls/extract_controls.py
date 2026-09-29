"""Extract validation features for the test-time corrections and the non-invertible
control (paper Sec. 4.6 / Tab. 7, Supp. Tab. 7).

Pipeline per clip:  load -> corrupt -> [correction] -> {CLIP middle frame, phase/flow/framediff maps}
All three motion representations are computed in one pass. Output layout matches the
full_study.py caches, so eval_controls.py can reuse the clean-train caches unchanged.

  --corruption main           x**gamma * gain (corruptions.low_light)
               noninvertible  + shot/read noise, clipping, 8-bit (corruptions_noninvertible)
  --arm none | gamma | oracle  no correction / mean-based gamma / exact inverse of `main`

Writes cache/{ds}_{rep}_val_{tag}.pt, tag = ll{sev}_{arm} (main) or ll{sev}r_{arm} (noninvertible).

Examples
  python -m experiments.controls.extract_controls --ds jester --corruption main --sev 5 --arm gamma
  python -m experiments.controls.extract_controls --ds ipn --corruption noninvertible --sev 3 --arm none
"""
from __future__ import annotations
import argparse, os, sys, time
import torch
from rieszmotion.train import make_dataset
from experiments.full_study import full_items
from rieszmotion.clip_features import CLIPAppearance
from rieszmotion.monogenic import MonogenicExtractor
from rieszmotion.motion_features import monogenic_maps, flow_maps, framediff_maps
from rieszmotion.corruptions import low_light
from rieszmotion.corruptions_noninvertible import low_light_noninvertible
from rieszmotion.corrections import gamma_mean_based, gamma_oracle, CLEAN_MEAN

SIZE, NF, OUT, NS = 112, 12, 32, 6
REPS = ["phase", "flow", "framediff"]
CLIP_ID = os.environ.get("CLIP_MODEL", "openai/clip-vit-large-patch14-336")


def tag(a):
    return f"ll{a.sev}{'r' if a.corruption == 'noninvertible' else ''}_{a.arm}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="jester", choices=["jester", "ipn"])
    ap.add_argument("--corruption", default="main", choices=["main", "noninvertible"])
    ap.add_argument("--sev", type=int, default=5)
    ap.add_argument("--arm", default="none", choices=["none", "gamma", "oracle"])
    ap.add_argument("--limit", type=int, default=0, help="first N clips only (smoke test)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default="cache")
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)

    va, tr = make_dataset(a.ds, "val", NF, SIZE), make_dataset(a.ds, "train", NF, SIZE)
    _, remap = full_items(tr)                         # same label mapping as full_study.py
    chosen, _ = full_items(va, remap)
    if a.limit:
        chosen = chosen[:a.limit]
    print(f"[controls] {a.ds} {tag(a)}  clips={len(chosen)}", flush=True)

    enc = CLIPAppearance(model_id=CLIP_ID, device=a.device)
    mono = MonogenicExtractor(img_size=SIZE, n_scales=NS, trainable=False).to(a.device)
    apps, maps, ys, t0 = [], {r: [] for r in REPS}, [], time.time()
    for n, (di, nl) in enumerate(chosen):
        clip = va[di][0].to(a.device)
        clip = (low_light(clip, a.sev) if a.corruption == "main"
                else low_light_noninvertible(clip, a.sev)).clamp(0, 1)
        if a.arm == "gamma":
            clip = gamma_mean_based(clip, CLEAN_MEAN[a.ds])
        elif a.arm == "oracle":
            clip = gamma_oracle(clip, a.sev)
        with torch.no_grad():
            apps.append(enc.mid_frame(clip).cpu())
            maps["phase"].append(monogenic_maps(mono, clip, out_size=OUT, temporal=True).cpu())
            maps["flow"].append(flow_maps(clip, out_size=OUT).cpu())
            maps["framediff"].append(framediff_maps(clip, out_size=OUT).cpu())
        ys.append(nl)
        if (n + 1) % 1000 == 0:
            print(f"  {n+1}/{len(chosen)}  {(time.time()-t0)/60:.1f} min", flush=True)
    A, Y = torch.stack(apps).half(), torch.tensor(ys, dtype=torch.long)
    for r in REPS:
        p = os.path.join(a.cache, f"{a.ds}_{r}_val_{tag(a)}.pt")
        torch.save({"app": A, "maps": torch.stack(maps[r]).half(), "ys": Y}, p)
        print(f"[controls] saved {p}", flush=True)


if __name__ == "__main__":
    main()
