"""Train on clean data (the paper protocol) and evaluate on the correction / control
caches written by extract_controls.py (paper Tab. 7, Supp. Tab. 7).

One model per seed is trained on the clean-train cache and evaluated on every condition,
so no training is added beyond the main protocol.

  --set gamma          main corruption at s5: none, mean-based gamma, oracle inverse
  --set noninvertible  non-invertible control at s3/s5: none, mean-based gamma

  python eval_controls.py --ds jester --rep phase --set gamma
  -> runs/controls_{set}_{ds}_{rep}.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_smoke import fit, evaluate

SETS = {"gamma": ["ll5_none", "ll5_gamma", "ll5_oracle"],
        "noninvertible": ["ll3r_none", "ll3r_gamma", "ll5r_none", "ll5r_gamma"]}


def load(p):
    b = torch.load(p, map_location="cpu", weights_only=False)
    return b["app"].float(), b["maps"].float(), b["ys"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="jester", choices=["jester", "ipn"])
    ap.add_argument("--rep", default="phase", choices=["phase", "flow", "framediff"])
    ap.add_argument("--set", default="gamma", choices=list(SETS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    c = lambda tag: os.path.join(a.cache, f"{a.ds}_{a.rep}_{tag}.pt")

    tr = load(c("train_clean"))
    nc = int(tr[2].max()) + 1
    conds = {"clean": load(c("val_clean"))}
    for t in SETS[a.set]:
        if os.path.exists(c("val_" + t)):
            conds[t] = load(c("val_" + t))
        else:
            print(f"[eval] missing {c('val_' + t)} (skipped)", flush=True)
    for k, v in conds.items():                         # extraction order must match
        assert torch.equal(v[2], conds["clean"][2]), f"label order differs in {k}"

    res = {k: [] for k in conds}
    for s in [int(x) for x in a.seeds.split(",")]:
        m = fit(tr, use_motion=True, n_classes=nc, app_dim=tr[0].shape[-1],
                device=a.device, seed=s)
        for k, v in conds.items():
            res[k].append(float(evaluate(m, v, use_motion=True)))
        print(f"[eval] seed {s}: " + "  ".join(f"{k}={res[k][-1]:.4f}" for k in conds), flush=True)
        del m; torch.cuda.empty_cache()
    out = {"ds": a.ds, "rep": a.rep, "set": a.set,
           "results": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "raw": v}
                       for k, v in res.items()}}
    os.makedirs("runs", exist_ok=True)
    p = f"runs/controls_{a.set}_{a.ds}_{a.rep}.json"
    json.dump(out, open(p, "w"), indent=1)
    print("saved", p)


if __name__ == "__main__":
    main()
