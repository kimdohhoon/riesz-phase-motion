"""Feature-level rescaling at low-light s5 (paper Sec. 4.6 / Tab. 7, Supp. Sec. 3).

Each motion channel c of the corrupted validation maps is multiplied by
sigma_train[c] / sigma_test[c]: its standard deviation over the clean TRAINING set divided
by that over the whole corrupted TEST set ("global"). Because it uses test-set statistics
this is a transductive upper bound, not a deployable method. "perclip" rescales every
test clip with its own statistics. One model per seed is trained on clean data (the
paper protocol) and evaluated on every variant.

  python feature_rescaling.py --rep framediff   -> runs/rescaling_{rep}.json
Needs the full_study.py caches jester_{rep}_{train,val}_clean.pt and _val_low_light_s5.pt.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_smoke import fit, evaluate


def load(cache, rep, split, cond):
    b = torch.load(os.path.join(cache, f"jester_{rep}_{split}_{cond}.pt"),
                   map_location="cpu", weights_only=False)
    return b["app"].float(), b["maps"].float(), b["ys"]


def chan_std(maps, chunk=1024):
    """per-channel std over (N,T,h,w), streamed to bound memory."""
    C = maps.shape[2]; s = torch.zeros(C, dtype=torch.float64); ss = s.clone(); n = 0
    for i in range(0, maps.shape[0], chunk):
        x = maps[i:i + chunk].permute(2, 0, 1, 3, 4).reshape(C, -1).double()
        s += x.sum(1); ss += (x ** 2).sum(1); n += x.shape[1]
    m = s / n
    return (ss / n - m ** 2).clamp_min(0).sqrt().float()


def rescale_global(maps, tgt):
    f = (tgt / (chan_std(maps) + 1e-8)).view(1, 1, -1, 1, 1)
    return maps * f, f.flatten().tolist()


def rescale_perclip(maps, tgt):
    s = maps.permute(0, 2, 1, 3, 4).reshape(maps.shape[0], maps.shape[2], -1).std(dim=2)
    return maps * (tgt.view(1, -1) / (s + 1e-8)).view(maps.shape[0], 1, maps.shape[2], 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="phase", choices=["phase", "flow", "framediff"])
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    tr = load(a.cache, a.rep, "train", "clean")
    nc = int(tr[2].max()) + 1
    tgt = chan_std(tr[1])
    cl = load(a.cache, a.rep, "val", "clean")
    dk = load(a.cache, a.rep, "val", "low_light_s5")
    assert torch.equal(cl[2], dk[2])
    Mg, factors = rescale_global(dk[1], tgt)
    print(f"[rescale] {a.rep} global factors = {[round(v, 3) for v in factors]}", flush=True)
    conds = {"clean": cl, "dark_as_is": dk, "dark_global": (dk[0], Mg, dk[2]),
             "dark_perclip": (dk[0], rescale_perclip(dk[1], tgt), dk[2])}
    res = {k: [] for k in conds}
    for s in (0, 1, 2):
        m = fit(tr, use_motion=True, n_classes=nc, app_dim=tr[0].shape[-1], device=a.device, seed=s)
        for k, v in conds.items():
            res[k].append(float(evaluate(m, v, use_motion=True)))
        print(f"[rescale] seed {s}: " + "  ".join(f"{k}={res[k][-1]:.4f}" for k in conds), flush=True)
        del m; torch.cuda.empty_cache()
    os.makedirs("runs", exist_ok=True)
    out = {"rep": a.rep, "clean_train_chan_std": tgt.tolist(), "global_factors": factors,
           "results": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "raw": v}
                       for k, v in res.items()}}
    json.dump(out, open(f"runs/rescaling_{a.rep}.json", "w"), indent=1)
    print(f"saved runs/rescaling_{a.rep}.json")


if __name__ == "__main__":
    main()
