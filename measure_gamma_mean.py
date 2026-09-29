"""How close does the mean-based gamma correction get to its target mean? (Supp. Sec. 3)

g = ln t / ln m maps the clip mean m itself to t, but mean(x**g) != m**g, so the
corrected mean luminance is not exactly t. This measures it on random validation clips,
and also reports the clean mean luminance (the target t is measured on the TRAINING split).

  python measure_gamma_mean.py --ds jester --n 400
"""
from __future__ import annotations
import argparse, os, random, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_smoke import make_dataset
from corruptions import low_light
from corrections import gamma_mean_based, luma, CLEAN_MEAN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="jester", choices=["jester", "ipn"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ds = make_dataset(a.ds, "val", 12, 112)
    random.seed(a.seed)
    idx = random.sample(range(len(ds)), min(a.n, len(ds)))
    t, out, clean = CLEAN_MEAN[a.ds], {1: [], 3: [], 5: []}, []
    for i in idx:
        x = ds[i][0]; clean.append(float(luma(x).mean()))
        for s in out:
            out[s].append(float(luma(gamma_mean_based(low_light(x, s).clamp(0, 1), t)).mean()))
    print(f"{a.ds}: {len(idx)} clips, target t={t}, clean val mean luma {np.mean(clean):.4f}")
    for s, v in out.items():
        v = np.array(v)
        print(f"  s{s}: corrected mean {v.mean():.3f} +/- {v.std():.3f}  ({100*(v.mean()-t)/t:+.1f}% vs t)")


if __name__ == "__main__":
    main()
