"""Aggregate full_study.py json outputs (phase/flow/framediff) for one dataset
into (1) markdown tables for the paper and (2) regenerated figures.

  python -m experiments.aggregate_full jester

Emits:
  runs/full_<dataset>_summary.md
  runs/lowlight_<dataset>_full.png      (low-light robustness curve, 3 seeds)
  runs/regime_<dataset>_full.png        (regime bars @ severity 5)
"""
from __future__ import annotations
import sys
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MOTIONS = ["phase", "flow", "framediff"]
PRETTY = {"phase": "Riesz-phase (ours)", "flow": "optical flow", "framediff": "frame-diff"}
LL = ["clean", "low_light_s1", "low_light_s3", "low_light_s5"]
SEV = [0, 1, 3, 5]


def load(dataset):
    d = {}
    for m in MOTIONS:
        p = f"runs/full_{dataset}_{m}.json"
        if os.path.exists(p):
            with open(p) as f:
                d[m] = json.load(f)
    return d


def cell(j, cond):
    c = j["conditions"][cond]
    return c["mean"], c["std"]


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "jester"
    d = load(dataset)
    have = [m for m in MOTIONS if m in d]
    if not have:
        print("no json found"); return
    ref = d[have[0]]
    n_tr, n_va, nc = ref["n_train"], ref["n_val"], ref["n_classes"]

    lines = []
    lines.append(f"# Full-scale results: {dataset} ({nc} classes, "
                 f"{n_tr} train / {n_va} val)\n")

    # contribution (from phase)
    if "phase" in d:
        c = d["phase"]["contribution"]
        lines.append("## Contribution (clean, 3 seeds)\n")
        lines.append("| config | acc |")
        lines.append("|---|---|")
        lines.append(f"| appearance-only | {c['app']['mean']:.3f} ± {c['app']['std']:.3f} |")
        lines.append(f"| motion-only (phase) | {c['motion']['mean']:.3f} ± {c['motion']['std']:.3f} |")
        lines.append(f"| **appearance + phase** | **{c['fused']['mean']:.3f} ± {c['fused']['std']:.3f}** |\n")

    # clean controlled comparison
    lines.append("## Clean controlled (fused, swap motion only)\n")
    lines.append("| motion | clean acc |")
    lines.append("|---|---|")
    for m in have:
        mu, sd = cell(d[m], "clean")
        lines.append(f"| {PRETTY[m]} | {mu:.3f} ± {sd:.3f} |")
    lines.append("")

    # low-light sweep
    lines.append("## Low-light robustness (fused, 3 seeds)\n")
    lines.append("| severity | " + " | ".join(PRETTY[m] for m in have) + " |")
    lines.append("|" + "---|" * (len(have) + 1))
    for cond in LL:
        row = [cond]
        for m in have:
            mu, sd = cell(d[m], cond)
            row.append(f"{mu:.3f} ± {sd:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # full corruption sweep
    lines.append("## Full corruption sweep (fused, 3 seeds)\n")
    conds = list(ref["conditions"].keys())
    lines.append("| condition | " + " | ".join(PRETTY[m] for m in have) + " |")
    lines.append("|" + "---|" * (len(have) + 1))
    for cond in conds:
        row = [cond]
        for m in have:
            mu, sd = cell(d[m], cond)
            row.append(f"{mu:.3f}±{sd:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    # relative drop at s5
    lines.append("\n### Relative drop vs clean (%)\n")
    drop_conds = [c for c in conds if c != "clean"]
    lines.append("| motion | " + " | ".join(drop_conds) + " |")
    lines.append("|" + "---|" * (len(drop_conds) + 1))
    for m in have:
        clean = d[m]["conditions"]["clean"]["mean"]
        row = [PRETTY[m]]
        for c in drop_conds:
            v = d[m]["conditions"][c]["mean"]
            row.append(f"{(clean - v) / clean * 100:.0f}%")
        lines.append("| " + " | ".join(row) + " |")

    md = "\n".join(lines)
    mdpath = f"runs/full_{dataset}_summary.md"
    with open(mdpath, "w") as f:
        f.write(md + "\n")
    print(md)
    print(f"\n[agg] saved -> {mdpath}")

    # ---- figure 1: low-light robustness curve ----
    plt.figure(figsize=(6, 4.2))
    for m in have:
        ys = [d[m]["conditions"][c]["mean"] for c in LL]
        es = [d[m]["conditions"][c]["std"] for c in LL]
        plt.errorbar(SEV, ys, yerr=es, marker="o", capsize=3, label=PRETTY[m])
    plt.xlabel("low-light severity (0 = clean)")
    plt.ylabel("accuracy (appearance + motion)")
    plt.title(f"{dataset.upper()} {nc}-class (FULL: {n_va} val): low-light robustness")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    p1 = f"runs/lowlight_{dataset}_full.png"
    plt.savefig(p1, dpi=130); plt.close()
    print(f"[agg] saved -> {p1}")

    # ---- figure 2: regime bars @ severity 5 ----
    regimes = [("low_light_s3", "low-light"), ("noise_s3", "noise"), ("blur_s3", "blur")]
    import numpy as np
    x = np.arange(len(regimes)); w = 0.25
    plt.figure(figsize=(6.5, 4.2))
    for i, m in enumerate(have):
        ys = [d[m]["conditions"][c]["mean"] for c, _ in regimes]
        plt.bar(x + (i - 1) * w, ys, w, label=PRETTY[m])
    plt.xticks(x, [lbl for _, lbl in regimes])
    plt.ylabel("accuracy @ severity 3")
    plt.title(f"{dataset.upper()} (FULL): regime map @ severity 3")
    plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    p2 = f"runs/regime_{dataset}_full.png"
    plt.savefig(p2, dpi=130); plt.close()
    print(f"[agg] saved -> {p2}")


if __name__ == "__main__":
    main()
