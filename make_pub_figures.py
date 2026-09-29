"""Publication-quality figures: (1) two-stream architecture diagram,
(2) polished low-light robustness curves (Jester | IPN, 2 panels).

  python make_pub_figures.py
Outputs: runs/fig_architecture.png, runs/fig_lowlight_pub.png
"""
from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FROZEN = "#cfe2f3"   # light blue  (frozen)
FROZEN_E = "#3d6fb4"
TRAIN = "#ffe0b2"    # light orange (trained)
TRAIN_E = "#e08a1e"
NEUT = "#eeeeee"
NEUT_E = "#888888"


def _box(ax, x, y, w, h, text, fc, ec, fs=11, bold=False):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=15, lw=1.6, color="#444444", zorder=1))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")

    # input
    _box(ax, 0.1, 1.7, 1.0, 1.0, "video\nclip\n$T{\\times}3{\\times}H{\\times}W$", NEUT, NEUT_E, 9)

    # appearance branch (top, frozen)
    _box(ax, 1.7, 2.55, 1.9, 1.0, "Frozen CLIP\nViT-L/14\n(appearance)", FROZEN, FROZEN_E, 10)
    _box(ax, 4.0, 2.7, 1.5, 0.7, "Linear+LN\n$\\rightarrow128$", FROZEN, FROZEN_E, 9.5)

    # motion branch (bottom): frozen extractor -> trained CNN
    _box(ax, 1.7, 0.45, 1.9, 1.05, "Riesz-phase\nextractor\n(frozen)", FROZEN, FROZEN_E, 10)
    _box(ax, 4.0, 0.5, 1.5, 0.95, "7-ch maps\n$\\phi,E,\\theta,\\Delta\\phi$", NEUT, NEUT_E, 9.5)
    _box(ax, 5.9, 0.5, 1.7, 0.95, "Temporal\n3D-CNN\n(trained)", TRAIN, TRAIN_E, 10)

    # fusion + head (trained)
    _box(ax, 7.95, 1.55, 1.0, 1.3, "concat\n+ MLP\n(trained)", TRAIN, TRAIN_E, 10)
    _box(ax, 9.05, 1.9, 0.85, 0.6, "action\nclass", NEUT, NEUT_E, 9.5)

    # arrows
    _arrow(ax, 1.1, 2.4, 1.7, 3.05)         # input -> CLIP
    _arrow(ax, 1.1, 2.0, 1.7, 0.97)         # input -> Riesz
    _arrow(ax, 3.6, 3.05, 4.0, 3.05)        # CLIP -> proj
    _arrow(ax, 5.5, 3.05, 8.0, 2.85)        # proj -> fusion
    _arrow(ax, 3.6, 0.97, 4.0, 0.97)        # Riesz -> maps
    _arrow(ax, 5.5, 0.97, 5.9, 0.97)        # maps -> CNN
    _arrow(ax, 7.6, 0.97, 8.2, 1.55)        # CNN -> fusion
    _arrow(ax, 8.95, 2.2, 9.05, 2.2)        # fusion -> class

    # legend
    _box(ax, 0.1, 0.05, 0.45, 0.3, "", FROZEN, FROZEN_E)
    ax.text(0.62, 0.2, "frozen (no training)", fontsize=9, va="center")
    _box(ax, 3.2, 0.05, 0.45, 0.3, "", TRAIN, TRAIN_E)
    ax.text(3.72, 0.2, "trained", fontsize=9, va="center")

    ax.text(2.65, 3.75, "appearance stream", fontsize=9.5, style="italic", color="#3d6fb4", ha="center")
    ax.text(2.65, 0.18, "motion stream", fontsize=9.5, style="italic", color="#3d6fb4", ha="center")
    fig.tight_layout()
    fig.savefig("runs/fig_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved runs/fig_architecture.png")


def fig_lowlight_pub():
    plt.rcParams.update({"font.size": 15})
    P = {"phase": ("Riesz-phase (ours)", "#1565C0", "o", 2.8, 9),
         "flow": ("optical flow", "#9e9e9e", "s", 2.0, 7),
         "framediff": ("frame-diff", "#c7a76b", "^", 2.0, 7)}
    conds = ["clean", "low_light_s1", "low_light_s3", "low_light_s5"]
    xs = [0, 1, 3, 5]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, ds, title in [(axes[0], "jester", "Jester"), (axes[1], "ipn", "IPN Hand")]:
        for m in ["flow", "framediff", "phase"]:                  # phase last = on top
            d = json.load(open(f"runs/full_{ds}_{m}.json"))["conditions"]
            ys = [d[c]["mean"] for c in conds]
            es = [d[c]["std"] for c in conds]
            lab, col, mk, lw, ms = P[m]
            ax.errorbar(xs, ys, yerr=es, marker=mk, color=col, lw=lw, markersize=ms,
                        capsize=3, label=lab, zorder=3 if m == "phase" else 2)
        ax.set_xlabel("Low-light severity")
        ax.set_xticks(xs)
        ax.grid(alpha=0.3)
        ax.text(0.04, 0.06, title, transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="bottom")
    axes[0].set_ylabel("Top-1 accuracy")
    axes[0].legend(frameon=False, fontsize=12, loc="upper right")
    fig.tight_layout()
    fig.savefig("runs/fig_lowlight_pub.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved runs/fig_lowlight_pub.png")


if __name__ == "__main__":
    fig_architecture()
    fig_lowlight_pub()
