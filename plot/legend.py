"""Standalone generator for the offline figure legend.

Writes ``paper/figs/evaluation/offline/legend.pdf`` shared by all offline
bar charts (BACG is Algorithm 1).

Run with:  python -m plot.legend     (from the experiment/ directory)
       or:  python plot/legend.py
"""

import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["hatch.linewidth"] = 0.3

# Palette (cs-v1 style)
C1 = "#7fcdbb"  # proposed (BACG)
C2 = "#edf8b1"
C3 = "#d9ebd4"
C4 = "#f8ac8c"

OUT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "paper", "figs", "evaluation", "offline"
    )
)


def _legend(order, colors, hatches, labels, out, ncol):
    fig, ax = plt.subplots(1, 1, figsize=(11, 1.0))
    elems = [
        Patch(facecolor=colors[a], edgecolor="black", hatch=hatches[a], label=labels[a])
        for a in order
    ]
    plt.legend(handles=elems, ncol=ncol, mode="expand", prop={"size": 16})
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    order = ["P2P", "RandPre", "Popularity", "BACG"]
    colors = {"P2P": C4, "RandPre": C3, "Popularity": C2, "BACG": C1}
    hatch = {"P2P": "/", "RandPre": "\\", "Popularity": "-", "BACG": "|"}
    lab = {k: k for k in order}
    _legend(order, colors, hatch, lab, os.path.join(OUT_DIR, "legend.pdf"), ncol=4)
    # P2P has no preheating gain, so the gain figures show only the three
    # preheating methods and use a dedicated legend.
    gain_order = ["RandPre", "Popularity", "BACG"]
    _legend(
        gain_order,
        colors,
        hatch,
        lab,
        os.path.join(OUT_DIR, "legend_gain.pdf"),
        ncol=3,
    )


if __name__ == "__main__":
    main()
