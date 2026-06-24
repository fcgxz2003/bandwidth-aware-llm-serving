"""Shared plotting layer for the evaluation figures.

The plotting scripts (``plot_offline.py``, ``plot_online.py``) read the JSON
produced by the experiment scripts and render the bar charts consumed by the
paper. No simulation runs here, so restyling a figure is instant.
"""

from __future__ import annotations
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["hatch.linewidth"] = 0.3

PAPER_FIGS = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..",
        "paper",
        "figs",
        "evaluation",
    )
)

# Palette (cs-v1 style)
C1 = "#7fcdbb"  # proposed (BACG / DEWMA)
C2 = "#edf8b1"
C3 = "#d9ebd4"
C4 = "#f8ac8c"
C5 = "#fdbf6f"
FONT = {"weight": "normal", "size": 28}


def _nice_ceil(v):
    """Smallest 1/2/2.5/5 * 10^k value that is >= v (for tidy axis steps)."""
    import math

    if v <= 0:
        return 1.0
    exp = math.floor(math.log10(v))
    base = 10**exp
    for m in (1, 2, 2.5, 5, 10):
        if m * base >= v - 1e-9:
            return m * base
    return 10 * base


def _aligned_demand_ticks(demand, left_ticks, fill=0.55):
    """Return (ticks, top) for the demand twin axis so its ticks land exactly on
    the left-axis gridlines (``left_ticks``) with tidy round numbers, while the
    demand curve peaks at roughly ``fill`` of the axis height."""
    dmax = max(demand) if len(demand) else 1.0
    n = max(1, len(left_ticks) - 1)
    step = _nice_ceil(dmax / fill / n)
    ticks = [step * i for i in range(n + 1)]
    return ticks, step * n


def bars(
    x_labels, series, order, colors, hatches, labels, xlabel, ylabel, out, yticks=None
):
    """Grouped bar chart with a tight, evenly-spaced y-axis (top tick flush).

    Pass an explicit ``yticks`` list (e.g. ``range(0, 501, 100)``) to pin the
    axis; otherwise it is sized automatically so the tallest bar fills ~85%.
    """
    N = len(x_labels)
    ind = np.arange(N)
    k = len(order)
    bw = 0.8 / k
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    for i, algo in enumerate(order):
        offset = (i - (k - 1) / 2.0) * bw
        ax.bar(
            ind + offset,
            series[algo],
            width=bw,
            color=colors[algo],
            linewidth=0,
            edgecolor="black",
            label=labels[algo],
            hatch=hatches[algo],
        )
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#A8BAC4", lw=1.2)
    ax.spines["bottom"].set_lw(1.2)
    plt.xlabel(xlabel, FONT)
    plt.ylabel(ylabel, FONT, loc="center")
    plt.tick_params(labelsize=20)
    ax.set_xticks(ind)
    ax.set_xticklabels([str(x) for x in x_labels])
    # Y-axis: top tick flush with the top edge (no empty gap), but sized so the
    # tallest bar fills ~85% of the height. Equal-magnitude (a)/(b) subplots
    # then round to the same top tick, keeping the pair symmetric.
    if yticks is not None:
        yticks = list(yticks)
    else:
        vmax = max(max(series[a]) for a in order)
        loc = MaxNLocator(nbins=4, steps=[1, 2, 5, 10])
        yticks = [t for t in loc.tick_values(0, vmax / 0.85) if t >= 0]
        if yticks[-1] < vmax:
            yticks.append(yticks[-1] + (yticks[-1] - yticks[-2]))
    ax.set_yticks(yticks)
    ax.set_ylim(0, yticks[-1])
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print("saved", out)


def lines(
    x,
    series,
    order,
    colors,
    xlabel,
    ylabel,
    out,
    markers=None,
    demand=None,
    demand_label="Number of requests",
    legend=True,
    yticks=None,
    xticks=None,
    markevery=None,
    figsize=(8, 6),
    legend_loc="upper left",
    legend_ncol=2,
):
    """Multi-series line chart over a time axis.

    Optionally shades a per-slot ``demand`` curve behind the lines on a twin
    axis to visualize the tidal traffic that drives the preheating gain.
    """
    default_markers = dict(zip(order, ["o", "s", "^", "D", "v", "P", "*"]))
    markers = markers or default_markers
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if demand is not None:
        ax2 = ax.twinx()
        ax2.fill_between(x, demand, color="#eef1f4", zorder=0)
        ax2.plot(x, demand, color="#c4ccd4", lw=1.0, zorder=0)
        ax2.set_ylabel(demand_label, FONT, loc="center")
        ax2.tick_params(labelsize=20)
        if yticks is not None:
            dticks, dtop = _aligned_demand_ticks(demand, yticks)
            ax2.set_ylim(0, dtop)
            ax2.set_yticks(dticks)
        else:
            ax2.set_ylim(0, max(demand) / 0.55 if max(demand) > 0 else 1)

    every = markevery if markevery is not None else max(1, len(x) // 12)
    for a in order:
        ax.plot(
            x,
            series[a],
            color=colors[a],
            marker=markers[a],
            markersize=8,
            markevery=every,
            markeredgecolor="black",
            markeredgewidth=0.6,
            lw=2.4,
            label=a,
            zorder=3,
        )
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#A8BAC4", lw=1.0)
    ax.spines["bottom"].set_lw(1.2)
    ax.set_xlabel(xlabel, FONT)
    ax.set_ylabel(ylabel, FONT, loc="center")
    ax.tick_params(labelsize=20)
    ax.set_xlim(min(x), max(x))
    if xticks is not None:
        ax.set_xticks(list(xticks))
    if yticks is not None:
        ax.set_yticks(list(yticks))
        ax.set_ylim(0, list(yticks)[-1])
    if legend:
        ax.legend(prop={"size": 22}, ncol=legend_ncol, framealpha=0.9, loc=legend_loc)
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print("saved", out)


def lines_broken(
    x,
    series,
    order,
    colors,
    xlabel,
    ylabel,
    out,
    yticks_bottom,
    yticks_top,
    markers=None,
    demand=None,
    demand_label="Number of requests",
    demand_top=None,
    legend=True,
    xticks=None,
    markevery=None,
    height_ratios=(1, 2.2),
    figsize=(16, 7),
):
    """Multi-series line chart with a broken (split) y-axis.

    The top panel shows the cold-start spikes (``yticks_top``) while the bottom
    panel zooms into the steady-state regime (``yticks_bottom``) and carries the
    shaded ``demand`` curve. A zigzag separates the two panels.
    """
    default_markers = dict(zip(order, ["o", "s", "^", "D", "v", "P", "*"]))
    markers = markers or default_markers
    every = markevery if markevery is not None else max(1, len(x) // 12)

    fig, (axt, axb) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": list(height_ratios), "hspace": 0.28},
    )

    # demand twin axes: ticks aligned to the bottom-panel gridlines. The bottom
    # twin carries the shaded curve and the (round, aligned) tick labels; the
    # top twin only reserves an identical right margin so both panels keep the
    # same width and their right edges line up.
    if demand is not None:
        if demand_top is not None:
            n = max(1, len(yticks_bottom) - 1)
            step = demand_top / n
            dticks = [step * i for i in range(n + 1)]
            dtop = demand_top
        else:
            dticks, dtop = _aligned_demand_ticks(demand, yticks_bottom)
        ax2b = axb.twinx()
        ax2b.fill_between(x, demand, color="#eef1f4", zorder=0)
        ax2b.plot(x, demand, color="#c4ccd4", lw=1.0, zorder=0)
        ax2b.set_ylim(0, dtop)
        ax2b.set_yticks(dticks)
        ax2b.tick_params(labelsize=20)

        ax2t = axt.twinx()
        ax2t.set_ylim(0, dtop)
        ax2t.set_yticks(dticks)
        ax2t.set_yticklabels([str(int(t)) for t in dticks])
        for lbl in ax2t.get_yticklabels():
            lbl.set_color("none")
        ax2t.tick_params(axis="y", length=0, labelsize=20)
        # hide the twin-axis spines at the break so they don't draw a flat line
        ax2b.spines["top"].set_visible(False)
        ax2t.spines["bottom"].set_visible(False)

    for a in order:
        common = dict(
            color=colors[a],
            marker=markers[a],
            markersize=8,
            markevery=every,
            markeredgecolor="black",
            markeredgewidth=0.6,
            lw=2.4,
            zorder=3,
        )
        axt.plot(x, series[a], label=a, **common)
        axb.plot(x, series[a], **common)

    for ax in (axt, axb):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#A8BAC4", lw=1.0)
        ax.set_xlim(min(x), max(x))
        ax.tick_params(labelsize=20)
    axb.set_zorder(2)
    axb.patch.set_visible(False)

    axt.set_ylim(yticks_top[0], yticks_top[-1])
    axt.set_yticks(list(yticks_top))
    axb.set_ylim(yticks_bottom[0], yticks_bottom[-1])
    axb.set_yticks(list(yticks_bottom))

    if xticks is not None:
        axb.set_xticks(list(xticks))
    axb.set_xlabel(xlabel, FONT)
    fig.supylabel(ylabel, x=0.015, fontsize=FONT["size"])
    if demand is not None:
        fig.text(
            0.985,
            0.5,
            demand_label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=FONT["size"],
        )

    # zigzag break between the two panels
    axt.spines["bottom"].set_visible(False)
    axb.spines["top"].set_visible(False)
    axt.tick_params(labelbottom=False, bottom=False)
    n_zig = 64
    xz = np.linspace(0, 1, n_zig)
    amp = 0.018
    yz = amp * (np.arange(n_zig) % 2 * 2 - 1)
    zkw = dict(color="k", lw=1.3, clip_on=False, zorder=10)
    axt.plot(xz, yz, transform=axt.transAxes, **zkw)
    axb.plot(xz, 1 + yz, transform=axb.transAxes, **zkw)

    if legend:
        axt.legend(prop={"size": 22}, ncol=2, framealpha=0.9, loc="upper right")

    fig.tight_layout()
    # tight_layout reserves a wide left margin for the y tick labels, which
    # leaves a visible gap between the (far-left) supylabel and the axes. Pull
    # the axes leftward so the plotting area expands and the gap shrinks.
    fig.subplots_adjust(left=0.09)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print("saved", out)
