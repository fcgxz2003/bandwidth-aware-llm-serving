"""Plot the online figures from results/online_timeseries.json.

Produces the daily-scenario time-series charts: average pulling time, cumulative
BTS data volume, and cumulative preheating gain.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # noqa: E402,F401  (configures sys.path for flat imports)

import expcommon as E
import plotcommon as P

ON = os.path.join(P.PAPER_FIGS, "online")
MARKERS = {"P2P": "o", "RandPre": "s", "LFU": "^", "MAB": "D", "DEWMA": "*"}
# Saturated high-contrast colors for the time-series line charts (the pastel
# bar palette is too light to read against the demand shading).
LINE_COLORS = {
    "P2P": "#e8743b",
    "RandPre": "#8c8c8c",
    "LFU": "#d4b106",
    "MAB": "#9467bd",
    "DEWMA": "#1f77b4",
}

SCENARIO = "daily"


def _ts_pull(data):
    """Paper Fig. fig_online_pull: average pulling time over time, with a
    broken y-axis so the cold-start spikes and the steady-state regime are both
    visible. The tidal demand is shaded behind it. Output:
    online_{tag}_ts_pull.pdf."""
    order = data["order"]
    tag = SCENARIO
    sc = data["scenarios"][tag]
    demand = sc["demand"]
    pull = {m: sc["pull"][m] for m in order}
    x = list(range(len(demand)))
    P.lines_broken(
        x,
        pull,
        order,
        LINE_COLORS,
        "Time slot",
        "Average pulling time (ms)",
        os.path.join(ON, f"online_{tag}_ts_pull.pdf"),
        yticks_bottom=[0, 20, 40, 60],
        yticks_top=[100, 200, 300, 400],
        markers=MARKERS,
        demand=demand,
        demand_top=300,
        markevery=10,
        xticks=list(range(0, len(x) + 1, 10)),
        height_ratios=(1, 1),
        figsize=(16, 7),
    )


def _ts_bts(data):
    """Paper Fig. fig_online_bts: cumulative BTS data volume over time.
    The slope of each curve is the per-slot BTS traffic; a flatter curve means
    less back-to-source. Output: online_{tag}_ts_bts.pdf."""
    order = data["order"]
    tag = SCENARIO
    sc = data["scenarios"][tag]
    demand = sc["demand"]
    cumbts = {}
    for m in order:
        series = sc["bts"][m]
        acc, run = [], 0.0
        for v in series:
            run += v
            acc.append(run)
        cumbts[m] = acc
    x = list(range(len(demand)))
    P.lines(
        x,
        cumbts,
        order,
        LINE_COLORS,
        "Time slot",
        "Cumulative BTS\nData Volume (GB)",
        os.path.join(ON, f"online_{tag}_ts_bts.pdf"),
        markers=MARKERS,
        demand=demand,
        figsize=(16, 6),
        yticks=[0, 100, 200, 300, 400, 500, 600],
        xticks=list(range(0, len(x) + 1, 10)),
        markevery=10,
    )


def _ts_gain(data):
    """Paper Fig. fig_online_ts: cumulative preheating gain over time, i.e. the
    running sum of the per-slot pulling-time reduction relative to the P2P
    baseline (P2P is the flat zero reference). Output:
    online_{tag}_ts_gain.pdf."""
    order = data["order"]
    tag = SCENARIO
    sc = data["scenarios"][tag]
    demand = sc["demand"]
    pull = {m: sc["pull"][m] for m in order}
    x = list(range(len(demand)))
    cumgain = {}
    for m in order:
        acc, run = [], 0.0
        for i in range(len(x)):
            run += pull["P2P"][i] - pull[m][i]
            acc.append(run)
        cumgain[m] = acc
    P.lines(
        x,
        cumgain,
        order,
        LINE_COLORS,
        "Time slot",
        "Cumulative\npreheating gain (ms)",
        os.path.join(ON, f"online_{tag}_ts_gain.pdf"),
        markers=MARKERS,
        demand=demand,
        figsize=(16, 6),
        yticks=[0, 200, 400, 600, 800, 1000, 1200],
        xticks=list(range(0, len(x) + 1, 10)),
        markevery=10,
    )


def main():
    ts = E.load_json("online_timeseries.json")
    _ts_pull(ts)  # Fig. fig_online_pull
    _ts_bts(ts)  # Fig. fig_online_bts
    _ts_gain(ts)  # Fig. fig_online_ts


if __name__ == "__main__":
    main()
