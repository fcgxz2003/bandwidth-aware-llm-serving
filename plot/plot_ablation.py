"""Plot the ablation figures from results/ablation.json.

Compare the contribution of DEWMA's two demand estimation components. 
Each figure overlays the three variants on the same axes so the
R-EWMA-only / D-EWMA-only / combined behaviour can be compared directly:

  * online_ablation_bts.pdf  -- cumulative BTS data volume over time
  * online_ablation_gain.pdf -- cumulative preheating gain over time
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common.exp as E
import common.plot as P

ON = os.path.join(P.PAPER_FIGS, "ablation")

LINE_COLORS = {
    "R-EWMA only": "#e8743b",
    "D-EWMA only": "#d4b106",
    "DEWMA": "#1f77b4",
}
MARKERS = {"R-EWMA only": "s", "D-EWMA only": "^", "DEWMA": "*"}


def _gain(data):
    """Cumulative preheating gain over time, i.e. the running sum of the
    per-slot pulling-time reduction relative to the P2P baseline. A steeper,
    higher curve means more pulling time saved. Output:
    online_ablation_gain.pdf."""
    demand = data["demand"]
    order = [data["labels"][k] for k in data["order"]]
    p2p = data["p2p_pull"]
    x = list(range(len(demand)))
    cumgain = {}
    for k in data["order"]:
        acc, run = [], 0.0
        for i in range(len(x)):
            run += p2p[i] - data["pull"][k][i]
            acc.append(run)
        cumgain[data["labels"][k]] = acc
    P.lines(
        x,
        cumgain,
        order,
        LINE_COLORS,
        "Time slot",
        "Cumulative\npreheating gain (ms)",
        os.path.join(ON, "online_ablation_gain.pdf"),
        markers=MARKERS,
        demand=demand,
        figsize=(16, 6),
        markevery=10,
        yticks=[0, 200, 400, 600, 800],
        xticks=list(range(0, len(x) + 1, 10)),
        legend_loc="upper left",
        legend_ncol=3,
    )


def _bts(data):
    """Cumulative BTS data volume over time. The slope is the per-slot
    back-to-source traffic; a flatter curve means less BTS. Output:
    online_ablation_bts.pdf."""
    demand = data["demand"]
    order = [data["labels"][k] for k in data["order"]]
    cumbts = {}
    for k in data["order"]:
        acc, run = [], 0.0
        for v in data["bts"][k]:
            run += v
            acc.append(run)
        cumbts[data["labels"][k]] = acc
    x = list(range(len(demand)))
    P.lines(
        x,
        cumbts,
        order,
        LINE_COLORS,
        "Time slot",
        "Cumulative BTS\nData Volume (GB)",
        os.path.join(ON, "online_ablation_bts.pdf"),
        markers=MARKERS,
        demand=demand,
        figsize=(16, 6),
        yticks=[0, 100, 200, 300, 400, 500],
        xticks=list(range(0, len(x) + 1, 10)),
        markevery=10,
        legend_loc="upper left",
        legend_ncol=3,
    )


def main():
    data = E.load_json("ablation.json")
    _bts(data)
    _gain(data)


if __name__ == "__main__":
    main()
