"""Plot the offline figures from results/offline_{scale,alpha}.json."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common.expcommon as E
import common.plotcommon as P

OFF = os.path.join(P.PAPER_FIGS, "offline")
COLORS = {"P2P": P.C4, "RandPre": P.C3, "Popularity": P.C2, "BACG": P.C1}
HATCH = {"P2P": "/", "RandPre": "\\", "Popularity": "-", "BACG": "|"}
YLABEL = {
    "pull": "Average pulling time (ms)",
    "bts": "BTS data volume (GB)",
    "gain": "Preheating gain (s)",
}
# (metric -> filename) for each sweep
SCALE_FILES = {
    "pull": "offline_pull_time.pdf",
    "bts": "offline_bts_volume.pdf",
    "gain": "offline_gain.pdf",
}
ALPHA_FILES = {
    "pull": "offline_pull_alpha.pdf",
    "bts": "offline_bts_alpha.pdf",
    "gain": "offline_gain_alpha.pdf",
}


def _render(data, files):
    order = data["order"]
    labels = {k: k for k in order}
    for metric, fname in files.items():
        # Pin the pulling-time axis to 0..500 (step 100) so (a)/(b) match.
        yticks = range(0, 501, 100) if metric == "pull" else None
        # Pin the gain axis to 0..200 (step 50) so (a)/(b) match.
        if metric == "gain":
            yticks = range(0, 201, 50)
        # P2P never preheats, so it has no preheating gain; show only the
        # three preheating methods in the gain figures.
        metric_order = [a for a in order if a != "P2P"] if metric == "gain" else order
        P.bars(
            data["x"],
            data["metrics"][metric],
            metric_order,
            COLORS,
            HATCH,
            labels,
            data["xlabel"],
            YLABEL[metric],
            os.path.join(OFF, fname),
            yticks=yticks,
        )


def main():
    _render(E.load_json("offline_scale.json"), SCALE_FILES)
    _render(E.load_json("offline_alpha.json"), ALPHA_FILES)


if __name__ == "__main__":
    main()
