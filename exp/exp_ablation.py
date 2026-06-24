"""Ablation experiment: contribution of the R-EWMA and D-EWMA components.

Single tidal (daily) scenario. The same demand trace is replayed three times,
varying only DEWMA's blend weight theta:

  * Recent   (theta=1)  -> R-EWMA only  (reactive, short-term)
  * Daily    (theta=0)  -> D-EWMA only  (periodic, day-ahead)
  * Combined (theta=0.5)-> full DEWMA

A P2P (no-preheating) reference is also run as the zero baseline for the
cumulative preheating gain. Saves results/ablation.json.

Run:  python exp_ablation.py [--days 3] [--seed 42]
"""

from __future__ import annotations
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # noqa: E402,F401  (configures sys.path for flat imports)

import numpy as np  # noqa: E402

import config as C  # noqa: E402
import expcommon as E  # noqa: E402
from setup import create_cloudlets  # noqa: E402
from online import run_preheat, run_nocache  # noqa: E402

ORDER = ["Recent", "Daily", "Combined"]
THETAS = {"Recent": 1.0, "Daily": 0.0, "Combined": 0.5}
LABELS = {
    "Recent": "R-EWMA only",
    "Daily": "D-EWMA only",
    "Combined": "DEWMA",
}


def main(num_days: int = 3, seed: int = 53):
    fx = E.build_cluster(seed)
    num_slots = num_days * C.SLOTS_PER_DAY
    rng = np.random.default_rng(seed)
    trace = E.daily_trace(fx, rng, num_slots, C.MEAN_REQUESTS_PER_SLOT)
    md, ad, delta = fx["models_dict"], fx["adapters_dict"], fx["delta"]

    # P2P reference (no preheating): the flat baseline for cumulative gain.
    cls = create_cloudlets(fx["num_cl"], storage_caps=fx["storage_caps"])
    p2p = run_nocache(trace, cls, md, ad, delta)

    pull, bts, summary = {}, {}, {}
    for name in ORDER:
        cls = create_cloudlets(fx["num_cl"], storage_caps=fx["storage_caps"])
        res = run_preheat(trace, cls, md, ad, delta, theta=THETAS[name])
        pull[name] = [float(v) for v in res["pull_times"]]
        bts[name] = [float(v) for v in res["bts_volumes"]]
        summary[name] = {
            "avg_pull": float(np.mean(res["pull_times"])),
            "total_bts": float(np.sum(res["bts_volumes"])),
        }

    E.save_json(
        "ablation.json",
        {
            "order": ORDER,
            "labels": LABELS,
            "thetas": THETAS,
            "slots": list(range(num_slots)),
            "demand": [len(r) for r in trace],
            "pull": pull,
            "bts": bts,
            "p2p_pull": [float(v) for v in p2p["pull_times"]],
            "summary": summary,
        },
    )

    print("ablation summary (single tidal scenario):")
    for name in ORDER:
        s = summary[name]
        print(
            f"  {name:9s} (theta={THETAS[name]}): "
            f"avg_pull={s['avg_pull']:.2f} ms  total_bts={s['total_bts']:.1f} GB"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DEWMA component ablation")
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--seed", type=int, default=53)
    args = p.parse_args()
    main(args.days, args.seed)
