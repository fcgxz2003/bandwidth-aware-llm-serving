"""Online experiment: DEWMA vs. baselines under tidal traffic.

Saves the per-slot time-series metrics of the daily tidal scenario:
  * results/online_timeseries.json   (per-slot pull, bts for every method)

Run:  python exp_online.py [--days 2] [--seed 42]
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config as C
import common.expcommon as E


def _timeseries(fx, trace_fn, num_slots, seed):
    """Per-slot pulling time of every method on a single tidal trace.

    Keeps the full per-slot trajectory (instead of aggregating it) so the
    figure can show how the preheating gain evolves over time relative to the
    demand peaks and valleys.
    """
    order = E.ONLINE_ORDER
    rng = np.random.default_rng(seed)
    trace = trace_fn(fx, rng, num_slots, C.MEAN_REQUESTS_PER_SLOT)
    res = E.run_online_methods(fx, trace)
    return {
        "slots": list(range(num_slots)),
        "demand": [len(reqs) for reqs in trace],
        "pull": {m: [float(v) for v in res[m]["pull_times"]] for m in order},
        "bts": {m: [float(v) for v in res[m]["bts_volumes"]] for m in order},
    }


def main(num_days=2, seed=42):
    fx = E.build_cluster(seed)
    num_slots = num_days * C.SLOTS_PER_DAY
    print(f"Cluster: {fx['num_cl']} cloudlets, {len(fx['models'])} models")

    scenarios = {"daily": _timeseries(fx, E.daily_trace, num_slots, seed)}
    E.save_json(
        "online_timeseries.json",
        {"order": E.ONLINE_ORDER, "scenarios": scenarios},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Online experiment")
    p.add_argument("--days", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args.days, args.seed)
