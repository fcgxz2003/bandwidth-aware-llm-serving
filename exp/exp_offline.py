"""Offline experiment: BACG vs. baselines.

Sweeps two dimensions on a cold, single-slot cluster and saves the metrics:
  * request-scale sweep (fixed Zipf skew)  -> results/offline_scale.json
  * Zipf-skew sweep     (fixed request #)  -> results/offline_alpha.json

Run:  python exp_offline.py [--seed 42]
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # configures sys.path for flat imports

import config as C
import expcommon as E


def _sweep(fx, xs, alpha_for, n_for, seed):
    order = E.OFFLINE_ORDER
    pull = {m: [] for m in order}
    bts = {m: [] for m in order}
    gain = {m: [] for m in order}
    for x in xs:
        reqs = E.one_slot(fx, n_for(x), alpha_for(x), seed)
        res = E.offline_methods(fx, reqs)
        for m in order:
            gain[m].append(res[m][0] / 1000.0)  # ms -> s
            pull[m].append(res[m][1])
            bts[m].append(res[m][2])
    return {"pull": pull, "bts": bts, "gain": gain}


def main(seed=42):
    fx = E.build_cluster(seed)
    print(f"Cluster: {fx['num_cl']} cloudlets, {len(fx['models'])} models")

    # request-scale sweep at the default Zipf skew
    metrics = _sweep(
        fx, E.SCALES, alpha_for=lambda x: C.ZIPF_ALPHA, n_for=lambda x: x, seed=seed
    )
    E.save_json(
        "offline_scale.json",
        {
            "x": E.SCALES,
            "xlabel": "Number of requests",
            "order": E.OFFLINE_ORDER,
            "alpha": C.ZIPF_ALPHA,
            "metrics": metrics,
        },
    )

    # Zipf-skew sweep at a fixed request count
    metrics_a = _sweep(
        fx,
        E.ALPHAS,
        alpha_for=lambda x: x,
        n_for=lambda x: E.ALPHA_REQUESTS,
        seed=seed,
    )
    E.save_json(
        "offline_alpha.json",
        {
            "x": E.ALPHAS,
            "xlabel": r"Zipf skew $\alpha$",
            "order": E.OFFLINE_ORDER,
            "requests": E.ALPHA_REQUESTS,
            "metrics": metrics_a,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Offline experiment")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args.seed)
