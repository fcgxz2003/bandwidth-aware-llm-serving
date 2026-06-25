"""P2P baseline: on-demand pull from the registry with peer-to-peer sharing.

Traditional P2P distribution: a request pulls its missing foundation model and
adapter from the central registry, then caches the pulled content at its home
cloudlet so that nearby peers can later fetch it peer-to-peer. No preheating is
performed and storage is reclaimed with least-recently-used (LRU) eviction.
"""

import numpy as np
from Class.model import Model
from Class.adapter import Adapter
from Class.request import Request
from Class.cloudlet import Cloudlet
from utils import compute_pulling_delays, compute_bts_volume, serve_and_cache_lru


def run_nocache(
    all_requests: list[list[Request]],
    cloudlets: list[Cloudlet],
    models_dict: dict[int, Model],
    adapters_dict: dict[tuple[int, int], Adapter],
    delta: np.ndarray,
) -> dict:
    for cl in cloudlets:
        cl.reset()

    pull_times, hit_rates, bts_volumes, idle_bw = [], [], [], []
    last_used: dict = {}
    for t, reqs in enumerate(all_requests):
        # metrics under the cache state at the start of the slot (peers only)
        D_M, D_W = compute_pulling_delays(
            reqs, cloudlets, models_dict, adapters_dict, delta
        )
        pull_times.append(float((D_M + D_W).mean()) if reqs else 0.0)

        hits = sum(1 for r in reqs if cloudlets[r.home].has_model(r.model_id))
        hit_rates.append(hits / len(reqs) if reqs else 0.0)

        bts_volumes.append(
            compute_bts_volume(reqs, cloudlets, models_dict, adapters_dict)
        )

        # reactively cache the pulled content for peer-to-peer sharing
        serve_and_cache_lru(reqs, cloudlets, models_dict, adapters_dict, last_used, t)
        idle_bw.append(0.0)  # P2P never preheats

    return {
        "pull_times": pull_times,
        "hit_rates": hit_rates,
        "bts_volumes": bts_volumes,
        "idle_bw": idle_bw,
    }
