"""LFU baseline: reactive caching with frequency-based eviction."""

import numpy as np
from Class.model import Model
from Class.adapter import Adapter
from Class.request import Request
from Class.cloudlet import Cloudlet
from utils import compute_pulling_delays, compute_bts_volume


def run_lfu(
    all_requests: list[list[Request]],
    cloudlets: list[Cloudlet],
    models_dict: dict[int, Model],
    adapters_dict: dict[tuple[int, int], Adapter],
    delta: np.ndarray,
) -> dict:
    for cl in cloudlets:
        cl.reset()

    freq_model: dict[tuple[int, int], int] = {}
    freq_adapter: dict[tuple[int, int, int], int] = {}
    pull_times, hit_rates, bts_volumes, idle_bw = [], [], [], []

    for reqs in all_requests:
        D_M, D_W = compute_pulling_delays(
            reqs, cloudlets, models_dict, adapters_dict, delta
        )
        pull_times.append(float((D_M + D_W).mean()) if reqs else 0.0)

        # BTS measured consistently at the start of the slot (shared helper)
        bts_volumes.append(
            compute_bts_volume(reqs, cloudlets, models_dict, adapters_dict)
        )

        hits = 0

        for req in reqs:
            cl = cloudlets[req.home]
            m = models_dict[req.model_id]
            adp_key = (req.model_id, req.service_type)
            adp = adapters_dict[adp_key]

            freq_model[(cl.id, req.model_id)] = (
                freq_model.get((cl.id, req.model_id), 0) + 1
            )
            freq_adapter[(cl.id, req.model_id, req.service_type)] = (
                freq_adapter.get((cl.id, req.model_id, req.service_type), 0) + 1
            )

            if cl.has_model(req.model_id):
                hits += 1

            if not cl.has_model(req.model_id):
                while cl.free_storage < m.size and cl.cached_models:
                    worst = min(
                        cl.cached_models,
                        key=lambda mid: freq_model.get((cl.id, mid), 0),
                    )
                    cl.evict_model(worst)
                if cl.free_storage >= m.size:
                    cl.cache_model(m)

            if not cl.has_adapter(req.model_id, req.service_type):
                while cl.free_storage < adp.size and cl.cached_adapters:
                    worst = min(
                        cl.cached_adapters,
                        key=lambda k: freq_adapter.get((cl.id, *k), 0),
                    )
                    cl.evict_adapter(worst)
                if cl.free_storage >= adp.size:
                    cl.cache_adapter(adp)

        hit_rates.append(hits / len(reqs) if reqs else 0.0)
        idle_bw.append(0.0)  # LFU is reactive, no preheating traffic

    return {
        "pull_times": pull_times,
        "hit_rates": hit_rates,
        "bts_volumes": bts_volumes,
        "idle_bw": idle_bw,
    }
