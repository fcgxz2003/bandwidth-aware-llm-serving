"""RandPre baseline: random preheating using residual bandwidth each slot."""

import numpy as np
from Class.model import Model
from Class.adapter import Adapter
from Class.request import Request
from Class.cloudlet import Cloudlet
from utils import compute_pulling_delays, compute_bts_volume, serve_and_cache_lru
import config as C


def run_randpre(
    all_requests: list[list[Request]],
    cloudlets: list[Cloudlet],
    models_dict: dict[int, Model],
    adapters_dict: dict[tuple[int, int], Adapter],
    delta: np.ndarray,
    rng: np.random.Generator | None = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(42)

    for cl in cloudlets:
        cl.reset()

    pull_times, hit_rates, bts_volumes, idle_bw = [], [], [], []
    slot_bw = C.CLUSTER_SLOT_GB
    registry_bw = C.REGISTRY_SLOT_GB
    all_model_ids = list(models_dict.keys())
    all_adapter_keys = list(adapters_dict.keys())
    last_used: dict = {}

    for t, reqs in enumerate(all_requests):
        D_M, D_W = compute_pulling_delays(
            reqs, cloudlets, models_dict, adapters_dict, delta
        )
        pull_times.append(float((D_M + D_W).mean()) if reqs else 0.0)

        hits = sum(1 for r in reqs if cloudlets[r.home].has_model(r.model_id))
        hit_rates.append(hits / len(reqs) if reqs else 0.0)

        bts_volumes.append(
            compute_bts_volume(reqs, cloudlets, models_dict, adapters_dict)
        )

        # pulled content enters the local cache (common to all methods)
        serve_and_cache_lru(reqs, cloudlets, models_dict, adapters_dict, last_used, t)

        # random preheating, bounded by the per-cloudlet residual bandwidth and
        # the shared registry uplink budget (Constraint P1.2)
        slot_preheat = 0.0
        residual_registry = registry_bw
        for cl in cloudlets:
            if residual_registry <= 0:
                break
            remaining = min(slot_bw, cl.free_storage, residual_registry)
            for idx in rng.permutation(len(all_model_ids)):
                mid = all_model_ids[idx]
                m = models_dict[mid]
                if (
                    not cl.has_model(mid)
                    and m.size <= remaining
                    and m.size <= residual_registry
                ):
                    cl.cache_model(m)
                    last_used[(cl.id, ("M", mid))] = t
                    remaining -= m.size
                    residual_registry -= m.size
                    slot_preheat += m.size
            for idx in rng.permutation(len(all_adapter_keys)):
                ak = all_adapter_keys[idx]
                adp = adapters_dict[ak]
                if (
                    not cl.has_adapter(*ak)
                    and adp.size <= remaining
                    and adp.size <= residual_registry
                ):
                    cl.cache_adapter(adp)
                    last_used[(cl.id, ("W", ak))] = t
                    remaining -= adp.size
                    residual_registry -= adp.size
                    slot_preheat += adp.size
        idle_bw.append(slot_preheat)

    return {
        "pull_times": pull_times,
        "hit_rates": hit_rates,
        "bts_volumes": bts_volumes,
        "idle_bw": idle_bw,
    }
