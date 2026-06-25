"""MAB baseline: learning-based online preheating via UCB.

Treats each candidate (cloudlet, foundation-model) placement as an arm of a
multi-armed bandit and explores preheating actions to gradually learn their
rewards from observed feedback, instead of directly estimating the demand from
history.
"""

import numpy as np
from Class.model import Model
from Class.adapter import Adapter
from Class.request import Request
from Class.cloudlet import Cloudlet
from utils import compute_pulling_delays, compute_bts_volume, serve_and_cache_lru
import config as C


def _count_demands(requests, num_cl, num_models, num_services):
    counts = np.zeros((num_cl, num_models, num_services))
    for req in requests:
        counts[req.home, req.model_id, req.service_type] += 1
    return counts


def run_mab(
    all_requests: list[list[Request]],
    cloudlets: list[Cloudlet],
    models_dict: dict[int, Model],
    adapters_dict: dict[tuple[int, int], Adapter],
    delta: np.ndarray,
    alpha: float = 1.0,
) -> dict:
    for cl in cloudlets:
        cl.reset()

    T = len(all_requests)
    num_cl = len(cloudlets)
    num_models = len(models_dict)
    num_services = C.NUM_SERVICE_TYPES

    # bandit statistics over arms (cloudlet, model)
    n_pulls = np.zeros((num_cl, num_models))
    mean_reward = np.zeros((num_cl, num_models))
    # observed service types per (cloudlet, model) for adapter co-caching
    seen_service = np.zeros((num_cl, num_models, num_services), dtype=bool)

    pull_times, hit_rates, bts_volumes, idle_bw = [], [], [], []

    slot_bw = C.CLUSTER_SLOT_GB
    registry_bw = C.REGISTRY_SLOT_GB
    last_used: dict = {}

    for t in range(T):
        reqs = all_requests[t]

        # ── current-slot metrics under existing cache ──
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

        # ── observe feedback and update arms ──
        counts = _count_demands(reqs, num_cl, num_models, num_services)
        demand_cm = counts.sum(axis=2)  # (cl, model)
        seen_service |= counts > 0
        rows, cols = np.nonzero(demand_cm)
        for ci, mid in zip(rows, cols):
            r = float(demand_cm[ci, mid])
            n_pulls[ci, mid] += 1
            mean_reward[ci, mid] += (r - mean_reward[ci, mid]) / n_pulls[ci, mid]

        # ── UCB scores and greedy preheating within budgets ──
        explore = alpha * np.sqrt(2.0 * np.log(t + 2) / np.maximum(n_pulls, 1e-9))
        ucb = mean_reward + explore
        ucb[n_pulls == 0] = np.inf  # force exploration of new arms

        order = np.argsort(-ucb, axis=None)
        residual_registry = registry_bw
        cl_budget = np.array([min(slot_bw, cl.free_storage) for cl in cloudlets])

        slot_preheat = 0.0
        for flat in order:
            if residual_registry <= 0:
                break
            ci = flat // num_models
            mid = flat % num_models
            if not np.isfinite(ucb[ci, mid]) and ucb[ci, mid] != np.inf:
                continue
            if ucb[ci, mid] <= 0:
                break
            cl = cloudlets[ci]
            if cl.has_model(mid):
                continue
            size = models_dict[mid].size
            if size > cl_budget[ci] or size > residual_registry:
                continue
            cl.cache_model(models_dict[mid])
            last_used[(ci, ("M", mid))] = t
            cl_budget[ci] -= size
            residual_registry -= size
            slot_preheat += size
            # co-cache adapters for observed service types
            for qt in range(num_services):
                if not seen_service[ci, mid, qt]:
                    continue
                adp = adapters_dict[(mid, qt)]
                if adp.size <= cl_budget[ci] and adp.size <= residual_registry:
                    if cl.cache_adapter(adp):
                        last_used[(ci, ("W", (mid, qt)))] = t
                        cl_budget[ci] -= adp.size
                        residual_registry -= adp.size
                        slot_preheat += adp.size

        idle_bw.append(slot_preheat)

    return {
        "pull_times": pull_times,
        "hit_rates": hit_rates,
        "bts_volumes": bts_volumes,
        "idle_bw": idle_bw,
    }
