"""Algorithm 2: PreHeat — online continuous preheating."""

from __future__ import annotations
import numpy as np
from Class import Model, Adapter, Request, Cloudlet
from utils import compute_pull_delays, compute_bts_volume, serve_and_cache_lru
from offline.greedy import offline_greedy
import config as C


class DemandEstimator:
    """Dual-EWMA demand estimator.

    Maintains per (cloudlet_i, model_j, service_type_m):
      - R-EWMA: lambda^R_{i,j,m}(t)   short-term estimate
      - D-EWMA: lambda^D_{i,j,m}(p)   daily-periodic estimate (H periods)
    """

    def __init__(self, num_cloudlets: int, num_models: int, num_services: int):
        shape = (num_cloudlets, num_models, num_services)
        self.lambda_R = np.zeros(shape)
        self.lambda_D = np.zeros((C.H_PERIODS, *shape))
        self._shape = shape

    def update(self, slot: int, counts: np.ndarray):
        self.lambda_R = (1 - C.OMEGA_R) * self.lambda_R + C.OMEGA_R * counts
        p = slot % C.SLOTS_PER_DAY * C.H_PERIODS // C.SLOTS_PER_DAY
        self.lambda_D[p] = (1 - C.OMEGA_D) * self.lambda_D[p] + C.OMEGA_D * counts

    def estimate(self, slot: int) -> np.ndarray:
        next_slot = slot + 1
        p_next = next_slot % C.SLOTS_PER_DAY * C.H_PERIODS // C.SLOTS_PER_DAY
        return (1 - C.THETA) * self.lambda_D[p_next] + C.THETA * self.lambda_R


def _count_demands(requests, num_cloudlets, num_models, num_services):
    counts = np.zeros((num_cloudlets, num_models, num_services))
    for req in requests:
        counts[req.home, req.model_id, req.service_type] += 1
    return counts


def _build_predicted_requests(lam, rng):
    """Turn the fractional demand estimate into a predicted request set.

    Each cell holds the expected number of requests for (cloudlet, model,
    service) in the next slot. We use unbiased stochastic rounding —
    floor(val) deterministic copies plus one extra with probability equal to
    the fractional part — so that the predicted set size matches the expected
    total demand instead of collapsing to zero (nearest rounding) or exploding
    to one-per-nonzero-cell (ceiling).
    """
    predicted = []
    it = np.nditer(lam, flags=["multi_index"])
    while not it.finished:
        val = float(it[0])
        if val > 0:
            base = int(np.floor(val))
            n = base + (1 if rng.random() < (val - base) else 0)
            if n > 0:
                i, j, m = it.multi_index
                for _ in range(n):
                    predicted.append(Request(home=i, model_id=j, service_type=m))
        it.iternext()
    return predicted


def run_preheat(
    all_requests: list[list[Request]],
    cloudlets: list[Cloudlet],
    models_dict: dict[int, Model],
    adapters_dict: dict[tuple[int, int], Adapter],
    delta: np.ndarray,
) -> dict:
    """Run Algorithm 2: online continuous preheating."""
    T = len(all_requests)
    num_cl = len(cloudlets)
    num_models = len(models_dict)
    num_services = C.NUM_SERVICE_TYPES

    estimator = DemandEstimator(num_cl, num_models, num_services)
    pull_times, hit_rates, bts_volumes, idle_bw = [], [], [], []

    slot_bw = C.CLUSTER_SLOT_GB
    registry_bw = C.REGISTRY_SLOT_GB
    rng = np.random.default_rng(0)
    last_used: dict = {}

    for t in range(T):
        reqs = all_requests[t]

        # compute current-slot metrics
        D_M, D_W = compute_pull_delays(
            reqs, cloudlets, models_dict, adapters_dict, delta
        )
        pull_times.append(float((D_M + D_W).mean()) if reqs else 0.0)

        hits = sum(1 for req in reqs if cloudlets[req.home].has_model(req.model_id))
        hit_rates.append(hits / len(reqs) if reqs else 0.0)

        bts_volumes.append(
            compute_bts_volume(reqs, cloudlets, models_dict, adapters_dict)
        )

        # pulled content enters the local cache (main.tex: "both pulled and
        # preheated content enter the local cache"), so the served foundation
        # models and adapters are retained for subsequent peer-to-peer reuse.
        serve_and_cache_lru(reqs, cloudlets, models_dict, adapters_dict, last_used, t)

        # demand estimation
        counts = _count_demands(reqs, num_cl, num_models, num_services)
        estimator.update(t, counts)
        lam = estimator.estimate(t)

        # build predicted request set
        predicted = _build_predicted_requests(lam, rng)
        if not predicted:
            idle_bw.append(0.0)
            continue

        # invoke offline greedy; storage-tight eviction of lowest-demand
        # cached content (Algorithm 2, line 6) is handled inside the greedy
        residual_peer = np.full(num_cl, slot_bw)
        mu, nu = offline_greedy(
            predicted,
            cloudlets,
            models_dict,
            adapters_dict,
            delta,
            residual_peer,
            registry_bw,
            demand=lam,
        )

        # preheated content is freshly useful: refresh its recency so the LRU
        # bookkeeping does not evict it before the predicted demand arrives.
        for ci, mid in mu:
            last_used[(ci, ("M", mid))] = t
        for ci, mid, qt in nu:
            last_used[(ci, ("W", (mid, qt)))] = t

        # idle bandwidth consumed by preheating traffic this slot
        slot_preheat = sum(models_dict[mid].size_gb for (_, mid) in mu)
        slot_preheat += sum(adapters_dict[(mid, qt)].size_gb for (_, mid, qt) in nu)
        idle_bw.append(slot_preheat)

    return {
        "pull_times": pull_times,
        "hit_rates": hit_rates,
        "bts_volumes": bts_volumes,
        "idle_bw": idle_bw,
    }
