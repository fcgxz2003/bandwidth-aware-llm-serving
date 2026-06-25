"""
Shared utility functions used by both offline and online algorithms.
"""

import numpy as np
from Class.model import Model
from Class.adapter import Adapter
from Class.request import Request
from Class.cloudlet import Cloudlet


def compute_pulling_delays(
    requests: list[Request],
    cloudlets: list[Cloudlet],
    models_dict: dict[int, Model],
    adapters_dict: dict[tuple[int, int], Adapter],
    delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pulling delays D^M_k and D^W_k under current cache state.

    Returns two arrays of length len(requests).
    """
    n = len(requests)
    D_M = np.empty(n)
    D_W = np.empty(n)
    registry = len(cloudlets)

    for k, req in enumerate(requests):
        m = models_dict[req.model_id]
        best_m = delta[registry, req.home] * m.size
        for cl in cloudlets:
            if cl.has_model(req.model_id):
                d = delta[cl.id, req.home] * m.size
                if d < best_m:
                    best_m = d
        D_M[k] = best_m

        adp_key = (req.model_id, req.service_type)
        adp = adapters_dict[adp_key]
        best_w = delta[registry, req.home] * adp.size
        for cl in cloudlets:
            if cl.has_adapter(req.model_id, req.service_type):
                d = delta[cl.id, req.home] * adp.size
                if d < best_w:
                    best_w = d
        D_W[k] = best_w

    return D_M, D_W


def compute_bts_volume(
    requests: list[Request],
    cloudlets: list[Cloudlet],
    models_dict: dict[int, Model],
    adapters_dict: dict[tuple[int, int], Adapter],
) -> float:
    """BTS data volume for a time slot.

    A foundation model or adapter triggers a registry pull only when it is not
    pulled by any cloudlet in the cluster.
    """
    cached_models: set[int] = set()
    cached_adapters: set[tuple[int, int]] = set()
    for cl in cloudlets:
        cached_models |= cl.cached_models
        cached_adapters |= cl.cached_adapters

    missing_models: set[int] = set()
    missing_adapters: set[tuple[int, int]] = set()
    for req in requests:
        if req.model_id not in cached_models:
            missing_models.add(req.model_id)
        adp_key = (req.model_id, req.service_type)
        if adp_key not in cached_adapters:
            missing_adapters.add(adp_key)

    bts = sum(models_dict[mid].size for mid in missing_models)
    bts += sum(adapters_dict[k].size for k in missing_adapters)
    return float(bts)


def serve_and_cache_lru(
    requests: list[Request],
    cloudlets: list[Cloudlet],
    models_dict: dict[int, Model],
    adapters_dict: dict[tuple[int, int], Adapter],
    last_used: dict,
    clock: int,
) -> None:
    """Reactively cache pulled content at the home cloudlet with LRU eviction.

    Models the on-demand P2P baseline: after a request pulls its missing
    foundation model and adapter, the content is cached at the requesting
    (home) cloudlet so that nearby peers can later pull it P2P. When
    storage is tight, the least recently used cached item is evicted.
    """
    for req in requests:
        cl = cloudlets[req.home]
        m = models_dict[req.model_id]
        adp_key = (req.model_id, req.service_type)
        adp = adapters_dict[adp_key]

        if cl.has_model(req.model_id):
            last_used[(cl.id, ("M", req.model_id))] = clock
        else:
            _evict_lru_until(cl, m.size, last_used)
            if cl.free_storage >= m.size and cl.cache_model(m):
                last_used[(cl.id, ("M", req.model_id))] = clock

        if cl.has_adapter(*adp_key):
            last_used[(cl.id, ("W", adp_key))] = clock
        else:
            _evict_lru_until(cl, adp.size, last_used)
            if cl.free_storage >= adp.size and cl.cache_adapter(adp):
                last_used[(cl.id, ("W", adp_key))] = clock


def _evict_lru_until(cl: Cloudlet, size_needed: float, last_used: dict) -> None:
    """Evict least recently used items from cl."""
    while cl.free_storage < size_needed and (cl.cached_models or cl.cached_adapters):
        candidates = [
            (last_used.get((cl.id, ("M", mid)), -1), "M", mid)
            for mid in cl.cached_models
        ]
        candidates += [
            (last_used.get((cl.id, ("W", key)), -1), "W", key)
            for key in cl.cached_adapters
        ]
        if not candidates:
            break
        _, typ, ref = min(candidates, key=lambda x: x[0])
        if typ == "M":
            cl.evict_model(ref)
            last_used.pop((cl.id, ("M", ref)), None)
        else:
            cl.evict_adapter(ref)
            last_used.pop((cl.id, ("W", ref)), None)
