"""Popularity-based offline baseline.

Sorts the requested (foundation model + adapter) co-cache units by their request
counts in the next time slot and greedily fills the residual data volumes with
the most requested ones on the cloudlets that requested them, ignoring the P2P
topology and the submodular marginal gain of each placement.
"""

from collections import Counter
import numpy as np
from Class.model import Model
from Class.adapter import Adapter
from Class.request import Request
from Class.cloudlet import Cloudlet
import config as C


def offline_popularity(
    requests: list[Request],
    cloudlets: list[Cloudlet],
    models_dict: dict[int, Model],
    adapters_dict: dict[tuple[int, int], Adapter],
    delta: np.ndarray,
    residual_peer: np.ndarray | None = None,
    residual_registry: float | None = None,
) -> tuple[dict[tuple[int, int], bool], dict[tuple[int, int, int], bool]]:
    num_cl = len(cloudlets)
    if residual_peer is None:
        bw = C.CLUSTER_SLOT_GB
        residual_peer = np.full(num_cl, bw)
    if residual_registry is None:
        residual_registry = C.REGISTRY_SLOT_GB

    # Co-cache each (foundation model + its requested adapter) as one unit,
    # ranked by the request count of that (cloudlet, model, service) tuple, so
    # that a placement actually serves the request that asked for it instead of
    # spending the scarce registry budget on bare models without their adapters.
    unit_count = Counter((r.home, r.model_id, r.service_type) for r in requests)

    budget = {
        cl.id: min(residual_peer[cl.id], residual_registry, cl.free_storage)
        for cl in cloudlets
    }

    # candidate co-cache units on the home cloudlets that requested them
    cands = [(cnt, ci, mid, qt) for (ci, mid, qt), cnt in unit_count.items()]
    cands.sort(key=lambda x: -x[0])

    mu: dict[tuple[int, int], bool] = {}
    nu: dict[tuple[int, int, int], bool] = {}
    for _, ci, mid, qt in cands:
        cl = cloudlets[ci]
        need_model = not cl.has_model(mid)
        need_adapter = not cl.has_adapter(mid, qt)
        size = 0.0
        if need_model:
            size += models_dict[mid].size
        if need_adapter:
            size += adapters_dict[(mid, qt)].size
        if size <= 0 or size > budget[ci] or size > residual_registry:
            continue
        if need_model:
            cl.cache_model(models_dict[mid])
            mu[(ci, mid)] = True
        if need_adapter:
            cl.cache_adapter(adapters_dict[(mid, qt)])
            nu[(ci, mid, qt)] = True
        budget[ci] -= size
        residual_registry -= size

    return mu, nu
