"""Random preheating offline baseline.

Greedily fills each cloudlet's residual data volume with randomly ordered
foundation models and adapters, ignoring both the request distribution and the
P2P topology.
"""

import numpy as np
from Class.model import Model
from Class.adapter import Adapter
from Class.cloudlet import Cloudlet
import config as C


def offline_randpre(
    cloudlets: list[Cloudlet],
    models_dict: dict[int, Model],
    adapters_dict: dict[tuple[int, int], Adapter],
    rng: np.random.Generator,
    residual_peer: np.ndarray | None = None,
    residual_registry: float | None = None,
) -> None:
    """random preheating across the cluster."""
    num_cl = len(cloudlets)
    if residual_peer is None:
        residual_peer = np.full(num_cl, C.CLUSTER_SLOT_GB)
    if residual_registry is None:
        residual_registry = C.REGISTRY_SLOT_GB

    model_ids = list(models_dict.keys())
    adapter_keys = list(adapters_dict.keys())
    for cl in cloudlets:
        if residual_registry <= 0:
            break
        budget = min(residual_peer[cl.id], residual_registry, cl.free_storage)
        for idx in rng.permutation(len(model_ids)):
            mid = model_ids[idx]
            m = models_dict[mid]
            if (
                not cl.has_model(mid)
                and m.size <= budget
                and m.size <= residual_registry
            ):
                cl.cache_model(m)
                budget -= m.size
                residual_registry -= m.size
        for idx in rng.permutation(len(adapter_keys)):
            ak = adapter_keys[idx]
            adp = adapters_dict[ak]
            if (
                not cl.has_adapter(*ak)
                and adp.size <= budget
                and adp.size <= residual_registry
            ):
                cl.cache_adapter(adp)
                budget -= adp.size
                residual_registry -= adp.size
