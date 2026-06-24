"""Edge cloudlet cl_i."""

from Class.model import Model
from Class.adapter import Adapter


class Cloudlet:
    """Edge cloudlet cl_i."""

    def __init__(self, cid: int, storage_cap: float):
        self.id = cid
        self.storage_cap = storage_cap       # S(cl_i)

        # cache state: x^t_{i,j}, y^t_{i,j,m}
        self.cached_models: set[int] = set()                    # model_id
        self.cached_adapters: set[tuple[int, int]] = set()      # (model_id, service_type)

        # track sizes for eviction accounting
        self._model_sizes: dict[int, float] = {}
        self._adapter_sizes: dict[tuple[int, int], float] = {}

    # ── capacity queries ─────────────────────────────────────
    @property
    def used_storage(self) -> float:
        return sum(self._model_sizes.values()) + sum(self._adapter_sizes.values())

    @property
    def free_storage(self) -> float:
        return self.storage_cap - self.used_storage

    # ── cache operations ─────────────────────────────────────
    def cache_model(self, model: Model) -> bool:
        """Cache a foundation model. Return True if newly added."""
        if model.id in self.cached_models:
            return False
        self.cached_models.add(model.id)
        self._model_sizes[model.id] = model.size
        return True

    def cache_adapter(self, adapter: Adapter) -> bool:
        """Cache an adapter. Return True if newly added."""
        key = (adapter.model_id, adapter.service_type)
        if key in self.cached_adapters:
            return False
        self.cached_adapters.add(key)
        self._adapter_sizes[key] = adapter.size
        return True

    def evict_model(self, model_id: int):
        """Evict a foundation model along with its co-cached adapters.

        Under the co-caching design an adapter is only useful when its
        foundation model is co-located on the same cloudlet, so evicting the
        model also releases all of its adapters to reclaim their storage.
        """
        if model_id not in self.cached_models:
            return
        self.cached_models.discard(model_id)
        self._model_sizes.pop(model_id, None)
        for key in [k for k in self.cached_adapters if k[0] == model_id]:
            self.cached_adapters.discard(key)
            self._adapter_sizes.pop(key, None)

    def evict_adapter(self, key: tuple[int, int]):
        """Evict a single adapter."""
        self.cached_adapters.discard(key)
        self._adapter_sizes.pop(key, None)

    def has_model(self, model_id: int) -> bool:
        return model_id in self.cached_models

    def has_adapter(self, model_id: int, service_type: int) -> bool:
        return (model_id, service_type) in self.cached_adapters

    def reset(self):
        """Clear all cached items (cold start)."""
        self.cached_models.clear()
        self.cached_adapters.clear()
        self._model_sizes.clear()
        self._adapter_sizes.clear()
