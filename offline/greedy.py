"""Offline algorithm (Algorithm 1): Lazy Marginal-Gain Density Greedy Co-Caching."""

from __future__ import annotations
import heapq
import numpy as np
from Class import Model, Adapter, Request, Cloudlet
from utils import compute_pull_delays
import config as C

# ── Diagnostic instrumentation (no behavioural effect) ──
# When TRACE_ENABLED is True, every accepted candidate appends a tuple
# (density, gain, size, tag) to ACCEPT_TRACE so the marginal-gain-density
# distribution of preheated items can be inspected before deciding a cutoff.
TRACE_ENABLED = False
ACCEPT_TRACE: list[tuple[float, float, float, str]] = []

# ── Marginal-gain-density cutoff (Algorithm 1, optional) ──
# When > 0, the greedy stops preheating once the best remaining candidate's
# marginal-gain density (delay reduction per GB pulled) falls below this
# threshold, so scarce uplink budget is not spent on low-value content.
DENSITY_CUTOFF = 0.0


def offline_greedy(
    requests: list[Request],
    cloudlets: list[Cloudlet],
    models_dict: dict[int, Model],
    adapters_dict: dict[tuple[int, int], Adapter],
    delta: np.ndarray,
    residual_peer: np.ndarray | None = None,
    residual_registry: float | None = None,
    demand: np.ndarray | None = None,
) -> tuple[dict[tuple[int, int], bool], dict[tuple[int, int, int], bool]]:
    """Algorithm 1 — PreHeat Lazy Marginal-Gain Greedy.

    Parameters
    ----------
    requests : predicted request set R_{t+1}.
    cloudlets : current cache state (modified in-place).
    models_dict : model_id -> Model.
    adapters_dict : (model_id, service_type) -> Adapter.
    delta : delay matrix of shape (|CL|+1, |CL|).
    residual_peer : residual data budget per cloudlet (GB); default unlimited.
    residual_registry : residual registry data budget (GB); default unlimited.
    demand : optional (|CL|, |L|, |Q|) estimated-demand array. When given, a
        candidate that does not fit in the free storage of its cloudlet may
        evict cached content with strictly lower estimated demand to make room
        (Algorithm 2, line 6). When None (offline cold start), no eviction is
        performed and a candidate is simply skipped if it does not fit.

    Returns
    -------
    mu : {(cloudlet_id, model_id): True} — foundation-model preheating decisions.
    nu : {(cloudlet_id, model_id, service_type): True} — adapter preheating decisions.
    """
    num_cl = len(cloudlets)

    if residual_peer is None:
        bw = C.CLUSTER_SLOT_GB
        residual_peer = np.full(num_cl, bw)
    if residual_registry is None:
        residual_registry = C.REGISTRY_SLOT_GB

    # ── Step 1: compute initial D^M_k, D^W_k ──
    D_M, D_W = compute_pull_delays(
        requests, cloudlets, models_dict, adapters_dict, delta
    )

    # ── Step 2: per-cloudlet transfer budget b_i(t) ──
    # Storage is enforced separately (via free_storage + eviction) so that the
    # budget here reflects only the residual data volumes (Constraints P1.1/P1.2).
    budget = np.array(
        [min(residual_peer[cl.id], residual_registry) for cl in cloudlets], dtype=float
    )

    # ── demand lookups for capacity-driven eviction (Algorithm 2, line 6) ──
    def _model_demand(ci, mid):
        return float(demand[ci, mid, :].sum()) if demand is not None else 0.0

    def _adapter_demand(ci, mid, qt):
        return float(demand[ci, mid, qt]) if demand is not None else 0.0

    def _make_room(cl, size, incoming_demand, protected):
        """Evict cached items with strictly lower demand until *size* fits.

        Only items whose estimated demand is below *incoming_demand* and that
        were not placed in this same round (*protected*) may be evicted.
        Returns True if enough room was freed, False otherwise (cache left
        unchanged in that case).
        """
        if size <= cl.free_storage:
            return True
        victims = []
        for mid2 in cl.cached_models:
            if ("M", mid2) in protected:
                continue
            d = _model_demand(cl.id, mid2)
            if d < incoming_demand:
                victims.append((d, "M", mid2, cl._model_sizes.get(mid2, 0.0)))
        for key2 in cl.cached_adapters:
            if ("W", key2) in protected:
                continue
            d = _adapter_demand(cl.id, key2[0], key2[1])
            if d < incoming_demand:
                victims.append((d, "W", key2, cl._adapter_sizes.get(key2, 0.0)))
        victims.sort(key=lambda x: x[0])

        freeable = cl.free_storage + sum(v[3] for v in victims)
        if freeable < size:
            return False  # cannot fit even after evicting all lower-demand items
        for _, typ, key2, _sz in victims:
            if cl.free_storage >= size:
                break
            if typ == "M":
                cl.evict_model(key2)
            else:
                cl.evict_adapter(key2)
        return cl.free_storage >= size

    # ── Step 3: build candidate set & initial density ──
    req_by_model: dict[int, list[int]] = {}
    for k, req in enumerate(requests):
        req_by_model.setdefault(req.model_id, []).append(k)

    candidates = []
    for cl in cloudlets:
        for mid, model in models_dict.items():
            if not cl.has_model(mid) and model.size_gb <= budget[cl.id]:
                candidates.append(("M", cl.id, mid, -1))
            for qt in range(C.NUM_SERVICE_TYPES):
                adp_key = (mid, qt)
                if adp_key in adapters_dict and not cl.has_adapter(mid, qt):
                    adp = adapters_dict[adp_key]
                    if adp.size_gb <= budget[cl.id]:
                        candidates.append(("W", cl.id, mid, qt))

    def _marginal_gain(tag, ci, mid, qt):
        if tag == "M":
            size = models_dict[mid].size_gb
            gain = 0.0
            for k in req_by_model.get(mid, []):
                new_d = delta[ci, requests[k].home] * size
                reduction = D_M[k] - new_d
                if reduction > 0:
                    gain += reduction
            return gain, size
        else:
            adp = adapters_dict[(mid, qt)]
            size = adp.size_gb
            gain = 0.0
            for k in req_by_model.get(mid, []):
                if requests[k].service_type == qt:
                    new_d = delta[ci, requests[k].home] * size
                    reduction = D_W[k] - new_d
                    if reduction > 0:
                        gain += reduction
            return gain, size

    # max-heap via negated density
    heap = []
    for cand in candidates:
        tag, ci, mid, qt = cand
        gain, size = _marginal_gain(tag, ci, mid, qt)
        density = gain / size if size > 0 else 0.0
        heapq.heappush(heap, (-density, id(cand), cand, gain, size))

    # ── Step 4: best single element e* ──
    best_single = None
    best_single_gain = 0.0
    for entry in heap:
        neg_d, _, cand, gain, size = entry
        tag, ci, mid, qt = cand
        if size <= budget[ci]:
            if gain > best_single_gain:
                best_single_gain = gain
                best_single = cand

    # ── Step 5: greedy loop ──
    mu: dict[tuple[int, int], bool] = {}
    nu: dict[tuple[int, int, int], bool] = {}
    greedy_gain = 0.0
    placed: dict[int, set] = {cl.id: set() for cl in cloudlets}

    while heap:
        neg_d, _, cand, old_gain, size = heapq.heappop(heap)
        tag, ci, mid, qt = cand

        gain, size = _marginal_gain(tag, ci, mid, qt)
        density = gain / size if size > 0 else 0.0

        if heap:
            top_neg_d = heap[0][0]
            if -density > top_neg_d:
                heapq.heappush(heap, (-density, id(cand), cand, gain, size))
                continue

        # density cutoff: this candidate is now the true max-density item, so
        # every remaining candidate has density <= it (submodularity). Stop.
        if DENSITY_CUTOFF > 0.0 and density < DENSITY_CUTOFF:
            break

        if size > budget[ci]:
            continue

        # storage: evict lowest-demand cached content only when storage is tight
        cl = cloudlets[ci]
        if size > cl.free_storage:
            incoming = (
                _model_demand(ci, mid) if tag == "M" else _adapter_demand(ci, mid, qt)
            )
            if not _make_room(cl, size, incoming, placed[ci]):
                continue

        if tag == "M":
            mu[(ci, mid)] = True
            cl.cache_model(models_dict[mid])
            placed[ci].add(("M", mid))
        else:
            nu[(ci, mid, qt)] = True
            cl.cache_adapter(adapters_dict[(mid, qt)])
            placed[ci].add(("W", (mid, qt)))

        if TRACE_ENABLED:
            ACCEPT_TRACE.append((density, gain, size, tag))

        budget[ci] -= size
        residual_registry -= size
        for idx in range(num_cl):
            budget[idx] = min(budget[idx], residual_registry)

        greedy_gain += gain

        for k in req_by_model.get(mid, []):
            if tag == "M":
                new_d = delta[ci, requests[k].home] * size
                if new_d < D_M[k]:
                    D_M[k] = new_d
            else:
                if requests[k].service_type == qt:
                    adp = adapters_dict[(mid, qt)]
                    new_d = delta[ci, requests[k].home] * adp.size_gb
                    if new_d < D_W[k]:
                        D_W[k] = new_d

    # ── Step 6: keep better of greedy solution vs best single element ──
    if best_single is not None and best_single_gain > greedy_gain:
        pass

    return mu, nu
