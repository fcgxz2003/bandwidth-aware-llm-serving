"""
Cluster initialisation and request generation.
Builds cloudlet topology and delay matrix from the EUA dataset.
"""

from __future__ import annotations
import csv
import numpy as np
from Class import Model, Adapter, Request, Cloudlet
import config as C

# ═══════════════════════ EUA dataset loading ══════════════════════════


def _haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance between two (lat, lon) points in km.

    Accepts scalars or numpy arrays.
    """
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def load_eua_cloudlet_coords(
    csv_path: str = C.EUA_EDGE_SERVER_CSV,
    num_sample: int = C.NUM_CLOUDLETS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Load edge-server coordinates from EUA CSV and sample *num_sample*.

    Returns an (num_sample, 2) array of [lat, lon].
    """
    coords = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = float(row["LATITUDE"])
            lon = float(row["LONGITUDE"])
            coords.append((lat, lon))
    coords = np.array(coords)
    if num_sample >= len(coords):
        return coords
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.choice(len(coords), size=num_sample, replace=False)
    return coords[idx]


def load_eua_user_coords(csv_path: str = C.EUA_USER_CSV) -> np.ndarray:
    """Load EUA user coordinates. Returns an (N, 2) array of [lat, lon]."""
    coords = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = float(row["Latitude"])
            lon = float(row["Longitude"])
            coords.append((lat, lon))
    return np.array(coords)


def assign_users_to_cloudlets(
    user_coords: np.ndarray, cloudlet_coords: np.ndarray
) -> np.ndarray:
    """Assign each user to the nearest cloudlet (vectorised).

    Returns a (num_users,) int array of cloudlet indices.
    """
    # shape: (n_users, 1, 2) vs (1, n_cloudlets, 2)
    u = user_coords[:, np.newaxis, :]  # (U, 1, 2)
    c = cloudlet_coords[np.newaxis, :, :]  # (1, C, 2)
    dists = _haversine_km(u[:, :, 0], u[:, :, 1], c[:, :, 0], c[:, :, 1])
    return np.argmin(dists, axis=1)


# ═══════════════════════ Cluster initialisation ══════════════════════


def create_models() -> list[Model]:
    """Create foundation models from the real open-source model catalog."""
    return [Model(id=j, size_gb=size) for j, (_, size) in enumerate(C.MODEL_CATALOG)]


def create_adapters(models: list[Model], rng: np.random.Generator) -> list[Adapter]:
    """Create one adapter per (model, service_type) pair.

    A LoRA adapter's footprint scales with its base model (hidden dimension and
    layer count), so each adapter's size is a per-task fraction of the base
    model size, clamped to a realistic absolute range.
    """
    frac_lo, frac_hi = getattr(C, "ADAPTER_SIZE_FRAC_RANGE", (0.01, 0.04))
    clamp_lo, clamp_hi = C.ADAPTER_SIZE_RANGE_GB
    adapters = []
    for m in models:
        for q in range(C.NUM_SERVICE_TYPES):
            frac = rng.uniform(frac_lo, frac_hi)
            size_gb = float(np.clip(m.size_gb * frac, clamp_lo, clamp_hi))
            adapters.append(
                Adapter(model_id=m.id, service_type=q, size_gb=round(size_gb, 4))
            )
    return adapters


def create_cloudlets(
    num_cloudlets: int = C.NUM_CLOUDLETS,
    rng: np.random.Generator | None = None,
    storage_caps: np.ndarray | None = None,
) -> list[Cloudlet]:
    """Create *num_cloudlets* cloudlets.

    Each cloudlet's storage capacity is sampled from CACHE_CAPACITY_RANGE_GB.
    Pass *storage_caps* to reuse a fixed capacity vector across algorithms
    (ensures a fair comparison); otherwise sample with *rng*.
    """
    if storage_caps is None:
        if rng is None:
            rng = np.random.default_rng(0)
        storage_caps = rng.uniform(*C.CACHE_CAPACITY_RANGE_GB, size=num_cloudlets)
    return [
        Cloudlet(cid=i, storage_cap_gb=float(storage_caps[i]))
        for i in range(num_cloudlets)
    ]


def sample_storage_caps(num_cloudlets: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a per-cloudlet storage capacity vector (GB)."""
    return rng.uniform(*C.CACHE_CAPACITY_RANGE_GB, size=num_cloudlets)


def build_delta_matrix(cloudlet_coords: np.ndarray) -> np.ndarray:
    """Build unit data-transfer delay matrix delta(cl_i, cl_j).

    Shape: (|CL|+1, |CL|).
      - Rows 0..|CL|-1 : peer-to-peer delay based on step function of
        geo-distance (near / mid / far).
      - Row  |CL|       : registry -> cloudlet (DELTA_REGISTRY).
      - Diagonal        : 0 (local hit).
    """
    n = len(cloudlet_coords)

    # Vectorised pairwise haversine  (n, n)
    lat = cloudlet_coords[:, 0]
    lon = cloudlet_coords[:, 1]
    dist = _haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])

    # Step function mapping
    peer = np.full_like(dist, C.DELTA_FAR)
    peer[dist <= C.DIST_THRESHOLD_MID] = C.DELTA_MID
    peer[dist <= C.DIST_THRESHOLD_NEAR] = C.DELTA_NEAR
    np.fill_diagonal(peer, C.DELTA_LOCAL)

    # Append registry row
    delta = np.vstack([peer, np.full((1, n), C.DELTA_REGISTRY)])
    return delta


# ═══════════════════════ Request generation ═══════════════════════════


def zipf_weights(num_models: int, alpha: float) -> np.ndarray:
    """Normalised Zipf weights over *num_models* ranks."""
    ranks = np.arange(1, num_models + 1, dtype=float)
    w = 1.0 / ranks**alpha
    return w / w.sum()


def daily_rate_multiplier(
    slot_in_day: int, slots_per_day: int, peak_valley: float
) -> float:
    """Sinusoidal intra-day request rate multiplier.

    slot_in_day in [0, slots_per_day).
    peak_valley = peak / valley ratio.
    """
    # r(t) = A + B*sin(2π t/T - π/2), max A+B, min A-B
    # peak/valley = (A+B)/(A-B) => B = A*(pv-1)/(pv+1)
    A = 1.0
    B = A * (peak_valley - 1) / (peak_valley + 1)
    phase = 2 * np.pi * slot_in_day / slots_per_day - np.pi / 2
    return A + B * np.sin(phase)


def generate_requests_for_slot(
    slot: int,
    models: list[Model],
    rng: np.random.Generator,
    model_weights: np.ndarray | None = None,
    mean_requests: int = C.MEAN_REQUESTS_PER_SLOT,
    peak_valley: float = C.PEAK_VALLEY_RATIO,
    user_assignments: np.ndarray | None = None,
    num_cloudlets: int = C.NUM_CLOUDLETS,
) -> list[Request]:
    """Generate a list of requests for time-slot *slot*.

    - Total count ~ Poisson(mean * daily_multiplier).
    - Foundation model chosen by Zipf weights.
    - Service type chosen uniformly at random.
    - Home cloudlet: sampled from user assignments if given, else uniform.
    """
    if model_weights is None:
        model_weights = zipf_weights(len(models), C.ZIPF_ALPHA)

    slot_in_day = slot % C.SLOTS_PER_DAY
    rate = mean_requests * daily_rate_multiplier(
        slot_in_day, C.SLOTS_PER_DAY, peak_valley
    )
    n_requests = max(1, rng.poisson(rate))

    model_ids = rng.choice(len(models), size=n_requests, p=model_weights)
    service_types = rng.integers(0, C.NUM_SERVICE_TYPES, size=n_requests)

    if user_assignments is not None and len(user_assignments) > 0:
        # sample random users; their assigned cloudlet becomes home
        user_idx = rng.integers(0, len(user_assignments), size=n_requests)
        homes = user_assignments[user_idx]
    else:
        homes = rng.integers(0, num_cloudlets, size=n_requests)

    return [
        Request(home=int(h), service_type=int(q), model_id=int(m))
        for h, q, m in zip(homes, service_types, model_ids)
    ]


def generate_daily_trace(
    models: list[Model],
    rng: np.random.Generator,
    num_slots: int,
    mean_requests: int = C.MEAN_REQUESTS_PER_SLOT,
    user_assignments: np.ndarray | None = None,
    num_cloudlets: int = C.NUM_CLOUDLETS,
) -> list[list[Request]]:
    """Daily scenario: a recurring intra-day tidal demand over a fixed catalog."""
    weights = zipf_weights(len(models), C.ZIPF_ALPHA)
    return [
        generate_requests_for_slot(
            t,
            models,
            rng,
            weights,
            mean_requests=mean_requests,
            user_assignments=user_assignments,
            num_cloudlets=num_cloudlets,
        )
        for t in range(num_slots)
    ]
