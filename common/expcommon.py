"""Shared compute layer for the evaluation experiments.

This module holds everything the *experiment* scripts need to run the
simulator: the cluster fixture, the offline/online method runners, and small
JSON helpers to persist results to ``experiment/results/``.

The experiment scripts (``exp_offline.py``, ``exp_online.py``) compute metrics
and save them as JSON; the plotting scripts (``plot_*.py``) only read those JSON
files, so changing a figure's style never re-runs the simulation.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config as C
from setup import (
    load_eua_cloudlet_coords,
    load_eua_user_coords,
    assign_users_to_cloudlets,
    create_models,
    create_adapters,
    build_delta_matrix,
    sample_storage_caps,
    create_cloudlets,
    zipf_weights,
    generate_requests_for_slot,
    generate_daily_trace,
)
from utils import compute_pulling_delays, compute_bts_volume
from offline.p2p import offline_p2p
from offline.randpre import offline_randpre
from offline.bacg import offline_bacg
from offline.popularity import offline_popularity
from online.preheat import run_preheat
from online.nocache import run_nocache
from online.lfu import run_lfu
from online.randpre import run_randpre
from online.mab import run_mab

# ─────────────────────────── sweep grids ───────────────────────────
SCALES = [100, 200, 500, 1000]
ALPHAS = [0.6, 0.8, 1.0, 1.2]
ALPHA_REQUESTS = 500  # fixed request count for the Zipf-skew sweep

OFFLINE_ORDER = ["P2P", "RandPre", "Popularity", "BACG"]
ONLINE_ORDER = ["P2P", "RandPre", "LFU", "MAB", "DEWMA"]

# ─────────────────────────── results I/O ───────────────────────────
# common/ -> experiment root -> results/
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)


def save_json(name: str, data: dict) -> str:
    """Write *data* to ``results/<name>`` and return the absolute path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print("saved", path)
    return path


def load_json(name: str) -> dict:
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


# ─────────────────────────── cluster fixture ───────────────────────────
def build_cluster(seed):
    rng = np.random.default_rng(seed)
    cl_coords = load_eua_cloudlet_coords(num_sample=C.NUM_CLOUDLETS, rng=rng)
    user_coords = load_eua_user_coords()
    user_assign = assign_users_to_cloudlets(user_coords, cl_coords)
    num_cl = len(cl_coords)
    models = create_models()
    adapters = create_adapters(models, rng)
    models_dict = {m.id: m for m in models}
    adapters_dict = {(a.model_id, a.service_type): a for a in adapters}
    delta = build_delta_matrix(cl_coords)
    storage_caps = sample_storage_caps(num_cl, rng)
    return dict(
        rng=rng,
        num_cl=num_cl,
        models=models,
        models_dict=models_dict,
        adapters_dict=adapters_dict,
        delta=delta,
        storage_caps=storage_caps,
        user_assign=user_assign,
    )


# ─────────────────────────── offline runners ───────────────────────────
def one_slot(fx, n_requests, alpha, seed):
    """Generate a single known slot of *n_requests* with the given Zipf skew."""
    rng = np.random.default_rng(seed)
    weights = zipf_weights(len(fx["models"]), alpha)
    reqs = generate_requests_for_slot(
        0,
        fx["models"],
        rng,
        weights,
        mean_requests=n_requests,
        peak_valley=1.0,  # flat: isolate scale, no tidal multiplier
        user_assignments=fx["user_assign"],
        num_cloudlets=fx["num_cl"],
    )
    return reqs[:n_requests] if len(reqs) > n_requests else reqs


def offline_methods(fx, reqs):
    """Return per-method ``(gain_ms, avg_pull_ms, bts_gb)`` on a cold cluster."""
    md, ad, delta = fx["models_dict"], fx["adapters_dict"], fx["delta"]
    slot_bw = C.CLUSTER_SLOT_GB
    registry_bw = C.REGISTRY_SLOT_GB
    out = {}
    for name in OFFLINE_ORDER:
        cls = create_cloudlets(fx["num_cl"], storage_caps=fx["storage_caps"])
        D_M0, D_W0 = compute_pulling_delays(reqs, cls, md, ad, delta)
        before = float((D_M0 + D_W0).sum())
        residual_peer = np.full(fx["num_cl"], slot_bw)
        if name == "P2P":
            offline_p2p(reqs, cls, md, ad)
        elif name == "RandPre":
            offline_randpre(cls, md, ad, fx["rng"], residual_peer, registry_bw)
        elif name == "Popularity":
            offline_popularity(reqs, cls, md, ad, delta, residual_peer, registry_bw)
        else:
            offline_bacg(reqs, cls, md, ad, delta, residual_peer, registry_bw)
        D_M1, D_W1 = compute_pulling_delays(reqs, cls, md, ad, delta)
        after = float((D_M1 + D_W1).sum())
        out[name] = (
            before - after,
            float((D_M1 + D_W1).mean()),
            compute_bts_volume(reqs, cls, md, ad),
        )
    return out


# ─────────────────────────── online runners ───────────────────────────
def run_online_methods(fx, trace):
    """Run every online method on *trace*, returning per-method raw results."""
    md, ad, delta = fx["models_dict"], fx["adapters_dict"], fx["delta"]
    runners = {
        "P2P": lambda c: run_nocache(trace, c, md, ad, delta),
        "RandPre": lambda c: run_randpre(trace, c, md, ad, delta, fx["rng"]),
        "LFU": lambda c: run_lfu(trace, c, md, ad, delta),
        "MAB": lambda c: run_mab(trace, c, md, ad, delta),
        "DEWMA": lambda c: run_preheat(trace, c, md, ad, delta),
    }
    out = {}
    for name, fn in runners.items():
        cls = create_cloudlets(fx["num_cl"], storage_caps=fx["storage_caps"])
        out[name] = fn(cls)
    return out


def daily_trace(fx, rng, num_slots, mean):
    return generate_daily_trace(
        fx["models"], rng, num_slots, mean, fx["user_assign"], fx["num_cl"]
    )
