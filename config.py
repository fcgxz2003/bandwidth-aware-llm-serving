"""
Global default parameters for the PreHeat simulation.
"""

import os as _os

# ── Paths ─────────────────────────────────────────────────
_EXPERIMENT_DIR = _os.path.dirname(_os.path.abspath(__file__))
EUA_EDGE_SERVER_CSV = _os.path.join(
    _EXPERIMENT_DIR, "eua-dataset", "edge-servers", "site-optus-melbCBD.csv"
)
EUA_USER_CSV = _os.path.join(
    _EXPERIMENT_DIR, "eua-dataset", "users", "users-melbcbd-generated.csv"
)

# ── Cluster topology ─────────────────────────────────────
NUM_CLOUDLETS = 125  # |CL|, sampled from EUA dataset
CLUSTER_BW_GBPS = 1.0  # cluster link bandwidth (Gbps)
REGISTRY_BW_GBPS = 0.24  # registry uplink bandwidth (Gbps)
# ── Foundation models & adapters ─────────────────────────
# Real open-source LLMs spanning the Qwen, LLaMA, Gemma, Mistral, Phi, Yi and
# CodeLlama families. Sizes are the actual on-disk footprint of the Q4_K_M GGUF
# weights distributed by Ollama (default tag download size, https://ollama.com/
# library), covering the 0.6B-34B edge-deployable range -> 0.5GB-20GB.
MODEL_CATALOG = [
    # Qwen family
    ("Qwen2.5-0.5B", 0.5),
    ("Qwen3-0.6B", 0.5),
    ("Qwen2.5-1.5B", 1.0),
    ("Qwen2.5-3B", 1.9),
    ("Qwen3-4B", 2.5),
    ("Qwen2.5-7B", 4.7),
    ("Qwen2.5-Coder-7B", 4.7),
    ("Qwen2.5-14B", 9.0),
    ("Qwen3-14B", 9.3),
    ("Qwen2.5-32B", 20.0),
    # LLaMA family
    ("Llama-3.2-1B", 1.3),
    ("Llama-3.2-3B", 2.0),
    ("Llama-2-7B", 3.8),
    ("Llama-3-8B", 4.7),
    ("Llama-3.1-8B", 4.9),
    ("Llama-2-13B", 7.4),
    # CodeLlama family
    ("CodeLlama-7B", 3.8),
    ("CodeLlama-13B", 7.4),
    ("CodeLlama-34B", 19.0),
    # Gemma family
    ("Gemma3-1B", 0.8),
    ("Gemma3-4B", 3.3),
    ("Gemma2-9B", 5.4),
    ("Gemma3-12B", 8.1),
    ("Gemma2-27B", 16.0),
    ("Gemma3-27B", 17.0),
    # Mistral family
    ("Mistral-7B", 4.1),
    ("Mistral-Nemo-12B", 7.1),
    # Phi family
    ("Phi-3-mini-3.8B", 2.2),
    ("Phi-4-14B", 9.1),
    # Yi family
    ("Yi-34B", 19.0),
]
NUM_FOUNDATION_MODELS = len(MODEL_CATALOG)  # |L| = 30
FM_SIZE_RANGE_GB = (0.5, 20)  # foundation model size range (GB), kept for fallback
NUM_SERVICE_TYPES = 10  # |Q|: task-specific fine-tuned adapters per model
# Adapter (LoRA/QLoRA) sizes follow real task-specific adapters such as the
# Predibase "LoRA Land" collection (27 task adapters on Mistral-7B): a rank-16
# adapter is ~13.6 MB (https://huggingface.co/predibase/magicoder), while
# higher-rank adapters targeting all linear layers on larger bases reach a few
# hundred MB. A LoRA's footprint scales with the base model's hidden dimension
# and layer count, so we size each (model, task) adapter as a fraction of its
# base model: adapter_gb = base_model_gb * f, with f drawn per task from the
# range below (different ranks / target modules). A larger base model thus
# yields proportionally larger adapters.
ADAPTER_SIZE_FRAC_RANGE = (0.01, 0.04)  # adapter footprint as fraction of base model
ADAPTER_SIZE_RANGE_GB = (0.005, 0.8)  # absolute clamp on adapter size (GB)

# ── Request generation ───────────────────────────────────
ZIPF_ALPHA = 0.7  # Zipf skewness for model popularity
PEAK_VALLEY_RATIO = 40  # daily peak / valley request ratio
MEAN_REQUESTS_PER_SLOT = 100  # baseline mean requests per slot

# ── Storage ────────────────────────────────
CACHE_CAPACITY_RANGE_GB = (16, 48)  # per-cloudlet cache capacity range (GB)

# ── Online estimation ────────────────────────────────────
OMEGA_R = 0.3  # R-EWMA smoothing factor
OMEGA_D = 0.3  # D-EWMA smoothing factor
THETA = 0.5  # blend weight for demand estimate
H_PERIODS = 48  # daily periods for D-EWMA
SLOT_MINUTES = 30  # slot length (minutes)
SLOTS_PER_DAY = 24 * 60 // SLOT_MINUTES  # 48

# ── Per-slot transferable data volume (GB) ───────────────
# Link rates are in Gbps (gigabit/s); divide by 8 to convert to GByte.
#   GB/slot = Gbps * (SLOT_MINUTES * 60 s) / 8
CLUSTER_SLOT_GB = CLUSTER_BW_GBPS * SLOT_MINUTES * 60 / 8  # per cloudlet
REGISTRY_SLOT_GB = REGISTRY_BW_GBPS * SLOT_MINUTES * 60 / 8  # registry uplink

# ── Simulation ────────────────────────────────────────────
NUM_DAYS = 3  # simulation duration (days)
NUM_REPLAYS = 20  # independent replays for averaging

# ── Network delay (ms/MB) ── step function of geo-distance
DELTA_LOCAL = 0.0  # local cache hit
DELTA_NEAR = 0.8  # dist <= 1 km
DELTA_MID = 8.0  # 1 km < dist <= 5 km
DELTA_FAR = 40.0  # dist > 5 km
DELTA_REGISTRY = 80.0  # registry -> any cloudlet
DIST_THRESHOLD_NEAR = 1.0  # km
DIST_THRESHOLD_MID = 5.0  # km
