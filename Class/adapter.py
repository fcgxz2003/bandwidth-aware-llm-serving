"""LoRA adapter w(l_j, q_m)."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Adapter:
    """LoRA adapter w(l_j, q_m)."""
    model_id: int
    service_type: int       # q_m index
    size_gb: float          # s^W(l_j, q_m) in GB
