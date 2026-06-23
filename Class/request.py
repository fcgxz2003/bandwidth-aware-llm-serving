"""Inference request r^t_k."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Request:
    """Inference request r^t_k = (h_k, q_k) with scheduled model z_{k,j}."""
    home: int               # h_k: home cloudlet index
    service_type: int       # q_k
    model_id: int           # scheduled foundation model (z_{k,j}=1)
