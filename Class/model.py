"""Foundation model l_j."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Model:
    """Foundation model l_j."""
    id: int
    size_gb: float          # s^M(l_j) in GB
