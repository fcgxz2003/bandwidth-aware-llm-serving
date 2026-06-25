"""On-demand P2P offline baseline.

The reference scheme performs no proactive preheating: every cloudlet starts
cold and content is only pulled reactively (peer-to-peer or from the registry)
when a request needs it. This module exposes that no-op placement explicitly so
that every offline method is dispatched the same way.
"""

from Class.model import Model
from Class.adapter import Adapter
from Class.request import Request
from Class.cloudlet import Cloudlet


def offline_p2p(
    requests: list[Request],
    cloudlets: list[Cloudlet],
    models_dict: dict[int, Model],
    adapters_dict: dict[tuple[int, int], Adapter],
) -> tuple[dict[tuple[int, int], bool], dict[tuple[int, int, int], bool]]:
    """No preheating: leave the cluster cold. Returns empty decisions."""
    return {}, {}
