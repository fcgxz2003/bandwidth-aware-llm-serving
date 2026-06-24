"""LoRA adapter w(l_j, q_m)."""


class Adapter:
    """LoRA adapter w(l_j, q_m)."""

    def __init__(self, model_id: int, service_type: int, size: float):
        self.model_id = model_id
        self.service_type = service_type    # q_m service index
        self.size = size                    # s^W(l_j, q_m) in GB
