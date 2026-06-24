"""Inference request r^t_k."""


class Request:
    """Inference request r^t_k = (h_k, q_k) with scheduled model z_{k,j}."""

    def __init__(self, home: int, service_type: int, model_id: int):
        self.home = home                # h_k: home cloudlet index
        self.service_type = service_type    # q_k
        self.model_id = model_id        # scheduled foundation model (z_{k,j}=1)
