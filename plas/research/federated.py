"""Federated learning with secure aggregation and differential privacy hooks."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass
class FederatedClient:
    """Cross-device or cross-silo client."""

    client_id: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


def secure_average(updates: Iterable[Mapping[str, Any]], weights: Iterable[float] | None = None) -> dict[str, Any]:
    """Weighted average aggregation for tensor-like update dictionaries."""

    import torch

    updates = list(updates)
    ws = list(weights or [1.0] * len(updates))
    total = sum(ws) or 1.0
    result: dict[str, Any] = {}
    for key in updates[0]:
        acc = None
        for update, weight in zip(updates, ws):
            value = update[key] * (weight / total)
            acc = value if acc is None else acc + value
        result[key] = acc
    return result


def add_dp_noise(update: Mapping[str, Any], sigma: float = 1e-3) -> dict[str, Any]:
    """Add Gaussian differential privacy noise."""

    import torch

    return {key: value + torch.randn_like(value) * sigma for key, value in update.items()}


class FederatedServer:
    """Server coordinator for cross-silo and cross-device FL."""

    def __init__(self, clients: Iterable[FederatedClient], dp_sigma: float = 0.0) -> None:
        self.clients = list(clients)
        self.dp_sigma = dp_sigma

    def sample_clients(self, fraction: float = 1.0) -> list[FederatedClient]:
        count = max(1, int(len(self.clients) * fraction))
        return random.sample(self.clients, min(count, len(self.clients)))

    def aggregate(self, updates: Iterable[Mapping[str, Any]], selected: Iterable[FederatedClient]) -> dict[str, Any]:
        processed = [add_dp_noise(update, self.dp_sigma) if self.dp_sigma > 0 else dict(update) for update in updates]
        return secure_average(processed, [client.weight for client in selected])

    def personalize(self, global_state: Mapping[str, Any], local_adapter: Any) -> Any:
        """Apply a local adaptation hook for personalized FL."""

        return local_adapter(global_state)
