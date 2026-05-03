"""Continual learning, domain adaptation, replay buffers, and EWC."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


class ReplayBuffer:
    """Reservoir-style replay buffer."""

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.items: deque[Any] = deque(maxlen=capacity)

    def add(self, item: Any) -> None:
        self.items.append(item)

    def extend(self, items: Iterable[Any]) -> None:
        for item in items:
            self.add(item)

    def sample(self, batch_size: int) -> list[Any]:
        return random.sample(list(self.items), min(batch_size, len(self.items)))


@dataclass
class EWCState:
    """Elastic weight consolidation state."""

    params: dict[str, Any]
    fisher: dict[str, Any]
    strength: float = 0.4


def ewc_penalty(model: Any, state: EWCState) -> Any:
    import torch

    loss = torch.tensor(0.0)
    for name, param in model.named_parameters():
        if name in state.params and name in state.fisher:
            loss = loss.to(param.device) + (state.fisher[name] * (param - state.params[name]).pow(2)).sum()
    return loss * state.strength


def incremental_class_mapping(old_classes: Iterable[str], new_classes: Iterable[str]) -> dict[str, int]:
    classes = list(dict.fromkeys(list(old_classes) + list(new_classes)))
    return {name: idx for idx, name in enumerate(classes)}


def domain_adaptation_weight(source_loss: float, target_loss: float) -> float:
    return float(target_loss / max(source_loss + target_loss, 1e-6))
