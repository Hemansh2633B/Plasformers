"""Structured pruning and sparsity training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from plasformers.optimization import collect_prunable_convs


def structured_prune(model: object, amount: float = 0.2, dim: int = 0) -> object:
    """Apply L1 structured pruning to eligible convolutions."""

    import torch.nn.utils.prune as prune

    for _, conv in collect_prunable_convs(model):
        prune.ln_structured(conv, name="weight", amount=amount, n=1, dim=dim)
        prune.remove(conv, "weight")
    return model


@dataclass
class SparsityRegularizer:
    """L1 sparsity regularizer for activations or weights."""

    weight: float = 1e-5

    def __call__(self, tensors: Iterable[object]) -> object:
        import torch

        loss = torch.tensor(0.0)
        for tensor in tensors:
            loss = loss.to(tensor.device) + tensor.abs().mean()
        return loss * self.weight


class LayerDropper:
    """Mark repeated blocks for layer dropping.

    The method keeps module names stable and replaces dropped layers with
    Identity modules so checkpoints remain understandable.
    """

    def __init__(self, drop_every: int = 4) -> None:
        self.drop_every = max(2, drop_every)

    def apply(self, sequential: object) -> object:
        import torch.nn as nn

        for idx, _module in enumerate(list(sequential.children())):
            if (idx + 1) % self.drop_every == 0:
                sequential[idx] = nn.Identity()
        return sequential
