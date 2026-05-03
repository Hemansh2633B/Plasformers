"""Hardware-aware neural architecture search."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping


@dataclass(frozen=True)
class ArchitectureCandidate:
    """Candidate architecture parameters."""

    width_mult: float
    depth_mult: float
    token_grid: int
    pyramid_channels: int
    qat: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ParetoPoint:
    """Architecture search result."""

    candidate: ArchitectureCandidate
    accuracy: float
    latency_ms: float
    memory_mb: float
    device: str


class HardwareAwareNAS:
    """Latency-constrained Pareto frontier optimizer."""

    def __init__(
        self,
        evaluator: Callable[[ArchitectureCandidate], ParetoPoint],
        latency_budget_ms: float,
        device: str = "cuda",
    ) -> None:
        self.evaluator = evaluator
        self.latency_budget_ms = latency_budget_ms
        self.device = device

    def sample(self, qat: bool = False) -> ArchitectureCandidate:
        return ArchitectureCandidate(
            width_mult=random.uniform(0.25, 1.5),
            depth_mult=random.uniform(0.25, 1.5),
            token_grid=random.choice([4, 6, 8, 10, 12]),
            pyramid_channels=random.choice([96, 128, 160, 192, 256, 320]),
            qat=qat,
        )

    def search(self, trials: int = 50, qat: bool = False) -> list[ParetoPoint]:
        points = [self.evaluator(self.sample(qat=qat)) for _ in range(trials)]
        feasible = [point for point in points if point.latency_ms <= self.latency_budget_ms]
        return pareto_frontier(feasible or points)


def dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    return (
        a.accuracy >= b.accuracy
        and a.latency_ms <= b.latency_ms
        and a.memory_mb <= b.memory_mb
        and (a.accuracy > b.accuracy or a.latency_ms < b.latency_ms or a.memory_mb < b.memory_mb)
    )


def pareto_frontier(points: Iterable[ParetoPoint]) -> list[ParetoPoint]:
    pts = list(points)
    return [point for point in pts if not any(dominates(other, point) for other in pts)]


def tpu_search_profile() -> Mapping[str, object]:
    return {"device": "tpu-v5e", "prefer_bfloat16": True, "avoid_dynamic_shapes": True, "shard_batch": True}


def edge_search_profile() -> Mapping[str, object]:
    return {"device": "edge-npu", "prefer_int8": True, "static_input": True, "avoid_grid_sample": True}
