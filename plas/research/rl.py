"""Reinforcement learning integration for policies and controllers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass
class PolicyAction:
    """RL action for augmentation, assignment, or architecture control."""

    name: str
    params: dict[str, Any]


class EpsilonGreedyController:
    """Simple policy controller for RL-based search loops."""

    def __init__(self, actions: Sequence[PolicyAction], epsilon: float = 0.1) -> None:
        self.actions = list(actions)
        self.epsilon = epsilon
        self.values = {action.name: 0.0 for action in self.actions}
        self.counts = {action.name: 0 for action in self.actions}

    def select(self) -> PolicyAction:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.actions, key=lambda action: self.values[action.name])

    def update(self, action: PolicyAction, reward: float) -> None:
        self.counts[action.name] += 1
        count = self.counts[action.name]
        self.values[action.name] += (reward - self.values[action.name]) / count


def rl_augmentation_actions() -> list[PolicyAction]:
    return [
        PolicyAction("mosaic_prob", {"low": 0.0, "high": 1.0}),
        PolicyAction("copy_paste_prob", {"low": 0.0, "high": 0.7}),
        PolicyAction("mixup_prob", {"low": 0.0, "high": 0.4}),
        PolicyAction("scale_range", {"low": 0.25, "high": 2.0}),
    ]


def vision_for_control_pipeline(perception_fn: Callable[[Any], Mapping[str, Any]], policy_fn: Callable[[Mapping[str, Any]], Any], observation: Any) -> Any:
    return policy_fn(perception_fn(observation))
