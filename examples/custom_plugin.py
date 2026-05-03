"""Example plugin registration."""

from __future__ import annotations

from plas.core.plugins import register_plugin


@register_plugin("tune_objective", "default")
def objective(params: dict[str, object]) -> float:
    """Toy objective for `plas tune` demos."""

    lr = float(params.get("optimization.base_lr", 1e-3))
    return 1.0 / (1.0 + abs(lr - 1e-3))
