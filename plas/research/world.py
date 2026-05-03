"""Latent world-model components for video prediction and planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorldModelConfig:
    """World model configuration."""

    latent_dim: int = 512
    action_dim: int = 16
    horizon: int = 8
    state_layers: int = 4


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("World-model modules require PyTorch") from exc
    return torch, nn


class LatentDynamicsModel:
    """Action-conditioned latent forecasting model."""

    def __init__(self, config: WorldModelConfig | None = None) -> None:
        torch, nn = _require_torch()
        cfg = config or WorldModelConfig()

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transition = nn.GRU(cfg.latent_dim + cfg.action_dim, cfg.latent_dim, cfg.state_layers, batch_first=True)
                self.decoder = nn.Linear(cfg.latent_dim, cfg.latent_dim)

            def forward(self, latents: Any, actions: Any) -> Any:
                x = torch.cat([latents, actions], dim=-1)
                future, _ = self.transition(x)
                return self.decoder(future)

        self.module = _Model()

    def as_module(self) -> Any:
        return self.module


def planning_representation(latents: Any, rewards: Any | None = None) -> dict[str, Any]:
    return {"latents": latents, "rewards": rewards, "state": latents[:, -1] if hasattr(latents, "__getitem__") else latents}
