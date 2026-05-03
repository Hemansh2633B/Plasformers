"""TPU-first runtime helpers for PJRT and XLA."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class TPUConfig:
    """TPU runtime configuration."""

    version: str = "v5e"
    pjrt_device: str = "TPU"
    bf16: bool = True
    static_shapes: bool = True
    shard_count: int = 8


def configure_pjrt(config: TPUConfig | None = None) -> TPUConfig:
    cfg = config or TPUConfig()
    os.environ.setdefault("PJRT_DEVICE", cfg.pjrt_device)
    os.environ.setdefault("XLA_USE_BF16", "1" if cfg.bf16 else "0")
    return cfg


def xla_device() -> Any:
    try:
        import torch_xla.core.xla_model as xm
    except ImportError as exc:
        raise RuntimeError("TPU runtime requires torch_xla") from exc
    return xm.xla_device()


def mark_step() -> None:
    try:
        import torch_xla.core.xla_model as xm
    except ImportError:
        return
    xm.mark_step()


def shard_dataloader(dataloader: Iterable[Any]) -> Any:
    try:
        import torch_xla.distributed.parallel_loader as pl
    except ImportError as exc:
        raise RuntimeError("Sharded TPU dataloaders require torch_xla") from exc
    return pl.MpDeviceLoader(dataloader, xla_device())
