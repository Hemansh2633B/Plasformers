"""Large-scale distributed training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DistributedPlan:
    """Distributed strategy plan."""

    strategy: str = "ddp"
    world_size: int = 1
    mixed_precision: str = "bf16"
    fsdp: bool = False
    deepspeed: bool = False
    zero_stage: int = 0
    pipeline_parallel: int = 1
    tensor_parallel: int = 1
    sequence_parallel: bool = False


def wrap_fsdp(model: Any, mixed_precision: str = "bf16") -> Any:
    """Wrap a model with PyTorch FSDP."""

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError as exc:
        raise RuntimeError("FSDP requires a PyTorch build with torch.distributed.fsdp") from exc
    return FSDP(model)


def initialize_deepspeed(model: Any, optimizer: Any, config: dict[str, Any]) -> Any:
    """Initialize DeepSpeed ZeRO/pipeline runtime."""

    try:
        import deepspeed
    except ImportError as exc:
        raise RuntimeError("DeepSpeed integration requires deepspeed") from exc
    engine, optimizer, _, scheduler = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
    return {"engine": engine, "optimizer": optimizer, "scheduler": scheduler}


def deepspeed_zero_config(stage: int = 2, offload: bool = False) -> dict[str, Any]:
    cfg: dict[str, Any] = {"zero_optimization": {"stage": stage}, "bf16": {"enabled": True}}
    if offload:
        cfg["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}
    return cfg


def tensor_parallel_plan(parts: int) -> dict[str, Any]:
    return {"tensor_parallel": max(1, parts), "requires": ["device_mesh", "collective_matmul"]}


def pipeline_parallel_plan(stages: int) -> dict[str, Any]:
    return {"pipeline_parallel": max(1, stages), "schedule": "1f1b"}
