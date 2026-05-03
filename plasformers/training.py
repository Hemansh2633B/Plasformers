"""Training helpers for cloud-scale Plasformers runs."""

from __future__ import annotations

import copy
import os
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterable, Mapping

import torch
from torch import Tensor, nn


def move_to_device(batch: Any, device: torch.device | str) -> Any:
    """Recursively move tensors in a batch to a device."""

    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, Mapping):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_to_device(item, device) for item in batch)
    if isinstance(batch, list):
        return [move_to_device(item, device) for item in batch]
    return batch


def setup_distributed(backend: str | None = None) -> Dict[str, int | bool | str]:
    """Initialize torch.distributed from environment variables when available."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not torch.distributed.is_initialized():
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "backend": backend or "none",
    }


def get_default_device(local_rank: int = 0) -> torch.device:
    """Pick CUDA when available, otherwise CPU."""

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def amp_context(device: torch.device | str, enabled: bool = True, dtype: str = "float16") -> Any:
    """Return an autocast context for CUDA/CPU training."""

    device_type = str(device).split(":")[0]
    if not enabled:
        return nullcontext()
    amp_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    if device_type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=True)
    return nullcontext()


class ModelEma:
    """Exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.module if hasattr(model, "module") else model
        ema_state = self.module.state_dict()
        source_state = source.state_dict()
        for key, ema_value in ema_state.items():
            source_value = source_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(source_value)


def xla_device() -> Any:
    """Return a torch_xla device when torch_xla is installed."""

    try:
        import torch_xla.core.xla_model as xm
    except ImportError as exc:
        raise RuntimeError("torch_xla is not installed") from exc
    return xm.xla_device()


def xla_mark_step() -> None:
    """Mark an XLA step when torch_xla is active."""

    try:
        import torch_xla.core.xla_model as xm
    except ImportError:
        return
    xm.mark_step()


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Any],
    criterion: Callable[[Dict[str, Any], Any], Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    scaler: torch.amp.GradScaler | None = None,
    ema: ModelEma | None = None,
    amp: bool = True,
    amp_dtype: str = "float16",
    grad_clip_norm: float | None = 10.0,
) -> Dict[str, float]:
    """Generic mixed-precision training loop.

    The criterion is intentionally injected so projects can plug in COCO,
    aerial, medical, or industrial assignment logic without changing the model.
    """

    model.train()
    total_loss = 0.0
    num_steps = 0
    for batch in dataloader:
        images = move_to_device(batch["images"], device)
        targets = move_to_device(batch["targets"], device)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, enabled=amp, dtype=amp_dtype):
            outputs = model(images)
            loss = criterion(outputs, targets)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        if ema is not None:
            ema.update(model)
        xla_mark_step()
        total_loss += float(loss.detach().cpu())
        num_steps += 1
    return {"loss": total_loss / max(1, num_steps), "steps": float(num_steps)}
