"""Self-supervised learning objectives and pretraining recipes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class SelfSupervisedMethod(str, Enum):
    """Supported self-supervised learning methods."""

    MAE = "masked_autoencoding"
    CONTRASTIVE = "contrastive"
    DINO = "dino"
    BYOL = "byol"
    SIMCLR = "simclr"
    MOCO = "moco"
    IJEPA = "ijepa"


@dataclass(frozen=True)
class SSLConfig:
    """Self-supervised pretraining configuration."""

    method: SelfSupervisedMethod
    projection_dim: int = 256
    hidden_dim: int = 4096
    mask_ratio: float = 0.75
    temperature: float = 0.2
    momentum: float = 0.996


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("SSL objectives require PyTorch") from exc
    return torch, nn


def simclr_loss(z1: Any, z2: Any, temperature: float = 0.2) -> Any:
    """NT-Xent contrastive loss."""

    torch, _ = _require_torch()
    import torch.nn.functional as F

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def byol_loss(prediction: Any, target: Any) -> Any:
    """BYOL cosine regression loss."""

    import torch.nn.functional as F

    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return 2 - 2 * (prediction * target).sum(dim=-1).mean()


def dino_loss(student_logits: Any, teacher_logits: Any, temperature: float = 0.1) -> Any:
    """DINO-style teacher-student cross entropy."""

    import torch.nn.functional as F

    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    student_log = F.log_softmax(student_logits / temperature, dim=-1)
    return -(teacher_prob * student_log).sum(dim=-1).mean()


def mae_reconstruction_loss(reconstruction: Any, target: Any, mask: Any) -> Any:
    """Masked autoencoding reconstruction loss over masked positions."""

    return ((reconstruction - target) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)


def ijepa_predictive_loss(predicted_context: Any, target_context: Any) -> Any:
    """I-JEPA-inspired predictive representation loss."""

    import torch.nn.functional as F

    return F.smooth_l1_loss(predicted_context, target_context.detach())


class SSLObjective:
    """Dispatch self-supervised losses by method."""

    def __init__(self, config: SSLConfig) -> None:
        self.config = config

    def __call__(self, outputs: Mapping[str, Any]) -> Any:
        method = self.config.method
        if method in {SelfSupervisedMethod.CONTRASTIVE, SelfSupervisedMethod.SIMCLR, SelfSupervisedMethod.MOCO}:
            return simclr_loss(outputs["z1"], outputs["z2"], self.config.temperature)
        if method == SelfSupervisedMethod.BYOL:
            return byol_loss(outputs["prediction"], outputs["target"])
        if method == SelfSupervisedMethod.DINO:
            return dino_loss(outputs["student"], outputs["teacher"], self.config.temperature)
        if method == SelfSupervisedMethod.MAE:
            return mae_reconstruction_loss(outputs["reconstruction"], outputs["target"], outputs["mask"])
        if method == SelfSupervisedMethod.IJEPA:
            return ijepa_predictive_loss(outputs["prediction"], outputs["target"])
        raise ValueError(f"Unsupported SSL method: {method}")
