"""Knowledge distillation suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass
class DistillationWeights:
    """Weights for distillation loss terms."""

    logits: float = 1.0
    features: float = 1.0
    relational: float = 0.25
    self_distill: float = 0.1
    temperature: float = 2.0


class DistillationLoss:
    """Logit, feature, relational, and self-distillation losses."""

    def __init__(self, weights: DistillationWeights | None = None) -> None:
        self.weights = weights or DistillationWeights()

    def __call__(self, student: Mapping[str, Any], teacher: Mapping[str, Any]) -> Any:
        import torch
        import torch.nn.functional as F

        total = torch.tensor(0.0, device=self._device(student))
        temp = self.weights.temperature
        if "logits" in student and "logits" in teacher:
            s_log = F.log_softmax(student["logits"] / temp, dim=-1)
            t_prob = F.softmax(teacher["logits"].detach() / temp, dim=-1)
            total = total + self.weights.logits * F.kl_div(s_log, t_prob, reduction="batchmean") * (temp**2)
        if "features" in student and "features" in teacher:
            for s_feat, t_feat in zip(student["features"], teacher["features"]):
                total = total + self.weights.features * F.smooth_l1_loss(s_feat, t_feat.detach())
        if "relations" in student and "relations" in teacher:
            total = total + self.weights.relational * F.mse_loss(
                self._gram(student["relations"]),
                self._gram(teacher["relations"].detach()),
            )
        if "self_teacher_logits" in student and "logits" in student:
            s_log = F.log_softmax(student["logits"] / temp, dim=-1)
            t_prob = F.softmax(student["self_teacher_logits"].detach() / temp, dim=-1)
            total = total + self.weights.self_distill * F.kl_div(s_log, t_prob, reduction="batchmean") * (temp**2)
        return total

    @staticmethod
    def _device(outputs: Mapping[str, Any]) -> Any:
        for value in outputs.values():
            if hasattr(value, "device"):
                return value.device
            if isinstance(value, (list, tuple)) and value and hasattr(value[0], "device"):
                return value[0].device
        import torch

        return torch.device("cpu")

    @staticmethod
    def _gram(features: Any) -> Any:
        import torch.nn.functional as F

        flat = features.flatten(2)
        flat = F.normalize(flat, dim=1)
        return flat.transpose(1, 2).matmul(flat)


class TeacherAdapter:
    """Adapter contract for YOLOv10, RT-DETR, and custom teachers."""

    def __init__(self, model: Any, output_mapper: Any | None = None) -> None:
        self.model = model
        self.output_mapper = output_mapper

    def __call__(self, images: Any) -> Dict[str, Any]:
        with self._inference_context():
            outputs = self.model(images)
        if self.output_mapper is not None:
            return self.output_mapper(outputs)
        return {"raw": outputs}

    @staticmethod
    def _inference_context() -> Any:
        import torch

        return torch.inference_mode()
