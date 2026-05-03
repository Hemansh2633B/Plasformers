"""Explainability toolkit for Plasformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Explanation:
    """Explanation result."""

    heatmap: Any
    method: str
    target_layer: str


class GradCAM:
    """Grad-CAM implementation for arbitrary target layers."""

    def __init__(self, model: Any, target_layer: str) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        layer = dict(model.named_modules())[target_layer]
        layer.register_forward_hook(self._forward_hook)
        layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        self.activations = output

    def _backward_hook(self, _module: Any, _grad_input: Any, grad_output: Any) -> None:
        self.gradients = grad_output[0]

    def __call__(self, images: Any, score_fn: Any | None = None) -> Explanation:
        import torch
        import torch.nn.functional as F

        self.model.zero_grad(set_to_none=True)
        outputs = self.model(images)
        if score_fn is None:
            score = sum(pred["cls_logits"].sigmoid().max() for pred in outputs["outputs"])
        else:
            score = score_fn(outputs)
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True).relu()
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam / cam.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return Explanation(cam.detach(), "gradcam", self.target_layer)


def eigen_cam(features: Any) -> Any:
    """Compute Eigen-CAM from feature maps."""

    import torch
    import torch.nn.functional as F

    b, c, h, w = features.shape
    flat = features.reshape(b, c, h * w)
    heatmaps = []
    for item in flat:
        u, _, _ = torch.pca_lowrank(item.transpose(0, 1), q=1)
        heatmap = u[:, 0].reshape(h, w).abs()
        heatmap = heatmap / heatmap.max().clamp_min(1e-6)
        heatmaps.append(heatmap)
    return F.interpolate(torch.stack(heatmaps).unsqueeze(1), scale_factor=1.0)


def attention_rollout(attention_matrices: list[Any]) -> Any:
    """Roll out attention matrices across layers."""

    import torch

    if not attention_matrices:
        raise ValueError("attention_matrices cannot be empty")
    result = torch.eye(attention_matrices[0].shape[-1], device=attention_matrices[0].device)
    for attn in attention_matrices:
        attn = attn.mean(dim=1) if attn.ndim == 4 else attn
        attn = attn + torch.eye(attn.shape[-1], device=attn.device)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        result = attn.matmul(result)
    return result


def failure_analysis(predictions: list[Any], targets: list[Any]) -> dict[str, int]:
    """Return coarse failure buckets for detections."""

    return {
        "false_positive": max(0, len(predictions) - len(targets)),
        "false_negative": max(0, len(targets) - len(predictions)),
        "matched": min(len(predictions), len(targets)),
    }
