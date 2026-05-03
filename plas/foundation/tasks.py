"""Universal task support and multi-task model factories."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Mapping


class VisionTask(str, Enum):
    """Supported computer vision task families."""

    CLASSIFICATION = "classification"
    DETECTION = "detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    PANOPTIC_SEGMENTATION = "panoptic_segmentation"
    POSE_ESTIMATION = "pose_estimation"
    ORIENTED_BOXES = "oriented_bounding_boxes"
    TRACKING = "multi_object_tracking"
    VISUAL_GROUNDING = "visual_grounding"
    REFERRING_EXPRESSION = "referring_expression_comprehension"
    CAPTIONING = "image_captioning"
    OCR = "ocr"
    DOCUMENT_UNDERSTANDING = "document_understanding"
    VQA = "visual_question_answering"
    DEPTH = "depth_estimation"
    OPTICAL_FLOW = "optical_flow"
    OBJECT_3D = "3d_object_detection"
    VIDEO_UNDERSTANDING = "video_understanding"


@dataclass(frozen=True)
class TaskSpec:
    """Task head description."""

    task: VisionTask
    num_classes: int = 80
    output_dim: int = 256
    stride: int = 4
    metadata: dict[str, Any] = field(default_factory=dict)


def get_default_task_specs(num_classes: int = 80) -> dict[VisionTask, TaskSpec]:
    """Return a complete set of default task specs."""

    return {
        VisionTask.CLASSIFICATION: TaskSpec(VisionTask.CLASSIFICATION, num_classes=num_classes),
        VisionTask.DETECTION: TaskSpec(VisionTask.DETECTION, num_classes=num_classes),
        VisionTask.INSTANCE_SEGMENTATION: TaskSpec(VisionTask.INSTANCE_SEGMENTATION, num_classes=num_classes, output_dim=32),
        VisionTask.SEMANTIC_SEGMENTATION: TaskSpec(VisionTask.SEMANTIC_SEGMENTATION, num_classes=num_classes),
        VisionTask.PANOPTIC_SEGMENTATION: TaskSpec(VisionTask.PANOPTIC_SEGMENTATION, num_classes=num_classes),
        VisionTask.POSE_ESTIMATION: TaskSpec(VisionTask.POSE_ESTIMATION, num_classes=17, output_dim=34),
        VisionTask.ORIENTED_BOXES: TaskSpec(VisionTask.ORIENTED_BOXES, num_classes=num_classes, output_dim=5),
        VisionTask.TRACKING: TaskSpec(VisionTask.TRACKING, num_classes=num_classes, output_dim=128),
        VisionTask.VISUAL_GROUNDING: TaskSpec(VisionTask.VISUAL_GROUNDING, num_classes=num_classes),
        VisionTask.REFERRING_EXPRESSION: TaskSpec(VisionTask.REFERRING_EXPRESSION, num_classes=num_classes),
        VisionTask.CAPTIONING: TaskSpec(VisionTask.CAPTIONING, num_classes=32000),
        VisionTask.OCR: TaskSpec(VisionTask.OCR, num_classes=128),
        VisionTask.DOCUMENT_UNDERSTANDING: TaskSpec(VisionTask.DOCUMENT_UNDERSTANDING, num_classes=512),
        VisionTask.VQA: TaskSpec(VisionTask.VQA, num_classes=32000),
        VisionTask.DEPTH: TaskSpec(VisionTask.DEPTH, num_classes=1, output_dim=1),
        VisionTask.OPTICAL_FLOW: TaskSpec(VisionTask.OPTICAL_FLOW, num_classes=2, output_dim=2),
        VisionTask.OBJECT_3D: TaskSpec(VisionTask.OBJECT_3D, num_classes=num_classes, output_dim=10),
        VisionTask.VIDEO_UNDERSTANDING: TaskSpec(VisionTask.VIDEO_UNDERSTANDING, num_classes=400),
    }


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Universal task heads require PyTorch") from exc
    return torch, nn


class TaskHeadBuilder:
    """Factory for task-specific heads over shared feature maps."""

    def __init__(self, channels: int) -> None:
        self.channels = channels

    def build(self, spec: TaskSpec) -> Any:
        torch, nn = _require_torch()
        c = self.channels
        if spec.task == VisionTask.CLASSIFICATION:
            return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, spec.num_classes))
        if spec.task in {VisionTask.SEMANTIC_SEGMENTATION, VisionTask.PANOPTIC_SEGMENTATION}:
            return nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, spec.num_classes, 1))
        if spec.task == VisionTask.INSTANCE_SEGMENTATION:
            return nn.ModuleDict(
                {
                    "mask_logits": nn.Conv2d(c, spec.output_dim, 1),
                    "classes": nn.Conv2d(c, spec.num_classes, 1),
                    "boxes": nn.Conv2d(c, 4, 1),
                }
            )
        if spec.task == VisionTask.POSE_ESTIMATION:
            return nn.Conv2d(c, spec.output_dim, 1)
        if spec.task == VisionTask.ORIENTED_BOXES:
            return nn.ModuleDict({"classes": nn.Conv2d(c, spec.num_classes, 1), "obb": nn.Conv2d(c, 5, 1)})
        if spec.task == VisionTask.DEPTH:
            return nn.Sequential(nn.Conv2d(c, c // 2, 3, padding=1), nn.SiLU(), nn.Conv2d(c // 2, 1, 1))
        if spec.task == VisionTask.OPTICAL_FLOW:
            return nn.Sequential(nn.Conv2d(c * 2, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, 2, 1))
        if spec.task == VisionTask.VIDEO_UNDERSTANDING:
            return nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(c, spec.num_classes))
        if spec.task in {
            VisionTask.CAPTIONING,
            VisionTask.OCR,
            VisionTask.DOCUMENT_UNDERSTANDING,
            VisionTask.VQA,
            VisionTask.VISUAL_GROUNDING,
            VisionTask.REFERRING_EXPRESSION,
        }:
            return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, spec.output_dim))
        if spec.task == VisionTask.OBJECT_3D:
            return nn.ModuleDict({"classes": nn.Conv2d(c, spec.num_classes, 1), "box3d": nn.Conv2d(c, spec.output_dim, 1)})
        if spec.task == VisionTask.TRACKING:
            return nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, spec.output_dim, 1))
        raise ValueError(f"Unsupported task: {spec.task}")


class UniversalVisionModel:
    """A shared-backbone, task-head container.

    This wrapper is intentionally framework-light: the backbone may be a
    Plasformers detector feature extractor, a VLM vision tower, or a plugin.
    """

    def __init__(self, backbone: Any, task_heads: Mapping[VisionTask, Any]) -> None:
        torch, nn = _require_torch()

        class _Model(nn.Module):
            def __init__(self, inner_backbone: Any, heads: Mapping[VisionTask, Any]) -> None:
                super().__init__()
                self.backbone = inner_backbone
                self.heads = nn.ModuleDict({task.value: head for task, head in heads.items()})

            def forward(self, images: Any, tasks: Iterable[VisionTask] | None = None, **kwargs: Any) -> Dict[str, Any]:
                features = self.backbone(images)
                if isinstance(features, (list, tuple)):
                    feature = features[-1]
                elif isinstance(features, dict) and "features" in features:
                    feature = features["features"][-1]
                else:
                    feature = features
                selected = tasks or [VisionTask(key) for key in self.heads.keys()]
                outputs: Dict[str, Any] = {}
                for task in selected:
                    head = self.heads[task.value]
                    if task == VisionTask.OPTICAL_FLOW:
                        other = kwargs.get("paired_features", feature)
                        outputs[task.value] = head(torch.cat([feature, other], dim=1))
                    elif isinstance(head, nn.ModuleDict):
                        outputs[task.value] = {key: module(feature) for key, module in head.items()}
                    else:
                        outputs[task.value] = head(feature)
                return outputs

        self.module = _Model(backbone, task_heads)

    def as_module(self) -> Any:
        """Return the underlying torch module."""

        return self.module


def build_task_heads(specs: Iterable[TaskSpec], channels: int) -> dict[VisionTask, Any]:
    """Build task heads for a shared feature dimension."""

    builder = TaskHeadBuilder(channels)
    return {spec.task: builder.build(spec) for spec in specs}
