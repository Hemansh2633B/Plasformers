"""Loss utilities for Plasformers training."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


def varifocal_loss(
    logits: Tensor,
    targets: Tensor,
    target_scores: Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    normalizer: float | Tensor = 1.0,
) -> Tensor:
    """Varifocal classification loss.

    Args:
        logits: B x N x C logits.
        targets: B x N x C binary class targets.
        target_scores: B x N x C IoU-aware soft targets.
    """

    pred_score = logits.sigmoid()
    focal_weight = target_scores * targets + alpha * (pred_score - target_scores).abs().pow(gamma) * (
        1.0 - targets
    )
    loss = F.binary_cross_entropy_with_logits(logits, target_scores, reduction="none") * focal_weight
    return loss.sum() / torch.as_tensor(normalizer, device=logits.device).clamp_min(1.0)


def distribution_focal_loss(pred_dist: Tensor, target: Tensor, reg_max: int) -> Tensor:
    """Distribution Focal Loss for discrete box distance distributions.

    Args:
        pred_dist: N x 4 x (reg_max + 1) logits.
        target: N x 4 continuous target distances in [0, reg_max].
    """

    target = target.clamp(min=0, max=reg_max - 1e-3)
    left = target.floor().long()
    right = left + 1
    weight_left = right.float() - target
    weight_right = target - left.float()
    loss_left = F.cross_entropy(pred_dist.flatten(0, 1), left.flatten(), reduction="none").view_as(target)
    loss_right = F.cross_entropy(pred_dist.flatten(0, 1), right.flatten(), reduction="none").view_as(target)
    return (loss_left * weight_left + loss_right * weight_right).mean()


def _box_area(boxes: Tensor) -> Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp_min(0) * (boxes[..., 3] - boxes[..., 1]).clamp_min(0)


def bbox_iou(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Pairwise aligned IoU for xyxy boxes."""

    lt = torch.maximum(pred[..., :2], target[..., :2])
    rb = torch.minimum(pred[..., 2:], target[..., 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    union = _box_area(pred) + _box_area(target) - inter
    return inter / union.clamp_min(eps)


def ciou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Complete IoU loss for aligned xyxy boxes."""

    iou = bbox_iou(pred, target, eps)
    pred_center = (pred[..., :2] + pred[..., 2:]) * 0.5
    target_center = (target[..., :2] + target[..., 2:]) * 0.5
    center_dist = ((pred_center - target_center) ** 2).sum(dim=-1)

    enc_lt = torch.minimum(pred[..., :2], target[..., :2])
    enc_rb = torch.maximum(pred[..., 2:], target[..., 2:])
    enc_diag = ((enc_rb - enc_lt) ** 2).sum(dim=-1).clamp_min(eps)

    pred_wh = (pred[..., 2:] - pred[..., :2]).clamp_min(eps)
    target_wh = (target[..., 2:] - target[..., :2]).clamp_min(eps)
    v = (4.0 / (torch.pi**2)) * (
        torch.atan(target_wh[..., 0] / target_wh[..., 1])
        - torch.atan(pred_wh[..., 0] / pred_wh[..., 1])
    ).pow(2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)
    ciou = iou - center_dist / enc_diag - alpha * v
    return 1.0 - ciou


def siou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Shape-aware IoU loss with a stable SIoU-style penalty."""

    iou = bbox_iou(pred, target, eps)
    pred_center = (pred[..., :2] + pred[..., 2:]) * 0.5
    target_center = (target[..., :2] + target[..., 2:]) * 0.5
    pred_wh = (pred[..., 2:] - pred[..., :2]).clamp_min(eps)
    target_wh = (target[..., 2:] - target[..., :2]).clamp_min(eps)
    center_cost = ((pred_center - target_center) / target_wh).pow(2).sum(dim=-1)
    shape_cost = (1.0 - torch.exp(-torch.abs(pred_wh - target_wh) / target_wh)).pow(2).sum(dim=-1)
    return 1.0 - iou + 0.25 * center_cost + 0.5 * shape_cost


def auxiliary_consistency_loss(
    main_logits: Tensor,
    aux_logits: Tensor,
    temperature: float = 2.0,
) -> Tensor:
    """KL consistency loss from one-to-many auxiliary branch to main branch."""

    main_prob = torch.softmax(main_logits.detach() / temperature, dim=-1)
    aux_log_prob = torch.log_softmax(aux_logits / temperature, dim=-1)
    return F.kl_div(aux_log_prob, main_prob, reduction="batchmean") * (temperature**2)


def distances_to_boxes(points: Tensor, distances: Tensor) -> Tensor:
    """Convert ltrb distances to xyxy boxes.

    Args:
        points: N x 2 center points.
        distances: B x N x 4 or N x 4 distances.
    """

    x1y1 = points[..., :2] - distances[..., :2]
    x2y2 = points[..., :2] + distances[..., 2:]
    return torch.cat([x1y1, x2y2], dim=-1)


def make_anchor_points(
    feature_shapes: Tuple[Tuple[int, int], ...],
    strides: Tuple[int, ...] = (8, 16, 32, 64, 128),
    device: torch.device | None = None,
) -> Tuple[Tensor, Tensor]:
    """Create anchor-free center points for P3-P7."""

    points = []
    stride_values = []
    for (height, width), stride in zip(feature_shapes, strides):
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        point = torch.stack([(x + 0.5) * stride, (y + 0.5) * stride], dim=-1).reshape(-1, 2)
        points.append(point)
        stride_values.append(torch.full((height * width, 1), stride, device=device))
    return torch.cat(points, dim=0).float(), torch.cat(stride_values, dim=0).float()
