"""Data-centric AI utilities."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def label_correction(
    annotations: Iterable[Mapping[str, object]],
    min_box_size: float = 1.0,
) -> list[dict[str, object]]:
    """Clip invalid boxes and remove impossible annotations."""

    corrected = []
    for ann in annotations:
        item = dict(ann)
        box = list(item.get("bbox", [0, 0, 0, 0]))
        if len(box) != 4:
            continue
        box[2] = max(float(box[2]), min_box_size)
        box[3] = max(float(box[3]), min_box_size)
        item["bbox"] = box
        item["area"] = box[2] * box[3]
        corrected.append(item)
    return corrected


def outlier_scores(features: Sequence[Sequence[float]]) -> list[float]:
    """Compute simple distance-to-centroid outlier scores."""

    if not features:
        return []
    import numpy as np

    matrix = np.asarray(features, dtype="float32")
    center = matrix.mean(axis=0, keepdims=True)
    dist = np.linalg.norm(matrix - center, axis=1)
    return (dist / max(float(dist.max()), 1e-6)).tolist()


def hard_example_mining(examples: Iterable[Mapping[str, object]], top_k: int = 100) -> list[dict[str, object]]:
    """Select high-loss or low-confidence examples."""

    def score(item: Mapping[str, object]) -> float:
        if "loss" in item:
            return float(item["loss"])
        if "confidence" in item:
            return 1.0 - float(item["confidence"])
        return 0.0

    ranked = sorted((dict(item) for item in examples), key=score, reverse=True)
    return ranked[:top_k]


def active_learning_candidates(
    predictions: Iterable[Mapping[str, object]],
    top_k: int = 100,
) -> list[dict[str, object]]:
    """Pick uncertain samples for annotation."""

    def uncertainty(item: Mapping[str, object]) -> float:
        score = float(item.get("score", item.get("confidence", 0.0)))
        margin = float(item.get("margin", 0.0))
        return (1.0 - score) + (1.0 - margin)

    return sorted((dict(item) for item in predictions), key=uncertainty, reverse=True)[:top_k]


def pseudo_label_manifest(
    predictions: Iterable[Mapping[str, object]],
    threshold: float = 0.7,
) -> list[dict[str, object]]:
    """Create a semi-supervised pseudo-label manifest."""

    return [dict(item) for item in predictions if float(item.get("score", 0.0)) >= threshold]
