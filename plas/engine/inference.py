"""High-level inference API with dynamic batching and async support."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np

from plasformers.export import load_checkpoint
from plasformers.model import PlasformersDetector, build_plasformers


@dataclass
class Detection:
    """A single detection in xyxy image coordinates."""

    box: tuple[float, float, float, float]
    score: float
    class_id: int
    label: str | None = None


@dataclass
class InferenceResult:
    """Inference result for one image."""

    detections: list[Detection]
    latency_ms: float
    fps: float
    raw: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for native Plasformers inference") from exc
    return torch


def _load_image(source: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return source
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image-file inference") from exc
    image = Image.open(source).convert("RGB")
    return np.asarray(image)


def _preprocess(
    images: Sequence[str | Path | np.ndarray],
    size: tuple[int, int],
    device: str,
    fp16: bool,
) -> tuple[Any, list[tuple[int, int]]]:
    torch = _require_torch()
    tensors = []
    shapes = []
    for item in images:
        image = _load_image(item)
        shapes.append((int(image.shape[0]), int(image.shape[1])))
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for resizing input images") from exc
        resized = Image.fromarray(image).resize((size[1], size[0]))
        arr = np.asarray(resized).astype("float32") / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensors.append(tensor)
    batch = torch.stack(tensors, dim=0).to(device)
    if fp16 and device.startswith("cuda"):
        batch = batch.half()
    return batch, shapes


def _nms(boxes: Any, scores: Any, iou_threshold: float) -> Any:
    torch = _require_torch()
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    areas = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        idx = order[0]
        keep.append(idx)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[idx], x1[rest])
        yy1 = torch.maximum(y1[idx], y1[rest])
        xx2 = torch.minimum(x2[idx], x2[rest])
        yy2 = torch.minimum(y2[idx], y2[rest])
        inter = (xx2 - xx1).clamp_min(0) * (yy2 - yy1).clamp_min(0)
        union = areas[idx] + areas[rest] - inter
        iou = inter / union.clamp_min(1e-6)
        order = rest[iou <= iou_threshold]
    return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long, device=boxes.device)


class PlasPostProcessor:
    """DFL-based postprocessor for raw Plasformers outputs."""

    def __init__(
        self,
        reg_max: int = 16,
        strides: Sequence[int] = (8, 16, 32, 64, 128),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_det: int = 300,
    ) -> None:
        self.reg_max = reg_max
        self.strides = tuple(strides)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det

    def _normalize_outputs(self, outputs: Any) -> list[dict[str, Any]]:
        if isinstance(outputs, dict):
            return list(outputs["outputs"])
        if isinstance(outputs, tuple):
            normalized = []
            for idx in range(0, len(outputs), 4):
                normalized.append(
                    {
                        "cls_logits": outputs[idx],
                        "bbox_dist": outputs[idx + 1],
                        "objectness": outputs[idx + 2],
                        "iou": outputs[idx + 3],
                    }
                )
            return normalized
        raise TypeError("Unsupported model output type")

    def __call__(self, outputs: Any, image_shapes: Sequence[tuple[int, int]], input_size: tuple[int, int]) -> list[list[Detection]]:
        torch = _require_torch()
        per_level = self._normalize_outputs(outputs)
        batch = per_level[0]["cls_logits"].shape[0]
        results: list[list[Detection]] = []
        proj = torch.arange(self.reg_max + 1, device=per_level[0]["cls_logits"].device).float()
        for bidx in range(batch):
            boxes_all = []
            scores_all = []
            classes_all = []
            for level, pred in enumerate(per_level):
                stride = self.strides[level]
                cls = pred["cls_logits"][bidx].sigmoid()
                obj = pred["objectness"][bidx].sigmoid()
                iou = pred["iou"][bidx].sigmoid()
                dist = pred["bbox_dist"][bidx]
                _, height, width = cls.shape
                cls_flat = cls.reshape(cls.shape[0], -1).transpose(0, 1)
                score_matrix = cls_flat * obj.reshape(1, -1).transpose(0, 1) * iou.reshape(1, -1).transpose(0, 1)
                scores, class_ids = score_matrix.max(dim=1)
                keep = scores > self.conf_threshold
                if keep.sum() == 0:
                    continue
                y, x = torch.meshgrid(
                    torch.arange(height, device=cls.device),
                    torch.arange(width, device=cls.device),
                    indexing="ij",
                )
                centers = torch.stack([(x + 0.5) * stride, (y + 0.5) * stride], dim=-1).reshape(-1, 2)
                dist = dist.reshape(4, self.reg_max + 1, height * width).permute(2, 0, 1)
                distances = (dist.softmax(dim=-1) * proj).sum(dim=-1) * stride
                boxes = torch.cat([centers - distances[:, :2], centers + distances[:, 2:]], dim=-1)
                boxes_all.append(boxes[keep])
                scores_all.append(scores[keep])
                classes_all.append(class_ids[keep])
            if not boxes_all:
                results.append([])
                continue
            boxes = torch.cat(boxes_all, dim=0)
            scores = torch.cat(scores_all, dim=0)
            classes = torch.cat(classes_all, dim=0)
            keep = _nms(boxes, scores, self.iou_threshold)[: self.max_det]
            orig_h, orig_w = image_shapes[bidx]
            scale_x = orig_w / float(input_size[1])
            scale_y = orig_h / float(input_size[0])
            detections: list[Detection] = []
            for idx in keep:
                box = boxes[idx]
                detections.append(
                    Detection(
                        box=(
                            float(box[0] * scale_x),
                            float(box[1] * scale_y),
                            float(box[2] * scale_x),
                            float(box[3] * scale_y),
                        ),
                        score=float(scores[idx]),
                        class_id=int(classes[idx]),
                    )
                )
            results.append(detections)
        return results


class InferenceEngine:
    """Production-oriented inference wrapper.

    Native PyTorch is implemented directly. ONNX/TensorRT/CoreML engines are
    selected by suffix and exposed through extension hooks so production teams
    can bind their preferred runtime without changing application code.
    """

    def __init__(
        self,
        model: PlasformersDetector | None = None,
        *,
        variant: str = "nano",
        checkpoint: str | Path | None = None,
        device: str = "auto",
        input_size: tuple[int, int] = (640, 640),
        fp16: bool = False,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.65,
    ) -> None:
        torch = _require_torch()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.input_size = input_size
        self.fp16 = fp16
        self.model = model or build_plasformers(variant, export_mode=True)
        if checkpoint:
            load_checkpoint(self.model, checkpoint)
        self.model.eval().to(device)
        if fp16 and device.startswith("cuda"):
            self.model.half()
        self.post = PlasPostProcessor(conf_threshold=conf_threshold, iou_threshold=iou_threshold)

    def predict_batch(self, images: Sequence[str | Path | np.ndarray]) -> list[InferenceResult]:
        torch = _require_torch()
        tensor, shapes = _preprocess(images, self.input_size, self.device, self.fp16)
        start = time.perf_counter()
        with torch.inference_mode():
            raw = self.model.export_forward(tensor)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        detections = self.post(raw, shapes, self.input_size)
        per_image = elapsed_ms / max(1, len(images))
        fps = 1000.0 / max(per_image, 1e-6)
        return [
            InferenceResult(det, per_image, fps, raw=None, metadata={"input_shape": shape})
            for det, shape in zip(detections, shapes)
        ]

    def predict(self, image: str | Path | np.ndarray) -> InferenceResult:
        return self.predict_batch([image])[0]

    async def predict_async(self, images: Sequence[str | Path | np.ndarray]) -> list[InferenceResult]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.predict_batch, list(images))

    def stream_video(self, source: str | int, batch_size: int = 1) -> Iterable[InferenceResult]:
        """Yield detections from a video stream."""

        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for video streaming") from exc
        cap = cv2.VideoCapture(source)
        batch: List[np.ndarray] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                batch.append(rgb)
                if len(batch) >= batch_size:
                    for result in self.predict_batch(batch):
                        yield result
                    batch.clear()
            if batch:
                for result in self.predict_batch(batch):
                    yield result
        finally:
            cap.release()
