"""Export helpers for ONNX, TensorRT, CoreML, and TorchScript workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn

from .model import PlasformersDetector, build_plasformers


class ExportWrapper(nn.Module):
    """Tuple-output wrapper for graph export."""

    def __init__(self, model: PlasformersDetector) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        self.model.set_export_mode(True)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        return self.model.export_forward(x)


def load_checkpoint(model: nn.Module, checkpoint: str | Path) -> None:
    """Load a checkpoint with common key conventions."""

    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)


def export_onnx(
    model: PlasformersDetector,
    output: str | Path,
    input_shape: Sequence[int] = (1, 3, 640, 640),
    opset: int = 17,
    dynamic: bool = True,
) -> None:
    """Export a Plasformers model to ONNX."""

    wrapper = ExportWrapper(model)
    dummy = torch.randn(*input_shape)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}}
        for idx in range(20):
            dynamic_axes[f"out_{idx}"] = {0: "batch", 2: "level_height", 3: "level_width"}
    torch.onnx.export(
        wrapper,
        dummy,
        str(output),
        input_names=["images"],
        output_names=[f"out_{idx}" for idx in range(20)],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )


def trace_torchscript(
    model: PlasformersDetector,
    output: str | Path,
    input_shape: Sequence[int] = (1, 3, 640, 640),
) -> None:
    """Trace an export-mode TorchScript graph."""

    wrapper = ExportWrapper(model)
    dummy = torch.randn(*input_shape)
    traced = torch.jit.trace(wrapper, dummy, strict=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output))


def build_export_model(variant: str, num_classes: int, checkpoint: str | None = None) -> PlasformersDetector:
    """Construct and optionally load a model for export."""

    model = build_plasformers(variant, num_classes=num_classes, export_mode=True)
    if checkpoint:
        load_checkpoint(model, checkpoint)
    model.eval()
    model.set_export_mode(True)
    return model
