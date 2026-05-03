"""Quantization-aware and post-training quantization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from plasformers.optimization import prepare_qat


def prepare_quantization_aware_training(model: object, backend: str = "fbgemm") -> object:
    """Prepare a model for QAT."""

    return prepare_qat(model, backend=backend)


def post_training_quantize(
    model: object,
    calibration_batches: Iterable[object],
    output: str | Path | None = None,
    backend: str = "fbgemm",
) -> object:
    """Run eager-mode static PTQ calibration and conversion."""

    import torch
    import torch.ao.quantization as quantization

    model.eval()
    model.qconfig = quantization.get_default_qconfig(backend)
    prepared = quantization.prepare(model, inplace=False)
    with torch.inference_mode():
        for batch in calibration_batches:
            images = batch["images"] if isinstance(batch, dict) else batch
            prepared(images)
    quantized = quantization.convert(prepared, inplace=False)
    if output is not None:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": quantized.state_dict()}, out)
    return quantized
