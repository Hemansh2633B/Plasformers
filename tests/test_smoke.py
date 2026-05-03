"""Smoke tests for Plasformers."""

from __future__ import annotations

import torch

from plasformers import build_plasformers


def test_nano_forward_export_mode() -> None:
    model = build_plasformers("nano", num_classes=3, export_mode=True).eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        outputs = model.export_forward(x)
    assert len(outputs) == 20
    assert outputs[0].shape[1] == 3
