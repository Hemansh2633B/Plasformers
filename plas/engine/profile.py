"""Profiling utilities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

from plasformers import build_plasformers


@dataclass
class ProfileRecord:
    """Model profile summary."""

    variant: str
    params_m: float
    trainable_params_m: float
    buffers_m: float
    output_path: str | None = None


def profile_model(variant: str = "base", output: str | Path | None = None) -> Dict[str, object]:
    """Profile parameter and optional torch profiler traces."""

    model = build_plasformers(variant, export_mode=True)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    record = ProfileRecord(variant, params / 1e6, trainable / 1e6, buffers / 1e6, None)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(asdict(record)), encoding="utf-8")
        record.output_path = str(path)
    return asdict(record)
