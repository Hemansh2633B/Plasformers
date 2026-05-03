"""Benchmark laboratory for Plasformers."""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List

from plasformers import build_plasformers, list_variants


@dataclass
class BenchmarkRecord:
    """Measured benchmark result."""

    variant: str
    device: str
    batch: int
    height: int
    width: int
    fps: float
    latency_ms: float
    throughput: float
    params_m: float
    memory_mb: float | None = None
    flops_g_estimate: float | None = None
    tensor_core_utilization: float | None = None
    energy_j: float | None = None


class BenchmarkRunner:
    """Run eager/export-mode latency and memory benchmarks."""

    def __init__(self, device: str = "auto", amp: bool = True, export_mode: bool = True) -> None:
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.amp = amp
        self.export_mode = export_mode

    @staticmethod
    def params_m(model: object) -> float:
        return sum(p.numel() for p in model.parameters()) / 1e6

    def run(
        self,
        variant: str,
        batch: int = 1,
        height: int = 640,
        width: int = 640,
        warmup: int = 10,
        iters: int = 50,
    ) -> BenchmarkRecord:
        import torch

        model = build_plasformers(variant, export_mode=self.export_mode).to(self.device).eval()
        x = torch.randn(batch, 3, height, width, device=self.device)
        use_amp = self.amp and self.device.startswith("cuda")
        if use_amp:
            model.half()
            x = x.half()
        device_type = self.device.split(":")[0]
        for _ in range(warmup):
            with torch.inference_mode(), torch.autocast(device_type, enabled=use_amp):
                _ = model.export_forward(x) if self.export_mode else model(x)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(iters):
            with torch.inference_mode(), torch.autocast(device_type, enabled=use_amp):
                _ = model.export_forward(x) if self.export_mode else model(x)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        images = batch * iters
        fps = images / max(elapsed, 1e-9)
        memory = None
        if self.device.startswith("cuda"):
            memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return BenchmarkRecord(
            variant=variant,
            device=self.device,
            batch=batch,
            height=height,
            width=width,
            fps=fps,
            latency_ms=1000.0 / max(fps / batch, 1e-9),
            throughput=fps,
            params_m=self.params_m(model),
            memory_mb=memory,
        )

    def run_all(self, variants: Iterable[str] | None = None, **kwargs: object) -> List[Dict[str, object]]:
        selected = variants or list_variants()
        return [asdict(self.run(variant, **kwargs)) for variant in selected]
