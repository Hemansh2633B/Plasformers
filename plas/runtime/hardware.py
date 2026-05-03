"""Hardware intelligence layer."""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""

    cpu: str
    accelerator: str = "cpu"
    cuda_available: bool = False
    rocm_available: bool = False
    intel_gpu: bool = False
    apple_silicon: bool = False
    tpu_available: bool = False
    edge_npu: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def detect_hardware() -> HardwareProfile:
    """Detect available accelerators and optimization recommendations."""

    cpu = platform.processor() or platform.machine()
    profile = HardwareProfile(cpu=cpu)
    try:
        import torch

        profile.cuda_available = torch.cuda.is_available()
        if profile.cuda_available:
            profile.accelerator = "nvidia-cuda"
            profile.recommendations.extend(["enable_amp_fp16", "try_tensorrt", "enable_cuda_graphs"])
        profile.rocm_available = bool(getattr(torch.version, "hip", None))
        if profile.rocm_available:
            profile.accelerator = "amd-rocm"
            profile.recommendations.append("enable_amp_fp16")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            profile.apple_silicon = True
            profile.accelerator = "apple-mps"
            profile.recommendations.append("export_coreml")
    except ImportError:
        pass
    try:
        import torch_xla.core.xla_model as xm

        _ = xm.xla_device()
        profile.tpu_available = True
        profile.accelerator = "tpu-xla"
        profile.recommendations.extend(["enable_bfloat16", "static_shapes", "sharded_dataloader"])
    except Exception:
        pass
    if shutil.which("edgetpu_compiler"):
        profile.edge_npu.append("coral-edge-tpu")
    if shutil.which("rknn-toolkit2") or shutil.which("rknn_server"):
        profile.edge_npu.append("rockchip-rknn")
    if _command_contains("lspci", "Intel"):
        profile.intel_gpu = True
        profile.recommendations.append("export_openvino")
    return profile


def _command_contains(command: str, needle: str) -> bool:
    if shutil.which(command) is None:
        return False
    try:
        result = subprocess.run([command], capture_output=True, text=True, check=False)
    except OSError:
        return False
    return needle.lower() in result.stdout.lower()


def optimal_runtime(profile: HardwareProfile | None = None) -> str:
    profile = profile or detect_hardware()
    if profile.cuda_available:
        return "tensorrt"
    if profile.tpu_available:
        return "xla"
    if profile.apple_silicon:
        return "coreml"
    if profile.intel_gpu:
        return "openvino"
    if profile.edge_npu:
        return profile.edge_npu[0]
    return "onnxruntime"
