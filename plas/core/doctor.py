"""Environment diagnostics for the plas CLI."""

from __future__ import annotations

import importlib.util
import platform
import shutil
from typing import Dict, List


OPTIONAL_PACKAGES = [
    "torch",
    "yaml",
    "rich",
    "typer",
    "gradio",
    "fastapi",
    "uvicorn",
    "onnx",
    "onnxruntime",
    "optuna",
    "mlflow",
    "ray",
    "cv2",
    "PIL",
    "transformers",
    "diffusers",
    "deepspeed",
    "torch_xla",
    "triton",
    "flash_attn",
]


def package_available(name: str) -> bool:
    """Return whether a Python package can be imported."""

    return importlib.util.find_spec(name) is not None


def run_doctor() -> Dict[str, object]:
    """Collect framework diagnostics."""

    packages = {name: package_available(name) for name in OPTIONAL_PACKAGES}
    cuda = False
    torch_version = None
    if packages["torch"]:
        import torch

        cuda = torch.cuda.is_available()
        torch_version = torch.__version__
    tools = {
        "git": shutil.which("git") is not None,
        "docker": shutil.which("docker") is not None,
        "kubectl": shutil.which("kubectl") is not None,
        "trtexec": shutil.which("trtexec") is not None,
        "mo": shutil.which("mo") is not None,
        "edgetpu_compiler": shutil.which("edgetpu_compiler") is not None,
        "rknn-toolkit2": package_available("rknn"),
    }
    issues: List[str] = []
    if not packages["torch"]:
        issues.append("PyTorch is not installed; model training and inference are unavailable.")
    if not packages["rich"]:
        issues.append("Rich is not installed; CLI will use plain text fallback.")
    if not packages["gradio"]:
        issues.append("Gradio is not installed; web UI requires the optional 'ui' extra.")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch_version,
        "cuda": cuda,
        "packages": packages,
        "tools": tools,
        "issues": issues,
    }
