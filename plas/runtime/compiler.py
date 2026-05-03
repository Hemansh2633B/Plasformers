"""Compiler and kernel optimization utilities."""

from __future__ import annotations

from typing import Any


def torch_compile(model: Any, mode: str = "max-autotune", fullgraph: bool = False) -> Any:
    import torch

    return torch.compile(model, mode=mode, fullgraph=fullgraph)


def capture_cuda_graph(model: Any, example_inputs: tuple[Any, ...]) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA graphs require an NVIDIA CUDA device")
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        static_outputs = model(*example_inputs)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        static_outputs = model(*example_inputs)
    return {"graph": graph, "outputs": static_outputs, "inputs": example_inputs}


def flash_attention_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


def triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def fused_optimizer(name: str, params: Any, **kwargs: Any) -> Any:
    if name.lower() == "adamw":
        try:
            from apex.optimizers import FusedAdam
        except ImportError:
            import torch

            return torch.optim.AdamW(params, **kwargs)
        return FusedAdam(params, adam_w_mode=True, **kwargs)
    raise ValueError(f"Unsupported fused optimizer: {name}")
