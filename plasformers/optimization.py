"""Inference and compression helpers for Plasformers."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor, nn

from .modules import ConvNormAct


def fuse_conv_bn_eval(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuse Conv2d and BatchNorm2d weights for inference."""

    if conv.training or bn.training:
        raise ValueError("Conv-BN fusion requires eval mode")
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)

    conv_weight = conv.weight.clone().view(conv.out_channels, -1)
    conv_bias = torch.zeros(conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype)
    if conv.bias is not None:
        conv_bias = conv.bias

    inv_std = torch.rsqrt(bn.running_var + bn.eps)
    scale = bn.weight * inv_std
    fused.weight.data.copy_((conv_weight * scale.reshape(-1, 1)).view_as(conv.weight))
    fused.bias.data.copy_(bn.bias + (conv_bias - bn.running_mean) * scale)
    return fused


def fuse_model_for_inference(module: nn.Module) -> nn.Module:
    """Recursively fuse ConvNormAct blocks in-place."""

    module.eval()
    for name, child in module.named_children():
        if isinstance(child, ConvNormAct):
            child.conv = fuse_conv_bn_eval(child.conv, child.norm)
            child.norm = nn.Identity()
        else:
            fuse_model_for_inference(child)
        setattr(module, name, child)
    return module


def collect_prunable_convs(model: nn.Module) -> List[Tuple[str, nn.Conv2d]]:
    """Return convolution layers suitable for structured channel pruning."""

    prunable: List[Tuple[str, nn.Conv2d]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.groups == 1 and module.out_channels >= 16:
            prunable.append((name, module))
    return prunable


def activation_sparsity(tensor: Tensor, threshold: float = 1e-5) -> Tensor:
    """Estimate sparse activation ratio for profiling."""

    return (tensor.abs() <= threshold).float().mean()


def prepare_qat(model: nn.Module, backend: str = "fbgemm") -> nn.Module:
    """Prepare a model for quantization-aware training with torch.ao."""

    try:
        import torch.ao.quantization as quantization
    except ImportError as exc:
        raise RuntimeError("torch.ao.quantization is unavailable in this PyTorch build") from exc
    model.train()
    model.qconfig = quantization.get_default_qat_qconfig(backend)
    return quantization.prepare_qat(model, inplace=True)
