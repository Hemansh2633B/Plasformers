"""Reusable neural network modules for Plasformers.

The modules are intentionally conservative in their operator choices. FFT and
continuous-offset resampling are available for training and PyTorch inference,
while export mode switches those paths to TensorRT/CoreML-friendly surrogates.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _activation(name: str | None = "silu") -> nn.Module:
    if name is None or name == "identity":
        return nn.Identity()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ConvNormAct(nn.Module):
    """Convolution followed by BatchNorm and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        activation: str | None = "silu",
        bias: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.act = _activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class BlurPool(nn.Module):
    """Fixed low-pass filter used before spatial decimation."""

    def __init__(self, channels: int, stride: int = 2) -> None:
        super().__init__()
        kernel_1d = torch.tensor([1.0, 2.0, 1.0])
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum()
        self.register_buffer("kernel", kernel_2d.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.channels = channels
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            return x
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(x, self.kernel.to(dtype=x.dtype), stride=self.stride, groups=self.channels)


class AntiAliasConv(nn.Module):
    """Anti-aliased strided convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2) -> None:
        super().__init__()
        self.blur = BlurPool(in_channels, stride=stride) if stride > 1 else nn.Identity()
        self.conv = ConvNormAct(in_channels, out_channels, kernel_size=kernel_size, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.blur(x))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise spatial convolution followed by pointwise projection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.depthwise = ConvNormAct(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
        )
        self.pointwise = ConvNormAct(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class SqueezeExcite(nn.Module):
    """Lightweight channel recalibration."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.net(x)


class LearnableFrequencyPreprocess(nn.Module):
    """Frequency-aware input conditioner for the adaptive hybrid stem."""

    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.mix = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels * 2), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(4, channels * 2), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.mix(self.local(x))
        return x + residual * self.gate(x)


class AdaptiveHybridStem(nn.Module):
    """Anti-aliased, detail-preserving input stem.

    For an input B x 3 x H x W, the output has shape
    B x C x ceil(H / 4) x ceil(W / 4) for common even image sizes.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        mid = max(16, out_channels // 2)
        self.freq_pre = LearnableFrequencyPreprocess(in_channels)
        self.down1 = AntiAliasConv(in_channels, mid, kernel_size=3, stride=2)
        self.refine = DepthwiseSeparableConv(mid, mid, kernel_size=3)
        self.down2 = AntiAliasConv(mid, out_channels, kernel_size=3, stride=2)
        self.detail = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4, ceil_mode=True),
            ConvNormAct(in_channels, out_channels, kernel_size=1),
        )
        self.fuse = ConvNormAct(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        conditioned = self.freq_pre(x)
        coarse = self.down2(self.refine(self.down1(conditioned)))
        detail = self.detail(x)
        if detail.shape[-2:] != coarse.shape[-2:]:
            detail = F.interpolate(detail, size=coarse.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([coarse, detail], dim=1))


class DynamicDepthwiseConv(nn.Module):
    """Input-adaptive mixture of depthwise kernels."""

    def __init__(
        self,
        channels: int,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        reduction: int = 4,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=k,
                    padding=k // 2,
                    groups=channels,
                    bias=False,
                )
                for k in kernel_sizes
            ]
        )
        hidden = max(8, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, len(kernel_sizes), 1),
        )
        self.norm = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        weights = torch.softmax(self.gate(x).flatten(1), dim=1)
        y = torch.zeros_like(x)
        for idx, branch in enumerate(self.branches):
            y = y + branch(x) * weights[:, idx].view(-1, 1, 1, 1)
        return self.act(self.norm(y))


class LocalPathBlock(nn.Module):
    """CSP-inspired partial residual local mixer."""

    def __init__(self, channels: int, kernel_size: int, expansion: float) -> None:
        super().__init__()
        routed = channels - channels // 2
        hidden = int(routed * expansion)
        self.part_channels = channels // 2
        self.route = nn.Sequential(
            ConvNormAct(routed, hidden, kernel_size=1),
            DynamicDepthwiseConv(hidden, kernel_sizes=(3, kernel_size, max(kernel_size + 2, 3))),
            SqueezeExcite(hidden),
            ConvNormAct(hidden, routed, kernel_size=1, activation=None),
        )
        self.proj = ConvNormAct(channels, channels, kernel_size=1, activation=None)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: Tensor) -> Tensor:
        left, right = torch.split(x, [self.part_channels, x.shape[1] - self.part_channels], dim=1)
        y = torch.cat([left, self.route(right)], dim=1)
        return x + self.proj(y) * self.scale


class CompressedLinearAttention(nn.Module):
    """Window-compressed global path with relative positional depthwise bias."""

    def __init__(self, channels: int, heads: int, token_grid: int) -> None:
        super().__init__()
        while heads > 1 and channels % heads != 0:
            heads -= 1
        self.channels = channels
        self.heads = max(1, heads)
        self.head_dim = channels // self.heads
        self.token_grid = token_grid
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.kv = nn.Conv2d(channels, channels * 2, 1, bias=False)
        self.window_bias = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.out = ConvNormAct(channels, channels, kernel_size=1, activation=None)
        self.scale = self.head_dim**-0.5

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        pooled = F.adaptive_avg_pool2d(x, (self.token_grid, self.token_grid))
        q = self.q(x).reshape(b, self.heads, self.head_dim, h * w).transpose(-1, -2)
        kv = self.kv(pooled).reshape(b, 2, self.heads, self.head_dim, self.token_grid * self.token_grid)
        k = kv[:, 0]
        v = kv[:, 1].transpose(-1, -2)
        attn = torch.softmax(torch.matmul(q, k) * self.scale, dim=-1)
        y = torch.matmul(attn, v).transpose(-1, -2).reshape(b, c, h, w)
        y = self.out(y + self.window_bias(x))
        return x + y


class SpectralGatingBlock(nn.Module):
    """FFT spectral path with export-friendly depthwise surrogate."""

    def __init__(self, channels: int, num_bands: int = 4, use_fft: bool = True) -> None:
        super().__init__()
        self.use_fft = use_fft
        self.export_mode = False
        self.num_bands = num_bands
        self.band_logits = nn.Parameter(torch.zeros(channels, num_bands))
        self.band_gain = nn.Parameter(torch.zeros(channels, num_bands))
        self.channel_gate = SqueezeExcite(channels)
        self.proxy = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03),
            nn.SiLU(inplace=True),
        )

    def _spectral_mask(self, channels: int, height: int, width: int, device: torch.device) -> Tensor:
        fy = torch.fft.fftfreq(height, device=device).abs().view(height, 1)
        fx = torch.fft.rfftfreq(width, device=device).abs().view(1, width // 2 + 1)
        radius = torch.sqrt(fy * fy + fx * fx)
        radius = radius / radius.max().clamp_min(1e-6)
        centers = torch.linspace(0.0, 1.0, self.num_bands, device=device).view(self.num_bands, 1, 1)
        basis = torch.exp(-((radius.unsqueeze(0) - centers) ** 2) / 0.045)
        basis = basis / basis.sum(dim=0, keepdim=True).clamp_min(1e-6)
        weights = torch.softmax(self.band_logits[:channels], dim=-1)
        gains = torch.tanh(self.band_gain[:channels])
        coeff = weights * (1.0 + gains)
        return torch.einsum("cb,bhw->chw", coeff, basis)

    def forward(self, x: Tensor) -> Tensor:
        if self.export_mode or not self.use_fft:
            return x + self.channel_gate(self.proxy(x))
        dtype = x.dtype
        b, c, h, w = x.shape
        x_float = x.float()
        spectrum = torch.fft.rfft2(x_float, norm="ortho")
        mask = self._spectral_mask(c, h, w, x.device).unsqueeze(0)
        filtered = torch.fft.irfft2(spectrum * (1.0 + mask), s=(h, w), norm="ortho")
        filtered = filtered.to(dtype=dtype)
        return x + self.channel_gate(filtered)


class AdaptiveFusionGate(nn.Module):
    """Input-adaptive tri-path fusion.

    F_out = alpha F_local + beta F_global + gamma F_freq, with
    alpha + beta + gamma = 1 for every sample.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(8, channels // 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 3, 1),
        )

    def forward(self, source: Tensor, local: Tensor, global_path: Tensor, freq: Tensor) -> Tensor:
        weights = torch.softmax(self.gate(source).flatten(1), dim=1)
        return (
            local * weights[:, 0].view(-1, 1, 1, 1)
            + global_path * weights[:, 1].view(-1, 1, 1, 1)
            + freq * weights[:, 2].view(-1, 1, 1, 1)
        )


class TriPathBlock(nn.Module):
    """Local, global, and frequency streams fused at each depth."""

    def __init__(
        self,
        channels: int,
        heads: int,
        token_grid: int,
        local_kernel: int,
        expansion: float,
        use_fft: bool = True,
    ) -> None:
        super().__init__()
        self.local = LocalPathBlock(channels, local_kernel, expansion)
        self.global_path = CompressedLinearAttention(channels, heads, token_grid)
        self.freq = SpectralGatingBlock(channels, use_fft=use_fft)
        self.fusion = AdaptiveFusionGate(channels)
        self.post = ConvNormAct(channels, channels, kernel_size=1, activation=None)
        self.scale = nn.Parameter(torch.ones(1) * 0.2)

    def set_export_mode(self, mode: bool = True) -> None:
        self.freq.export_mode = mode

    def forward(self, x: Tensor) -> Tensor:
        fused = self.fusion(x, self.local(x), self.global_path(x), self.freq(x))
        return x + self.post(fused) * self.scale


class WeightedAdd(nn.Module):
    """Positive normalized weighted sum."""

    def __init__(self, inputs: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(inputs))
        self.eps = 1e-4

    def forward(self, tensors: Sequence[Tensor]) -> Tensor:
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        out = tensors[0] * weights[0]
        for idx in range(1, len(tensors)):
            out = out + tensors[idx] * weights[idx]
        return out


class DeformableResampler(nn.Module):
    """Continuous-offset cross-scale resampler with export fallback."""

    def __init__(self, channels: int, use_offsets: bool = True, offset_scale: float = 0.08) -> None:
        super().__init__()
        self.use_offsets = use_offsets
        self.export_mode = False
        self.offset_scale = offset_scale
        self.offset = nn.Conv2d(channels, 2, 3, padding=1)

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        size = target.shape[-2:]
        y = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if self.export_mode or not self.use_offsets:
            return y
        b, _, h, w = y.shape
        offset = torch.tanh(self.offset(y)).permute(0, 2, 3, 1) * self.offset_scale
        yy = torch.linspace(-1.0, 1.0, h, device=y.device, dtype=y.dtype)
        xx = torch.linspace(-1.0, 1.0, w, device=y.device, dtype=y.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
        return F.grid_sample(y, grid + offset, mode="bilinear", padding_mode="border", align_corners=False)


class CrossScaleSparseExchange(nn.Module):
    """Sparse token exchange between pyramid levels."""

    def __init__(self, channels: int, levels: int = 5) -> None:
        super().__init__()
        self.levels = levels
        self.route = nn.Linear(channels, levels)
        self.context = nn.Linear(channels, channels)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        descriptors = torch.stack([feat.mean(dim=(2, 3)) for feat in features], dim=1)
        logits = self.route(descriptors) / self.temperature.clamp_min(0.2)
        weights = torch.softmax(logits, dim=-1)
        mixed = torch.bmm(weights, descriptors)
        outputs: List[Tensor] = []
        for idx, feat in enumerate(features):
            gate = torch.sigmoid(self.context(mixed[:, idx])).view(feat.shape[0], feat.shape[1], 1, 1)
            outputs.append(feat * (1.0 + gate))
        return outputs


class ContextDynamicDepthwise(nn.Module):
    """Depthwise dynamic convolution controlled by scene-level context."""

    def __init__(self, channels: int, context_dim: int, kernel_sizes: Sequence[int] = (3, 5, 7)) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, k, padding=k // 2, groups=channels, bias=False)
                for k in kernel_sizes
            ]
        )
        self.gate = nn.Linear(context_dim, len(kernel_sizes))
        self.norm = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        weights = torch.softmax(self.gate(context), dim=-1)
        y = torch.zeros_like(x)
        for idx, branch in enumerate(self.branches):
            y = y + branch(x) * weights[:, idx].view(-1, 1, 1, 1)
        return self.act(self.norm(y))
