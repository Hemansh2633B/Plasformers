"""Configuration objects for the Plasformers detector family."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


def make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    """Round channel counts to hardware-friendly multiples."""

    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


@dataclass(frozen=True)
class StageSpec:
    """Backbone stage definition.

    Resolution is expressed by stride relative to the input image.
    """

    name: str
    stride: int
    channels: int
    blocks: int
    heads: int
    token_grid: int
    local_kernel: int
    expansion: float


@dataclass
class PlasformersConfig:
    """Complete model configuration."""

    variant: str = "base"
    num_classes: int = 80
    in_channels: int = 3
    stem_channels: int = 64
    pyramid_channels: int = 192
    stages: List[StageSpec] = field(default_factory=list)
    reg_max: int = 16
    head_depth: int = 2
    mask_dim: int = 0
    keypoint_dim: int = 0
    use_fft: bool = True
    use_deformable_resampler: bool = True
    export_mode: bool = False


_BASE_STAGE_CHANNELS: Tuple[int, ...] = (64, 128, 256, 512, 768)
_BASE_BLOCKS: Tuple[int, ...] = (2, 3, 5, 4, 2)
_BASE_HEADS: Tuple[int, ...] = (1, 2, 4, 8, 8)
_BASE_GRIDS: Tuple[int, ...] = (8, 8, 8, 6, 4)
_BASE_KERNELS: Tuple[int, ...] = (5, 5, 7, 9, 11)
_BASE_EXPANSIONS: Tuple[float, ...] = (1.75, 2.0, 2.25, 2.5, 2.5)

_VARIANTS: Dict[str, Dict[str, float | int]] = {
    "nano": {"width": 0.375, "depth": 0.34, "pyramid": 96, "head_depth": 1},
    "small": {"width": 0.50, "depth": 0.50, "pyramid": 128, "head_depth": 1},
    "base": {"width": 0.75, "depth": 0.75, "pyramid": 192, "head_depth": 2},
    "large": {"width": 1.00, "depth": 1.00, "pyramid": 256, "head_depth": 2},
    "xlarge": {"width": 1.25, "depth": 1.20, "pyramid": 320, "head_depth": 3},
}


def list_variants() -> Sequence[str]:
    """Return supported scaling family names."""

    return tuple(_VARIANTS.keys())


def _scaled_blocks(blocks: int, depth_mult: float) -> int:
    return max(1, int(round(blocks * depth_mult)))


def _heads_for_channels(channels: int, requested: int) -> int:
    heads = min(requested, channels)
    while heads > 1 and channels % heads != 0:
        heads -= 1
    return max(1, heads)


def build_config(
    variant: str = "base",
    *,
    num_classes: int = 80,
    mask_dim: int = 0,
    keypoint_dim: int = 0,
    use_fft: bool = True,
    export_mode: bool = False,
) -> PlasformersConfig:
    """Build a scaled Plasformers configuration."""

    key = variant.lower()
    if key not in _VARIANTS:
        valid = ", ".join(list_variants())
        raise ValueError(f"Unknown Plasformers variant '{variant}'. Valid variants: {valid}")

    spec = _VARIANTS[key]
    width = float(spec["width"])
    depth = float(spec["depth"])
    pyramid_channels = int(spec["pyramid"])
    stem_channels = make_divisible(_BASE_STAGE_CHANNELS[0] * width)

    stages: List[StageSpec] = []
    for idx, base_channels in enumerate(_BASE_STAGE_CHANNELS):
        channels = make_divisible(base_channels * width)
        heads = _heads_for_channels(channels, _BASE_HEADS[idx])
        stages.append(
            StageSpec(
                name=f"S{idx + 1}",
                stride=4 * (2**idx),
                channels=channels,
                blocks=_scaled_blocks(_BASE_BLOCKS[idx], depth),
                heads=heads,
                token_grid=_BASE_GRIDS[idx],
                local_kernel=_BASE_KERNELS[idx],
                expansion=_BASE_EXPANSIONS[idx],
            )
        )

    return PlasformersConfig(
        variant=key,
        num_classes=num_classes,
        stem_channels=stem_channels,
        pyramid_channels=pyramid_channels,
        stages=stages,
        head_depth=int(spec["head_depth"]),
        mask_dim=mask_dim,
        keypoint_dim=keypoint_dim,
        use_fft=use_fft,
        export_mode=export_mode,
    )
