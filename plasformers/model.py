"""Plasformers detector implementation."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn

from .config import PlasformersConfig, StageSpec, build_config
from .modules import (
    AdaptiveHybridStem,
    AntiAliasConv,
    ContextDynamicDepthwise,
    ConvNormAct,
    CrossScaleSparseExchange,
    DeformableResampler,
    DepthwiseSeparableConv,
    SqueezeExcite,
    TriPathBlock,
    WeightedAdd,
)


class TriPathBackbone(nn.Module):
    """Five-stage hierarchical tri-path backbone."""

    def __init__(self, stem_channels: int, stages: List[StageSpec], use_fft: bool = True) -> None:
        super().__init__()
        self.stages = stages
        modules: List[nn.Module] = []
        in_channels = stem_channels
        for idx, spec in enumerate(stages):
            layers: List[nn.Module] = []
            if idx > 0 or in_channels != spec.channels:
                stride = 2 if idx > 0 else 1
                layers.append(AntiAliasConv(in_channels, spec.channels, kernel_size=3, stride=stride))
            for _ in range(spec.blocks):
                layers.append(
                    TriPathBlock(
                        spec.channels,
                        heads=spec.heads,
                        token_grid=spec.token_grid,
                        local_kernel=spec.local_kernel,
                        expansion=spec.expansion,
                        use_fft=use_fft,
                    )
                )
            modules.append(nn.Sequential(*layers))
            in_channels = spec.channels
        self.stage_modules = nn.ModuleList(modules)

    def set_export_mode(self, mode: bool = True) -> None:
        for module in self.modules():
            if hasattr(module, "export_mode"):
                setattr(module, "export_mode", mode)

    def forward(self, x: Tensor) -> List[Tensor]:
        outputs: List[Tensor] = []
        for stage in self.stage_modules:
            x = stage(x)
            outputs.append(x)
        return outputs


class CrossScaleGlobalFusionNeck(nn.Module):
    """Bidirectional pyramid neck with cross-scale token exchange."""

    def __init__(
        self,
        stage_channels: List[int],
        pyramid_channels: int,
        use_deformable_resampler: bool = True,
    ) -> None:
        super().__init__()
        self.pyramid_channels = pyramid_channels
        self.c1_to_p3 = nn.Sequential(
            AntiAliasConv(stage_channels[0], pyramid_channels, kernel_size=3, stride=2),
            SqueezeExcite(pyramid_channels),
        )
        self.lateral = nn.ModuleList(
            [ConvNormAct(ch, pyramid_channels, kernel_size=1) for ch in stage_channels[1:]]
        )
        self.p7_down = AntiAliasConv(pyramid_channels, pyramid_channels, kernel_size=3, stride=2)
        self.resamplers = nn.ModuleList(
            [
                DeformableResampler(pyramid_channels, use_offsets=use_deformable_resampler)
                for _ in range(8)
            ]
        )
        self.top_down = nn.ModuleList([WeightedAdd(2) for _ in range(4)])
        self.bottom_up = nn.ModuleList([WeightedAdd(2) for _ in range(4)])
        self.exchange = CrossScaleSparseExchange(pyramid_channels, levels=5)
        self.recalibrate = nn.ModuleList([SqueezeExcite(pyramid_channels) for _ in range(5)])
        self.refine = nn.ModuleList(
            [DepthwiseSeparableConv(pyramid_channels, pyramid_channels, kernel_size=3) for _ in range(5)]
        )

    def set_export_mode(self, mode: bool = True) -> None:
        for module in self.modules():
            if hasattr(module, "export_mode"):
                setattr(module, "export_mode", mode)

    def forward(self, stages: List[Tensor]) -> List[Tensor]:
        if len(stages) != 5:
            raise ValueError(f"Expected five backbone features, got {len(stages)}")
        p3 = self.lateral[0](stages[1]) + self.c1_to_p3(stages[0])
        p4 = self.lateral[1](stages[2])
        p5 = self.lateral[2](stages[3])
        p6 = self.lateral[3](stages[4])
        p7 = self.p7_down(p6)
        feats: List[Tensor] = [p3, p4, p5, p6, p7]

        resampler_idx = 0
        for idx in range(3, -1, -1):
            up = self.resamplers[resampler_idx](feats[idx + 1], feats[idx])
            resampler_idx += 1
            feats[idx] = self.top_down[idx]([feats[idx], up])

        for idx in range(1, 5):
            down = self.resamplers[resampler_idx](feats[idx - 1], feats[idx])
            resampler_idx += 1
            feats[idx] = self.bottom_up[idx - 1]([feats[idx], down])

        feats = self.exchange(feats)
        return [self.refine[idx](self.recalibrate[idx](feat)) for idx, feat in enumerate(feats)]


class SceneContext(nn.Module):
    """Scene descriptor for query-guided dynamic prediction."""

    def __init__(self, channels: int, levels: int, context_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels * levels, context_dim),
            nn.SiLU(inplace=True),
            nn.Linear(context_dim, context_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, features: List[Tensor]) -> Tensor:
        pooled = [feat.mean(dim=(2, 3)) for feat in features]
        return self.net(torch.cat(pooled, dim=1))


class QueryGuidedPredictionBlock(nn.Module):
    """Per-level dynamic, decoupled prediction branch."""

    def __init__(
        self,
        channels: int,
        context_dim: int,
        num_classes: int,
        reg_max: int,
        depth: int,
        mask_dim: int = 0,
        keypoint_dim: int = 0,
    ) -> None:
        super().__init__()
        self.dynamic = ContextDynamicDepthwise(channels, context_dim)
        cls_layers: List[nn.Module] = []
        reg_layers: List[nn.Module] = []
        for _ in range(depth):
            cls_layers.append(DepthwiseSeparableConv(channels, channels, kernel_size=3))
            reg_layers.append(DepthwiseSeparableConv(channels, channels, kernel_size=3))
        self.cls_tower = nn.Sequential(*cls_layers)
        self.reg_tower = nn.Sequential(*reg_layers)
        self.cls_logits = nn.Conv2d(channels, num_classes, 1)
        self.bbox_dist = nn.Conv2d(channels, 4 * (reg_max + 1), 1)
        self.objectness = nn.Conv2d(channels, 1, 1)
        self.iou = nn.Conv2d(channels, 1, 1)
        self.mask_dim = mask_dim
        self.keypoint_dim = keypoint_dim
        self.mask_coeff = nn.Conv2d(channels, mask_dim, 1) if mask_dim > 0 else None
        self.keypoints = nn.Conv2d(channels, keypoint_dim, 1) if keypoint_dim > 0 else None
        self._init_prediction_biases(num_classes)

    def _init_prediction_biases(self, num_classes: int) -> None:
        prior_prob = 0.01
        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)
        nn.init.constant_(self.objectness.bias, bias_value)
        nn.init.constant_(self.iou.bias, 0.0)
        nn.init.constant_(self.bbox_dist.bias, 1.0)
        nn.init.normal_(self.cls_logits.weight, std=0.01)
        nn.init.normal_(self.bbox_dist.weight, std=0.01)

    def forward(self, x: Tensor, context: Tensor) -> Dict[str, Tensor]:
        x = x + self.dynamic(x, context)
        cls_feat = self.cls_tower(x)
        reg_feat = self.reg_tower(x)
        out: Dict[str, Tensor] = {
            "cls_logits": self.cls_logits(cls_feat),
            "bbox_dist": self.bbox_dist(reg_feat),
            "objectness": self.objectness(cls_feat),
            "iou": self.iou(reg_feat),
        }
        if self.mask_coeff is not None:
            out["mask_coeff"] = self.mask_coeff(reg_feat)
        if self.keypoints is not None:
            out["keypoints"] = self.keypoints(reg_feat)
        return out


class DynamicQueryGuidedDetectionHead(nn.Module):
    """Anchor-free multi-task detection head with auxiliary training branch."""

    def __init__(
        self,
        channels: int,
        num_classes: int,
        reg_max: int,
        depth: int,
        levels: int = 5,
        mask_dim: int = 0,
        keypoint_dim: int = 0,
    ) -> None:
        super().__init__()
        context_dim = max(128, channels)
        self.context = SceneContext(channels, levels, context_dim)
        self.blocks = nn.ModuleList(
            [
                QueryGuidedPredictionBlock(
                    channels,
                    context_dim,
                    num_classes,
                    reg_max,
                    depth,
                    mask_dim,
                    keypoint_dim,
                )
                for _ in range(levels)
            ]
        )
        self.aux_blocks = nn.ModuleList(
            [
                QueryGuidedPredictionBlock(
                    channels,
                    context_dim,
                    num_classes,
                    reg_max,
                    max(1, depth - 1),
                    mask_dim=0,
                    keypoint_dim=0,
                )
                for _ in range(levels)
            ]
        )

    def forward(self, features: List[Tensor]) -> Dict[str, List[Dict[str, Tensor]]]:
        context = self.context(features)
        outputs = [block(feat, context) for block, feat in zip(self.blocks, features)]
        result: Dict[str, List[Dict[str, Tensor]]] = {"outputs": outputs}
        if self.training:
            result["aux_outputs"] = [
                block(feat.detach(), context.detach())
                for block, feat in zip(self.aux_blocks, features)
            ]
        return result

    def export_forward(self, features: List[Tensor]) -> Tuple[Tensor, ...]:
        context = self.context(features)
        flat: List[Tensor] = []
        for block, feat in zip(self.blocks, features):
            pred = block(feat, context)
            flat.extend([pred["cls_logits"], pred["bbox_dist"], pred["objectness"], pred["iou"]])
        return tuple(flat)


class PlasformersDetector(nn.Module):
    """End-to-end Plasformers detector."""

    def __init__(self, config: PlasformersConfig) -> None:
        super().__init__()
        self.config = config
        self.stem = AdaptiveHybridStem(config.in_channels, config.stem_channels)
        self.backbone = TriPathBackbone(config.stem_channels, config.stages, use_fft=config.use_fft)
        self.neck = CrossScaleGlobalFusionNeck(
            [stage.channels for stage in config.stages],
            config.pyramid_channels,
            use_deformable_resampler=config.use_deformable_resampler,
        )
        self.head = DynamicQueryGuidedDetectionHead(
            config.pyramid_channels,
            config.num_classes,
            config.reg_max,
            config.head_depth,
            levels=5,
            mask_dim=config.mask_dim,
            keypoint_dim=config.keypoint_dim,
        )
        self.set_export_mode(config.export_mode)
        self.apply(self._init_weights)
        for block in list(self.head.blocks) + list(self.head.aux_blocks):
            block._init_prediction_biases(config.num_classes)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def set_export_mode(self, mode: bool = True) -> None:
        self.config.export_mode = mode
        for module in self.modules():
            if module is not self and hasattr(module, "export_mode"):
                setattr(module, "export_mode", mode)

    def forward_features(self, x: Tensor) -> List[Tensor]:
        stem = self.stem(x)
        stages = self.backbone(stem)
        return self.neck(stages)

    def forward(self, x: Tensor) -> Dict[str, List[Dict[str, Tensor]]]:
        return self.head(self.forward_features(x))

    def export_forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        return self.head.export_forward(self.forward_features(x))


def build_plasformers(
    variant: str = "base",
    *,
    num_classes: int = 80,
    mask_dim: int = 0,
    keypoint_dim: int = 0,
    use_fft: bool = True,
    export_mode: bool = False,
) -> PlasformersDetector:
    """Factory for Plasformers model variants."""

    config = build_config(
        variant,
        num_classes=num_classes,
        mask_dim=mask_dim,
        keypoint_dim=keypoint_dim,
        use_fft=use_fft,
        export_mode=export_mode,
    )
    return PlasformersDetector(config)
