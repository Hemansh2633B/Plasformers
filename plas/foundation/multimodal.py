"""Vision-language and open-vocabulary modeling primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class VisionLanguageConfig:
    """Configuration for CLIP-style dual encoders and grounding heads."""

    vision_dim: int = 768
    text_dim: int = 768
    projection_dim: int = 512
    vocab_size: int = 32000
    max_text_len: int = 77
    temperature: float = 0.07


@dataclass
class ContrastiveBatch:
    """Image-text contrastive batch."""

    images: Any
    input_ids: Any
    attention_mask: Any | None = None


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Multimodal modules require PyTorch") from exc
    return torch, nn


class ClipStyleDualEncoder:
    """CLIP-style image-text retrieval and contrastive pretraining module."""

    def __init__(self, vision_encoder: Any, config: VisionLanguageConfig | None = None) -> None:
        torch, nn = _require_torch()
        cfg = config or VisionLanguageConfig()

        class _DualEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.vision_encoder = vision_encoder
                self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.text_dim)
                self.text_encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(cfg.text_dim, nhead=8, batch_first=True),
                    num_layers=4,
                )
                self.image_proj = nn.Linear(cfg.vision_dim, cfg.projection_dim)
                self.text_proj = nn.Linear(cfg.text_dim, cfg.projection_dim)
                self.logit_scale = nn.Parameter(torch.ones(()) * (1.0 / cfg.temperature))

            def encode_image(self, images: Any) -> Any:
                features = self.vision_encoder(images)
                if isinstance(features, (list, tuple)):
                    features = features[-1]
                if features.ndim == 4:
                    features = features.mean(dim=(2, 3))
                return torch.nn.functional.normalize(self.image_proj(features), dim=-1)

            def encode_text(self, input_ids: Any, attention_mask: Any | None = None) -> Any:
                tokens = self.token_embedding(input_ids)
                encoded = self.text_encoder(tokens)
                if attention_mask is not None:
                    denom = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
                    pooled = (encoded * attention_mask.unsqueeze(-1)).sum(dim=1) / denom
                else:
                    pooled = encoded.mean(dim=1)
                return torch.nn.functional.normalize(self.text_proj(pooled), dim=-1)

            def forward(self, batch: ContrastiveBatch) -> Mapping[str, Any]:
                image = self.encode_image(batch.images)
                text = self.encode_text(batch.input_ids, batch.attention_mask)
                logits = self.logit_scale.exp() * image @ text.t()
                return {"image_embeds": image, "text_embeds": text, "logits_per_image": logits, "logits_per_text": logits.t()}

        self.module = _DualEncoder()

    def as_module(self) -> Any:
        return self.module


def contrastive_loss(logits_per_image: Any, logits_per_text: Any) -> Any:
    """Symmetric image-text contrastive loss."""

    torch, _ = _require_torch()
    import torch.nn.functional as F

    targets = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
    return 0.5 * (F.cross_entropy(logits_per_image, targets) + F.cross_entropy(logits_per_text, targets))


class GroundingHead:
    """Open-vocabulary grounding head that scores regions against text embeddings."""

    def __init__(self, region_dim: int = 256, text_dim: int = 512) -> None:
        torch, nn = _require_torch()

        class _GroundingHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.region_proj = nn.Linear(region_dim, text_dim)
                self.box_refine = nn.Linear(region_dim, 4)

            def forward(self, region_features: Any, text_embeds: Any) -> Mapping[str, Any]:
                regions = torch.nn.functional.normalize(self.region_proj(region_features), dim=-1)
                text = torch.nn.functional.normalize(text_embeds, dim=-1)
                return {"scores": regions @ text.transpose(-1, -2), "boxes_delta": self.box_refine(region_features)}

        self.module = _GroundingHead()

    def as_module(self) -> Any:
        return self.module


def promptable_detection_queries(prompts: Iterable[str], tokenizer: Any) -> dict[str, Any]:
    """Tokenize text prompts for promptable/open-vocabulary detection."""

    return tokenizer(list(prompts), padding=True, truncation=True, return_tensors="pt")
