"""Cloud marketplace publishing templates."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass(frozen=True)
class MarketplaceListing:
    """Cloud marketplace listing metadata."""

    name: str
    provider: str
    image: str
    description: str
    support_url: str
    license: str = "Apache-2.0"


def write_listing(listing: MarketplaceListing, output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(listing), indent=2), encoding="utf-8")
    return out


def aws_marketplace_template(image: str) -> MarketplaceListing:
    return MarketplaceListing("Plasformers Vision Platform", "aws", image, "Full-stack computer vision platform", "https://example.com/support")


def gcp_marketplace_template(image: str) -> MarketplaceListing:
    return MarketplaceListing("Plasformers Vision Platform", "gcp", image, "Cloud-native vision platform for GKE and TPU", "https://example.com/support")


def azure_marketplace_template(image: str) -> MarketplaceListing:
    return MarketplaceListing("Plasformers Vision Platform", "azure", image, "Enterprise computer vision platform for AKS", "https://example.com/support")
