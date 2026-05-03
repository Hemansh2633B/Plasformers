"""Extension marketplace for plugins, models, deployment recipes, and benchmark suites."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MarketplaceItem:
    """Marketplace item metadata."""

    name: str
    kind: str
    version: str
    description: str
    source: str
    sha256: str | None = None
    tags: list[str] = field(default_factory=list)


class MarketplaceIndex:
    """Filesystem-backed extension marketplace index."""

    def __init__(self, path: str | Path = "marketplace/index.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.items: list[MarketplaceItem] = []
        if self.path.exists():
            self.items = [MarketplaceItem(**item) for item in json.loads(self.path.read_text(encoding="utf-8"))]

    def add(self, item: MarketplaceItem) -> None:
        self.items = [existing for existing in self.items if not (existing.name == item.name and existing.version == item.version)]
        self.items.append(item)
        self.save()

    def search(self, query: str = "", kind: str | None = None) -> list[MarketplaceItem]:
        q = query.lower()
        return [
            item
            for item in self.items
            if (kind is None or item.kind == kind)
            and (not q or q in item.name.lower() or q in item.description.lower() or q in " ".join(item.tags).lower())
        ]

    def save(self) -> Path:
        self.path.write_text(json.dumps([asdict(item) for item in self.items], indent=2), encoding="utf-8")
        return self.path

    def extend(self, items: Iterable[MarketplaceItem]) -> None:
        for item in items:
            self.add(item)
