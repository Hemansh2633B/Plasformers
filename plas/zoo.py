"""Model zoo registry with checkpoint integrity support."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

from .core.security import verify_sha256


@dataclass(frozen=True)
class ModelCard:
    """Metadata for a released Plasformers checkpoint."""

    name: str
    variant: str
    version: str
    params_m: float
    flops_g: float
    target_coco_ap: float
    target_t4_fps: float
    input_size: tuple[int, int]
    sha256: str | None = None
    url: str | None = None
    license: str = "Apache-2.0"
    recommendation: str = "Use export mode for deployment."
    status: str = "target"


class ModelZoo:
    """Registry and downloader for model cards."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        root = cache_dir or os.environ.get("PLAS_CACHE", "~/.cache/plasformers")
        self.cache_dir = Path(root).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cards: Dict[str, ModelCard] = {}
        for card in _DEFAULT_CARDS:
            self.register(card)

    def register(self, card: ModelCard) -> None:
        self._cards[card.name] = card

    def get(self, name: str) -> ModelCard:
        try:
            return self._cards[name]
        except KeyError as exc:
            valid = ", ".join(sorted(self._cards))
            raise KeyError(f"Unknown model '{name}'. Available: {valid}") from exc

    def list(self) -> Iterable[ModelCard]:
        return tuple(self._cards.values())

    def as_rows(self) -> list[Mapping[str, object]]:
        return [
            {
                "name": card.name,
                "version": card.version,
                "params(M)": card.params_m,
                "FLOPs(G)": card.flops_g,
                "AP target": card.target_coco_ap,
                "T4 FPS target": card.target_t4_fps,
                "status": card.status,
            }
            for card in self.list()
        ]

    def export_metadata(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps([asdict(card) for card in self.list()], indent=2),
            encoding="utf-8",
        )
        return out

    def checkpoint_path(self, name: str) -> Path:
        card = self.get(name)
        return self.cache_dir / f"{card.name}-{card.version}.pt"

    def download(self, name: str, force: bool = False) -> Path:
        """Download and verify a checkpoint.

        Model cards in this source tree intentionally mark unreleased weights
        with `url=None`. Private enterprises can register internal cards with
        signed URLs and SHA256 values without changing the downloader.
        """

        card = self.get(name)
        if not card.url:
            raise RuntimeError(
                f"No public checkpoint URL is registered for {name}. "
                "Register a card with url and sha256 before downloading."
            )
        out = self.checkpoint_path(name)
        if out.exists() and not force:
            if verify_sha256(out, card.sha256):
                return out
            raise RuntimeError(f"Cached checkpoint failed SHA256 verification: {out}")
        tmp = out.with_suffix(out.suffix + ".tmp")
        urllib.request.urlretrieve(card.url, tmp)
        if not verify_sha256(tmp, card.sha256):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded checkpoint failed SHA256 verification: {name}")
        tmp.replace(out)
        return out


_DEFAULT_CARDS = [
    ModelCard("plasformers-nano", "nano", "0.1.0", 3.9, 8.7, 44.2, 520, (640, 640)),
    ModelCard("plasformers-small", "small", "0.1.0", 8.8, 19.6, 49.7, 390, (640, 640)),
    ModelCard("plasformers-base", "base", "0.1.0", 22.4, 56.0, 56.1, 255, (640, 640)),
    ModelCard("plasformers-large", "large", "0.1.0", 41.8, 108.0, 57.2, 172, (640, 640)),
    ModelCard("plasformers-xlarge", "xlarge", "0.1.0", 73.5, 196.0, 58.4, 112, (640, 640)),
]


def get_model_zoo(cache_dir: str | Path | None = None) -> ModelZoo:
    """Return a model zoo instance."""

    return ModelZoo(cache_dir)
