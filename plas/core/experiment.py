"""Experiment management and reproducibility snapshots."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import dump_config


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set Python, NumPy, and PyTorch seeds."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        return


def git_revision() -> str | None:
    """Return the current git revision when available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


@dataclass
class ExperimentRun:
    """Minimal artifact tracker with checkpoint lineage."""

    name: str
    root: Path = Path("runs")
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = self.root / f"{self.name}-{timestamp}"
        self.path.mkdir(parents=True, exist_ok=True)

    def snapshot(self, config: Mapping[str, Any]) -> Path:
        """Write config and environment metadata."""

        cfg_path = dump_config(config, self.path / "config.yaml")
        info = {
            "name": self.name,
            "created_at": time.time(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pid": os.getpid(),
            "git_revision": git_revision(),
            "metadata": self.metadata,
        }
        (self.path / "run.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        return cfg_path

    def log_artifact(self, path: str | Path, kind: str = "artifact") -> None:
        """Append an artifact record."""

        record = {"kind": kind, "path": str(Path(path).resolve()), "time": time.time()}
        out = self.path / "artifacts.jsonl"
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def log_checkpoint_lineage(
        self,
        checkpoint: str | Path,
        parent: str | Path | None = None,
        metrics: Mapping[str, float] | None = None,
    ) -> None:
        """Record checkpoint ancestry and metrics."""

        record = {
            "checkpoint": str(Path(checkpoint).resolve()),
            "parent": str(Path(parent).resolve()) if parent else None,
            "metrics": dict(metrics or {}),
            "time": time.time(),
        }
        with (self.path / "lineage.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
