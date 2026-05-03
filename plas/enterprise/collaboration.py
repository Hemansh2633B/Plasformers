"""Collaboration platform primitives."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass
class Workspace:
    """Team workspace."""

    name: str
    members: list[str] = field(default_factory=list)
    experiments: list[str] = field(default_factory=list)


@dataclass
class AnnotationReview:
    """Annotation review item."""

    sample_id: str
    reviewer: str
    decision: str
    comments: str = ""
    time: float = field(default_factory=time.time)


def share_experiment(workspace: Workspace, experiment_path: str | Path) -> Workspace:
    workspace.experiments.append(str(experiment_path))
    return workspace


def write_workspace(workspace: Workspace, output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(workspace), indent=2), encoding="utf-8")
    return out


def model_comparison_dashboard(records: list[Mapping[str, object]]) -> dict[str, object]:
    return {"models": [dict(record) for record in records], "generated_at": time.time()}
