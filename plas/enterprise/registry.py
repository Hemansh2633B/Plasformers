"""Central model registry with semantic versions, signing, rollback, and promotion."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

from .security import SignedArtifact, sign_artifact


@dataclass
class RegistryEntry:
    """Versioned registry entry."""

    name: str
    version: str
    artifact: str
    stage: str = "dev"
    benchmarks: dict[str, float] = field(default_factory=dict)
    parent_version: str | None = None
    signature: SignedArtifact | None = None
    created_at: float = field(default_factory=time.time)


class EnterpriseModelRegistry:
    """Filesystem-backed model registry."""

    def __init__(self, root: str | Path = "registry") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "index.jsonl"

    def register(
        self,
        name: str,
        version: str,
        artifact: str | Path,
        *,
        secret: str | None = None,
        benchmarks: Mapping[str, float] | None = None,
        parent_version: str | None = None,
    ) -> RegistryEntry:
        signature = sign_artifact(artifact, secret) if secret else None
        entry = RegistryEntry(
            name=name,
            version=version,
            artifact=str(Path(artifact).resolve()),
            benchmarks=dict(benchmarks or {}),
            parent_version=parent_version,
            signature=signature,
        )
        with self.index.open("a", encoding="utf-8") as handle:
            payload = asdict(entry)
            handle.write(json.dumps(payload) + "\n")
        return entry

    def list(self, name: str | None = None) -> list[RegistryEntry]:
        if not self.index.exists():
            return []
        entries = []
        for line in self.index.read_text(encoding="utf-8").splitlines():
            data = json.loads(line)
            if data.get("signature"):
                data["signature"] = SignedArtifact(**data["signature"])
            entry = RegistryEntry(**data)
            if name is None or entry.name == name:
                entries.append(entry)
        return entries

    def promote(self, name: str, version: str, stage: str) -> RegistryEntry:
        entries = self.list(name)
        for entry in reversed(entries):
            if entry.version == version:
                promoted = RegistryEntry(**{**asdict(entry), "stage": stage})
                with self.index.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(promoted)) + "\n")
                return promoted
        raise KeyError(f"No registry entry for {name}:{version}")

    def rollback(self, name: str, stage: str = "prod") -> RegistryEntry:
        candidates = [entry for entry in self.list(name) if entry.stage == stage]
        if len(candidates) < 2:
            raise RuntimeError(f"No previous {stage} version available for {name}")
        return candidates[-2]
