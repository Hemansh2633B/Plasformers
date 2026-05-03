"""Security and integrity utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a file SHA256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str | None) -> bool:
    """Verify a file digest when an expected digest is provided."""

    if not expected:
        return True
    return sha256_file(path).lower() == expected.lower()


def safe_torch_load(path: str | Path, map_location: str = "cpu") -> Any:
    """Load a PyTorch checkpoint with weights-only deserialization when possible."""

    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def write_integrity_manifest(path: str | Path, artifact: str | Path) -> Path:
    """Write a JSON integrity manifest for an artifact."""

    manifest = {
        "artifact": str(Path(artifact).resolve()),
        "sha256": sha256_file(artifact),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out
