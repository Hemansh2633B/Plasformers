"""Dataset governance, audit trails, compliance, and privacy scanning."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping


@dataclass
class DatasetLineageRecord:
    """Dataset lineage and fingerprint record."""

    dataset_id: str
    version: str
    source: str
    fingerprint: str
    created_at: float = field(default_factory=time.time)
    parents: list[str] = field(default_factory=list)


class AuditTrail:
    """Append-only audit log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, action: str, actor: str, metadata: Mapping[str, object] | None = None) -> None:
        record = {"time": time.time(), "action": action, "actor": actor, "metadata": dict(metadata or {})}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def dataset_fingerprint(paths: Iterable[str | Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(str(Path(p)) for p in paths):
        digest.update(path.encode("utf-8"))
        p = Path(path)
        if p.exists() and p.is_file():
            digest.update(str(p.stat().st_size).encode("utf-8"))
            digest.update(str(int(p.stat().st_mtime)).encode("utf-8"))
    return digest.hexdigest()


def write_lineage(record: DatasetLineageRecord, output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    return out


def pii_scan_text(text: str) -> dict[str, list[str]]:
    patterns = {
        "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "phone": r"\+?\d[\d\-\s]{7,}\d",
        "ssn_like": r"\b\d{3}-\d{2}-\d{4}\b",
    }
    return {name: re.findall(pattern, text) for name, pattern in patterns.items()}


def compliance_report(lineage: DatasetLineageRecord, privacy_findings: Mapping[str, object]) -> dict[str, object]:
    return {"lineage": asdict(lineage), "privacy_findings": dict(privacy_findings), "generated_at": time.time()}
