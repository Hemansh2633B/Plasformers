"""Enterprise security: signing, watermarking, IP protection, and enclave hooks."""

from __future__ import annotations

import hmac
import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plas.core.security import sha256_file


@dataclass(frozen=True)
class SignedArtifact:
    """Signed artifact metadata."""

    path: str
    sha256: str
    signature: str
    algorithm: str = "hmac-sha256"


def sign_artifact(path: str | Path, secret: str) -> SignedArtifact:
    digest = sha256_file(path)
    signature = hmac.new(secret.encode("utf-8"), digest.encode("utf-8"), "sha256").hexdigest()
    return SignedArtifact(str(path), digest, signature)


def verify_artifact_signature(record: SignedArtifact, secret: str) -> bool:
    expected = hmac.new(secret.encode("utf-8"), record.sha256.encode("utf-8"), "sha256").hexdigest()
    return hmac.compare_digest(expected, record.signature)


def write_signature(record: SignedArtifact, output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record.__dict__, indent=2), encoding="utf-8")
    return out


def watermark_payload(owner: str, model_id: str) -> dict[str, Any]:
    return {"owner": owner, "model_id": model_id, "watermark": secrets.token_hex(16)}


def secure_enclave_request(model_artifact: str | Path) -> dict[str, Any]:
    return {"artifact": str(model_artifact), "requires": ["attestation", "sealed_storage", "encrypted_inputs"]}
