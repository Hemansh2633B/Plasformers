import pytest
from pathlib import Path
import torch
from plas.core.security import sha256_file, verify_sha256

@pytest.mark.security
def test_integrity_verification(tmp_path):
    artifact = tmp_path / "model.pt"
    artifact.write_text("dummy data")

    digest = sha256_file(artifact)
    assert verify_sha256(artifact, digest)
    assert not verify_sha256(artifact, "wrong-digest")

@pytest.mark.security
def test_safe_torch_load_malicious(tmp_path):
    unsafe_path = tmp_path / "unsafe.pt"
    # Create a malicious-looking pickle
    with open(unsafe_path, "wb") as f:
        f.write(b"cos\nsystem\n(S'ls'\ntR.")

    import torch
    # weights_only=True should prevent arbitrary code execution
    with pytest.raises(Exception):
        torch.load(unsafe_path, weights_only=True)
