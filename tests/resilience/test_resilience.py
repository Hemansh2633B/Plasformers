import pytest
import torch
import os

@pytest.mark.resilience
def test_resilience_mock_disk_full(tmp_path):
    # This is a mock test for resilience
    checkpoint_path = tmp_path / "checkpoint.pt"

    def save_checkpoint(path):
        if os.environ.get("MOCK_DISK_FULL") == "1":
            raise OSError("No space left on device")
        torch.save({"model": {}}, path)

    # Success case
    save_checkpoint(checkpoint_path)
    assert checkpoint_path.exists()

    # Failure case
    os.environ["MOCK_DISK_FULL"] = "1"
    with pytest.raises(OSError, match="No space left on device"):
        save_checkpoint(checkpoint_path)
    del os.environ["MOCK_DISK_FULL"]
