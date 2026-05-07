import pytest
import torch
from plasformers import build_plasformers
from pathlib import Path
import os

class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model.export_forward(x)

@pytest.mark.export
def test_export_onnx(tmp_path):
    try:
        import onnx
    except ImportError:
        pytest.skip("onnx unavailable")

    # Use export_forward for ONNX export to avoid dict outputs
    model = build_plasformers("nano", num_classes=80, export_mode=True).eval()
    wrapper = ExportWrapper(model)
    output_path = tmp_path / "model.onnx"
    x = torch.randn(1, 3, 640, 640)

    try:
        torch.onnx.export(
            wrapper,
            (x,),
            str(output_path),
            input_names=['images'],
            output_names=['output'],
            opset_version=12
        )
    except ModuleNotFoundError as e:
        if "onnxscript" in str(e):
            pytest.skip("onnxscript unavailable, required by torch.onnx.export in this environment")
        raise e

    assert output_path.exists()

@pytest.mark.export
def test_export_torchscript(tmp_path):
    # Use export_forward for TorchScript to avoid dict outputs
    model = build_plasformers("nano", num_classes=80, export_mode=True).eval()
    wrapper = ExportWrapper(model)
    output_path = tmp_path / "model.pt"
    x = torch.randn(1, 3, 224, 224)

    scripted_model = torch.jit.trace(wrapper, (x,))
    scripted_model.save(str(output_path))

    assert output_path.exists()

@pytest.mark.integration
def test_inference_engine_basic():
    from plas.engine.inference import InferenceEngine
    engine = InferenceEngine(variant="nano")

    # Generate mock image
    import cv2
    import numpy as np
    img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

    # We might need to mock cv2.imread if we want to pass a path
    # or use a temp file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, img)
        result = engine.predict(Path(tmp.name))

    assert hasattr(result, "detections")
    assert hasattr(result, "fps")
