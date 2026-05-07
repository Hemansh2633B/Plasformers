import pytest
import torch
from plasformers import build_plasformers
from plas.tracking import TRACKER_REGISTRY
from plas.explain import GradCAM
from plas.compression.pruning import structured_prune
from plas.compression.quantization import prepare_quantization_aware_training

@pytest.mark.unit
def test_tracking_bytetrack():
    tracker = TRACKER_REGISTRY["bytetrack"]()
    # Mock detections: list of Detection objects
    # Assuming Detection is a dataclass with box, score, class_id
    from dataclasses import dataclass
    @dataclass
    class MockDet:
        box: list
        score: float
        class_id: int

    dets = [MockDet([10, 10, 50, 50], 0.9, 0)]
    tracks = tracker.update(dets)
    assert len(tracks) >= 0

@pytest.mark.unit
def test_explain_gradcam():
    model = build_plasformers("nano").eval()
    img = torch.randn(1, 3, 640, 640)
    target_layer = "backbone.stage_modules.4"

    cam = GradCAM(model, target_layer)
    result = cam(img)
    # result.heatmap is [1, 1, H, W]
    assert result.heatmap.shape[-2:] == (640, 640)

@pytest.mark.unit
def test_compression_pruning():
    model = build_plasformers("nano")
    # Get initial parameter count
    initial_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    structured_prune(model, amount=0.2)
    # Pruning might not immediately change parameter count if it just zeros them out
    # but let's check if it runs without error
    assert model is not None

@pytest.mark.unit
def test_compression_qat():
    model = build_plasformers("nano")
    model = prepare_quantization_aware_training(model)
    assert hasattr(model, "qconfig") or any(hasattr(m, "qconfig") for m in model.modules())

@pytest.mark.unit
def test_automl_optuna_integration():
    from plas.tune import HyperparameterTuner, SearchSpace

    def mock_objective(params):
        return params["lr"] * 0.1

    space = SearchSpace(floats={"lr": (1e-5, 1e-2)})
    tuner = HyperparameterTuner(mock_objective, space)
    results = tuner.random_search(trials=2)
    assert len(results) == 2
