import pytest
import torch
import os


@pytest.fixture(scope="session")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Hardware availability checks
def has_tensorrt():
    try:
        import tensorrt
        return True
    except ImportError:
        return False

def has_openvino():
    try:
        import openvino
        return True
    except ImportError:
        return False

def has_coreml():
    try:
        import coremltools
        return True
    except ImportError:
        return False

def has_onnxruntime():
    try:
        import onnxruntime
        return True
    except ImportError:
        return False

@pytest.fixture(scope="session")
def check_hardware(request):
    hw = request.param
    if hw == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    elif hw == "tpu":
        try:
            import torch_xla
        except ImportError:
            pytest.skip("TPU (torch_xla) unavailable")
    elif hw == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("Apple Metal (MPS) unavailable")
    elif hw == "tensorrt" and not has_tensorrt():
        pytest.skip("TensorRT unavailable")
    elif hw == "openvino" and not has_openvino():
        pytest.skip("OpenVINO unavailable")
    elif hw == "coreml" and not has_coreml():
        pytest.skip("CoreML unavailable")
    elif hw == "onnxruntime" and not has_onnxruntime():
        pytest.skip("ONNX Runtime unavailable")
