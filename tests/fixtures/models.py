import pytest
import torch
from plasformers import build_plasformers

@pytest.fixture
def model_nano():
    return build_plasformers("nano", num_classes=80)

@pytest.fixture
def model_small():
    return build_plasformers("small", num_classes=80)

@pytest.fixture
def model_base():
    return build_plasformers("base", num_classes=80)

@pytest.fixture
def model_large():
    return build_plasformers("large", num_classes=80)

@pytest.fixture
def model_xlarge():
    return build_plasformers("xlarge", num_classes=80)

@pytest.fixture(params=["nano", "small", "base", "large", "xlarge"])
def any_model(request):
    return build_plasformers(request.param, num_classes=80)
