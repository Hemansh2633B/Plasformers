import pytest
import torch
from plasformers import build_plasformers

@pytest.mark.benchmark
def test_performance_latency(benchmark):
    model = build_plasformers("nano").eval()
    x = torch.randn(1, 3, 640, 640)

    def run_forward():
        with torch.inference_mode():
            model(x)

    benchmark(run_forward)

@pytest.mark.benchmark
def test_performance_throughput(benchmark):
    model = build_plasformers("nano").eval()
    x = torch.randn(8, 3, 640, 640)

    def run_forward():
        with torch.inference_mode():
            model(x)

    benchmark(run_forward)
