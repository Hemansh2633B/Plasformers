import pytest
import torch
from plasformers import build_plasformers
from plasformers.modules import (
    AdaptiveHybridStem,
    TriPathBlock,
    AdaptiveFusionGate,
    SpectralGatingBlock,
    CompressedLinearAttention,
    LocalPathBlock
)

@pytest.mark.unit
@pytest.mark.parametrize("variant", ["nano", "small", "base", "large", "xlarge"])
def test_model_variants_forward(variant):
    model = build_plasformers(variant, num_classes=80).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        output = model(x)

    # Expected output is a dict with 'outputs' key in eval mode
    assert isinstance(output, dict)
    assert "outputs" in output
    assert len(output["outputs"]) == 5

@pytest.mark.unit
def test_adaptive_hybrid_stem():
    stem = AdaptiveHybridStem(in_channels=3, out_channels=32)
    x = torch.randn(1, 3, 128, 128)
    y = stem(x)
    assert y.shape == (1, 32, 32, 32)

@pytest.mark.unit
def test_tri_path_block():
    block = TriPathBlock(
        channels=64,
        heads=4,
        token_grid=8,
        local_kernel=3,
        expansion=2.0
    )
    x = torch.randn(1, 64, 32, 32)
    y = block(x)
    assert y.shape == (1, 64, 32, 32)

@pytest.mark.unit
def test_gradient_flow():
    model = build_plasformers("nano", num_classes=80).train()
    # Use batch size > 1 for BatchNorm in training mode
    x = torch.randn(2, 3, 128, 128)
    out = model(x)

    # out is a dict with 'outputs' and 'aux_outputs'
    loss = 0
    for key in ["outputs", "aux_outputs"]:
        for layer_out in out[key]:
            for val in layer_out.values():
                loss = loss + val.sum()

    loss.backward()

    # Check some parameters for gradients
    # Note: Some modules like SpectralGatingBlock might have inactive paths (proxy vs FFT)
    # in training mode if not in export mode.
    # backbone.stage_modules.0.0.freq.proxy is only used in export_mode or if use_fft=False
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "freq.proxy" in name and model.config.use_fft and not model.config.export_mode:
                continue
            assert param.grad is not None, f"No gradient for {name}"

@pytest.mark.unit
def test_mixed_precision_stability():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_plasformers("nano", num_classes=80).to(device).train()
    x = torch.randn(2, 3, 128, 128).to(device)

    with torch.amp.autocast(device_type=device, enabled=torch.cuda.is_available()):
        out = model(x)
        loss = 0
        for key in ["outputs", "aux_outputs"]:
            for layer_out in out[key]:
                for val in layer_out.values():
                    loss = loss + val.sum()

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

@pytest.mark.unit
def test_dynamic_input_shapes():
    model = build_plasformers("nano", num_classes=80).eval()
    shapes = [(128, 128), (160, 192), (224, 224)]
    for h, w in shapes:
        x = torch.randn(1, 3, h, w)
        with torch.inference_mode():
            output = model(x)
        assert output is not None

@pytest.mark.unit
def test_spectral_gating_block_fft_vs_proxy():
    block = SpectralGatingBlock(channels=32, use_fft=True)
    x = torch.randn(1, 32, 32, 32)

    # Normal mode (FFT)
    y_fft = block(x)

    # Export mode (Proxy)
    block.export_mode = True
    y_proxy = block(x)

    assert y_fft.shape == y_proxy.shape == (1, 32, 32, 32)
    # They shouldn't be identical but should be valid tensors
    assert not torch.allclose(y_fft, y_proxy)
