import pytest
import torch
import torch.nn.functional as F
from plasformers.losses import (
    varifocal_loss,
    distribution_focal_loss,
    ciou_loss,
    siou_loss
)

@pytest.mark.unit
def test_varifocal_loss():
    # logits: B x N x C, targets: B x N x C, target_scores: B x N x C
    logits = torch.randn(1, 10, 80)
    targets = (torch.rand(1, 10, 80) > 0.5).float()
    target_scores = torch.rand(1, 10, 80)

    loss = varifocal_loss(logits, targets, target_scores)
    assert loss >= 0
    assert not torch.isnan(loss)

@pytest.mark.unit
def test_dfl_loss():
    # pred_dist: N x 4 x (reg_max + 1), target: N x 4
    reg_max = 16
    pred = torch.randn(10, 4, reg_max + 1)
    target = torch.rand(10, 4) * (reg_max - 1)

    loss = distribution_focal_loss(pred, target, reg_max)
    assert loss >= 0
    assert not torch.isnan(loss)

@pytest.mark.unit
def test_iou_losses():
    pred = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    target = torch.tensor([[12.0, 12.0, 22.0, 22.0]])

    closs = ciou_loss(pred, target)
    sloss = siou_loss(pred, target)

    assert closs >= 0
    assert sloss >= 0
    assert not torch.isnan(closs)
    assert not torch.isnan(sloss)

@pytest.mark.unit
def test_numerical_stability_fp16():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable for FP16 test")

    from plasformers import build_plasformers
    model = build_plasformers("nano").cuda().half()
    x = torch.randn(1, 3, 128, 128).cuda().half()

    with torch.inference_mode():
        output = model(x)

    for key in ["outputs"]:
        for layer_out in output[key]:
            for val in layer_out.values():
                assert not torch.isnan(val).any()
                assert not torch.isinf(val).any()

@pytest.mark.unit
def test_loss_finite_difference():
    # Simple check for varifocal_loss gradient
    # varifocal_loss(logits, targets, target_scores, ...)
    logits = torch.tensor([0.5], requires_grad=True)
    targets = torch.tensor([1.0])
    scores = torch.tensor([1.0])

    loss = varifocal_loss(logits, targets, scores)
    loss.backward()
    grad_auto = logits.grad.item()

    eps = 1e-4
    loss_plus = varifocal_loss(logits + eps, targets, scores)
    loss_minus = varifocal_loss(logits - eps, targets, scores)
    grad_num = (loss_plus.item() - loss_minus.item()) / (2 * eps)

    assert abs(grad_auto - grad_num) < 1e-4
