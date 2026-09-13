import torch

from nndet.arch.conv import ConvInstanceRelu, Generator
from nndet.arch.heads.segmenter import DiCESegmenterFgBg


def _build_segmenter():
    torch.manual_seed(0)
    return DiCESegmenterFgBg(
        conv=Generator(ConvInstanceRelu, 3),
        seg_classes=1,
        in_channels=[8],
        decoder_levels=[1, 2, 3, 4],
    ).cpu().eval()


def test_segmenter_forward_pass_returns_two_class_logits():
    model = _build_segmenter()
    # forward() only consumes the largest decoder level (segmenter.py:179).
    features = [torch.randn(1, 8, 4, 16, 16)]

    with torch.no_grad():
        prediction = model(features)

    seg_logits = prediction["seg_logits"]
    # seg_classes=1 foreground class; the head unconditionally adds one
    # background channel (segmenter.py:46), so 2 output channels are expected.
    assert tuple(seg_logits.shape) == (1, 2, 4, 16, 16)
    assert torch.isfinite(seg_logits).all()


def test_segmenter_postprocess_returns_normalized_probabilities():
    model = _build_segmenter()
    features = [torch.randn(1, 8, 4, 16, 16)]

    with torch.no_grad():
        prediction = model(features)
        postprocessed = model.postprocess_for_inference(prediction)

    probabilities = postprocessed["pred_seg"]
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1, 4, 16, 16), atol=1e-5)
    assert bool((probabilities >= 0).all()) and bool((probabilities <= 1).all())


def test_segmenter_dice_loss_rewards_correct_predictions_over_wrong_ones():
    model = _build_segmenter()
    torch.manual_seed(1)
    target = (torch.rand(1, 4, 16, 16) > 0.5).float()

    # One-hot logits that exactly match (perfect) or exactly invert (wrong)
    # the target, so the resulting seg_dice loss has a known direction: a
    # silently broken dice term would not separate these two cases.
    perfect_logits = torch.stack([1 - target, target], dim=1) * 20.0 - 10.0
    wrong_logits = torch.stack([target, 1 - target], dim=1) * 20.0 - 10.0

    perfect_losses = model.compute_loss({"seg_logits": perfect_logits}, target.clone())
    wrong_losses = model.compute_loss({"seg_logits": wrong_logits}, target.clone())

    assert torch.isfinite(perfect_losses["seg_ce"])
    assert torch.isfinite(perfect_losses["seg_dice"])
    assert torch.isfinite(wrong_losses["seg_dice"])
    assert perfect_losses["seg_dice"].item() < 0.01
    assert perfect_losses["seg_dice"].item() < wrong_losses["seg_dice"].item()
