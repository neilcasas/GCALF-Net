import torch

from nndet.arch.conv import ConvGroupRelu, Generator
from nndet.arch.heads.classifier import AsymmetricFocalClassifier
from nndet.arch.heads.comb import DetectionHead
from nndet.arch.heads.regressor import MedicalSmallTargetRegressor
from nndet.core.boxes.coder import BoxCoderND


def _build_head(num_classes=1, anchors_per_pos=3):
    torch.manual_seed(0)
    conv = Generator(ConvGroupRelu, 3)
    # internal_channels must be >= ConvGroupRelu's default norm_channels_per_group
    # (16, nndet/arch/conv.py:237): a smaller value makes GroupNorm compute
    # num_groups=0 and crash with ZeroDivisionError.
    classifier = AsymmetricFocalClassifier(
        conv=conv,
        in_channels=8,
        internal_channels=16,
        num_classes=num_classes,
        anchors_per_pos=anchors_per_pos,
        num_levels=1,
    )
    regressor = MedicalSmallTargetRegressor(
        conv=conv,
        in_channels=8,
        internal_channels=16,
        anchors_per_pos=anchors_per_pos,
        num_levels=1,
        small_target_enhancement=False,
    )
    coder = BoxCoderND(weights=(1.,) * 6)
    # DetectionHead.forward/postprocess_for_inference are shared, unmodified
    # by DetectionHeadHNMNative (the class the live configs actually build) --
    # exercising the base class here avoids constructing a sampler that only
    # compute_loss (not forward) needs.
    return DetectionHead(classifier=classifier, regressor=regressor, coder=coder).cpu().eval()


def test_detection_head_forward_pass_returns_anchor_consistent_shapes():
    head = _build_head(num_classes=1, anchors_per_pos=3)
    feature_map = torch.randn(1, 8, 4, 16, 16)
    num_positions = 4 * 16 * 16
    num_anchors = num_positions * 3

    with torch.no_grad():
        prediction = head([feature_map])

    # binary csPCa detection endpoint: classifier_classes=1 in the live configs.
    assert tuple(prediction["box_logits"].shape) == (num_anchors, 1)
    assert tuple(prediction["box_deltas"].shape) == (num_anchors, 6)
    assert torch.isfinite(prediction["box_logits"]).all()
    assert torch.isfinite(prediction["box_deltas"]).all()


def test_detection_head_postprocess_decodes_boxes_and_probabilities():
    head = _build_head(num_classes=1, anchors_per_pos=3)
    feature_map = torch.randn(1, 8, 4, 16, 16)
    num_anchors = 4 * 16 * 16 * 3
    anchors = [torch.tensor([[0., 0., 4., 4., 0., 4.]] * num_anchors)]

    with torch.no_grad():
        prediction = head([feature_map])
        postprocessed = head.postprocess_for_inference(prediction, anchors)

    assert tuple(postprocessed["pred_boxes"].shape) == (num_anchors, 6)
    assert tuple(postprocessed["pred_probs"].shape) == (num_anchors, 1)
    assert bool((postprocessed["pred_probs"] >= 0).all())
    assert bool((postprocessed["pred_probs"] <= 1).all())
    assert torch.isfinite(postprocessed["pred_boxes"]).all()
