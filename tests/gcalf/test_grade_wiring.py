"""Model-side grade-loss wiring (ARCHITECTURE.md Sec 8): matcher-based grade-target
gathering (BaseRetinaNet.assign_grades_to_anchors) and the per-anchor feature
extractor + head that produce grade logits laid out like box_logits
(GradeAnchorFeatureExtractor, GradeClassifierHead).

Data-pipeline threading (raw label json -> boxes_file pickle -> training batch
`target_grades`/`target_grade_supervised`) is a separate, not-yet-approved
change to the preprocessing pickle format and is out of scope here; these
tests exercise the model-side plumbing directly with hand-built targets.
"""
import pytest
import torch
import torch.nn as nn

from nndet.arch.conv import ConvInstanceRelu, Generator
from nndet.arch.encoder.gcalf.grade_head import GradeHead
from nndet.arch.heads.grade_classifier import GradeAnchorFeatureExtractor, GradeClassifierHead
from nndet.core.boxes.matcher.iou import IoUMatcher
from nndet.core.retina import BaseRetinaNet


class _FakeAnchorGenerator:
    def __init__(self, num_anchors_per_level, anchors_per_loc=1):
        self._num_anchors_per_level = num_anchors_per_level
        self._anchors_per_loc = anchors_per_loc

    def get_num_acnhors_per_level(self):
        return self._num_anchors_per_level

    def num_anchors_per_location(self):
        return [self._anchors_per_loc]


def _make_retina_net_for_matching(anchors_per_level):
    # Constructed via __new__ to exercise assign_grades_to_anchors in isolation,
    # without building the full encoder/decoder/head stack it doesn't need.
    model = BaseRetinaNet.__new__(BaseRetinaNet)
    model.proposal_matcher = IoUMatcher(low_threshold=0.3, high_threshold=0.5, allow_low_quality_matches=False)
    model.anchor_generator = _FakeAnchorGenerator(anchors_per_level)
    return model


def test_matched_grades_align_with_iou_matches_and_mask_background():
    model = _make_retina_net_for_matching(anchors_per_level=[2])
    anchors = [torch.tensor([[0., 0., 1., 1.], [5., 5., 6., 6.]])]
    boxes = [torch.tensor([[0., 0., 1., 1.]])]
    grades = [torch.tensor([2])]
    supervised = [torch.tensor([True])]

    matched_grades, matched_grade_supervised = model.assign_grades_to_anchors(anchors, boxes, grades, supervised)

    assert matched_grades[0].dtype == torch.long
    assert matched_grade_supervised[0].dtype == torch.bool
    # Only one GT box exists, so clamp(min=0) gathers its grade (2) for every
    # anchor including the background one -- that gathered value is
    # meaningless there, which is exactly why grade_supervised must be forced
    # False for it below, independent of what the gather returned.
    assert matched_grades[0].tolist() == [2, 2]
    assert matched_grade_supervised[0].tolist() == [True, False]


def test_matched_anchor_feature_export_carries_instance_ids_and_iou():
    model = _make_retina_net_for_matching(anchors_per_level=[4])
    anchors = [torch.tensor([
        [0., 0., 1., 1.],
        [0.1, 0.1, 0.9, 0.9],
        [5., 5., 6., 6.],
        [9., 9., 10., 10.],
    ])]
    boxes = [torch.tensor([[0., 0., 1., 1.], [5., 5., 6., 6.]])]
    grades = [torch.tensor([4, 5])]
    supervised = [torch.tensor([True, True])]
    instance_ids = [torch.tensor([11, 22])]

    _, _, matched_ids, matched_ious = model.assign_grades_to_anchors(
        anchors, boxes, grades, supervised, target_instance_ids=instance_ids)

    assert matched_ids[0].tolist() == [11, 11, 22, 11]
    assert matched_ious[0].tolist() == pytest.approx([1.0, 0.64, 1.0, 0.0])


def test_no_ground_truth_boxes_yields_all_unsupervised():
    model = _make_retina_net_for_matching(anchors_per_level=[3])
    anchors = [torch.zeros(3, 4)]
    boxes = [torch.zeros(0, 4)]
    grades = [torch.zeros(0, dtype=torch.long)]
    supervised = [torch.zeros(0, dtype=torch.bool)]

    matched_grades, matched_grade_supervised = model.assign_grades_to_anchors(anchors, boxes, grades, supervised)

    assert matched_grades[0].shape == (3,)
    assert not matched_grade_supervised[0].any()


class _LossHead:
    def compute_loss(self, prediction, labels, matched_boxes, anchors):
        return {"cls": prediction["grade_logits"].sum() * 0}, torch.tensor([0, 1]), torch.tensor([], dtype=torch.long)


class _RetinaTrainStep(BaseRetinaNet):
    def __init__(self):
        nn.Module.__init__(self)
        self.grade_head = nn.Identity()
        self.grade_head.loss_type = "ce"
        self.segmenter = None
        self.head = _LossHead()
        self.logits = nn.Parameter(torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))

    def forward(self, images):
        return {"grade_logits": self.logits}, [torch.empty(2, 4)], None

    def assign_targets_to_anchors(self, anchors, target_boxes, target_classes):
        return [], []

    def assign_grades_to_anchors(self, anchors, target_boxes, target_grades, target_grade_supervised):
        return target_grades, target_grade_supervised


def test_train_step_returns_grade_loss_weight_outside_the_optimized_losses_dict():
    model = _RetinaTrainStep()
    targets = {
        "target_boxes": [torch.empty(2, 4)],
        "target_classes": [torch.tensor([0, 0])],
        "target_seg": torch.empty(1),
        "target_grades": [torch.tensor([2, 3])],
        "target_grade_supervised": [torch.tensor([True, False])],
        "grade_class_weights": torch.tensor([1.0, 2.0, 4.0, 8.0]),
    }

    losses, _, log_scalars = model.train_step(torch.empty(1), targets, evaluation=False, batch_num=0)

    assert log_scalars["grade_loss_weight"].item() == 1.0
    assert "grade_loss_weight" not in losses


def test_train_step_has_no_grade_log_scalar_when_grade_targets_are_absent():
    model = _RetinaTrainStep()
    losses, _, log_scalars = model.train_step(
        torch.empty(1), {"target_boxes": [], "target_classes": [], "target_seg": torch.empty(1)},
        evaluation=False, batch_num=0)

    assert log_scalars == {}
    assert "grade" not in losses


def test_grade_classifier_head_output_count_matches_box_logits_arithmetic():
    """GradeClassifierHead's per-anchor count must satisfy the same
    anchors_per_pos * spatial_positions arithmetic DetectionHead.forward's
    box_logits does, so the two align position-for-position after both are
    concatenated per-image (BaseRetinaNet.train_step)."""
    torch.manual_seed(0)
    conv = Generator(ConvInstanceRelu, 3)
    anchors_per_pos = 2
    feature_extractor = GradeAnchorFeatureExtractor(
        conv=conv, in_channels=8, internal_channels=6,
        anchors_per_pos=anchors_per_pos, num_levels=2)
    head = GradeClassifierHead(feature_extractor, GradeHead(in_channels=6, num_grades=4))

    level0 = torch.randn(1, 8, 4, 4, 4)
    level1 = torch.randn(1, 8, 2, 2, 2)
    logits = head([level0, level1])

    expected_anchors = anchors_per_pos * (4 * 4 * 4 + 2 * 2 * 2)
    assert logits.shape == (expected_anchors, 4)


def test_grade_feature_rows_repeat_per_voxel_not_per_block():
    """Guards the row contract (ARCHITECTURE.md F9): rows are voxel-major, anchor
    fastest (`row = voxel * anchors_per_pos + anchor`). A `repeat`-vs-`repeat_interleave`
    mix-up would be shape-identical everywhere else in the stack but would silently
    attach each detection's grade feature to the wrong voxel."""
    torch.manual_seed(0)
    conv = Generator(ConvInstanceRelu, 3)
    anchors_per_pos = 3
    internal_channels = 5
    feature_extractor = GradeAnchorFeatureExtractor(
        conv=conv, in_channels=4, internal_channels=internal_channels,
        anchors_per_pos=anchors_per_pos, num_levels=1)

    # Each voxel of this 3x3x3 map holds a distinct constant so the extractor's
    # output is expected to differ from voxel to voxel.
    num_voxels = 3 * 3 * 3
    x = torch.arange(num_voxels, dtype=torch.float32).view(1, 1, 3, 3, 3).expand(1, 4, 3, 3, 3).contiguous()

    features = feature_extractor(x, level=0)

    assert features.shape == (1, num_voxels * anchors_per_pos, internal_channels)
    for voxel in range(num_voxels):
        block = features[0, voxel * anchors_per_pos:(voxel + 1) * anchors_per_pos]
        assert torch.equal(block, block[0].expand_as(block))
    assert not torch.equal(features[0, 0], features[0, anchors_per_pos])


def test_grade_anchor_feature_extractor_does_not_scale_conv_out_with_anchors():
    """Default mode (`per_anchor_features=False`, 1.33M params): `build_conv_out`
    emits `internal_channels` regardless of `anchors_per_pos`."""
    conv = Generator(ConvInstanceRelu, 3)
    small = GradeAnchorFeatureExtractor(
        conv=conv, in_channels=4, internal_channels=8, anchors_per_pos=1, num_levels=1, per_anchor_features=False)
    large = GradeAnchorFeatureExtractor(
        conv=conv, in_channels=4, internal_channels=8, anchors_per_pos=27, num_levels=1, per_anchor_features=False)

    small_params = sum(p.numel() for p in small.conv_out.parameters())
    large_params = sum(p.numel() for p in large.conv_out.parameters())

    assert small_params == large_params
    # Conv3d(8 -> 8, k=3) + bias: 8*8*27 + 8.
    assert large_params == 8 * 8 * 27 + 8


def test_grade_anchor_feature_extractor_scales_conv_out_when_per_anchor_enabled():
    """Legacy/Ablation mode (`per_anchor_features=True`, 12.83M params):
    `build_conv_out` emits `internal_channels * anchors_per_pos` channels."""
    conv = Generator(ConvInstanceRelu, 3)
    small = GradeAnchorFeatureExtractor(
        conv=conv, in_channels=4, internal_channels=8, anchors_per_pos=1, num_levels=1, per_anchor_features=True)
    large = GradeAnchorFeatureExtractor(
        conv=conv, in_channels=4, internal_channels=8, anchors_per_pos=27, num_levels=1, per_anchor_features=True)

    small_params = sum(p.numel() for p in small.conv_out.parameters())
    large_params = sum(p.numel() for p in large.conv_out.parameters())

    # Small: Conv3d(8 -> 8*1, k=3) + bias: 8*8*27 + 8.
    assert small_params == 8 * 8 * 27 + 8
    # Large: Conv3d(8 -> 8*27, k=3) + bias: 8*216*27 + 216.
    assert large_params == 8 * 216 * 27 + 216
    assert large_params > small_params * 26

    # Verify output shape matches contract
    x = torch.randn(2, 4, 3, 3, 3)
    out_small = small(x, level=0)
    out_large = large(x, level=0)
    assert out_small.shape == (2, 27 * 1, 8)
    assert out_large.shape == (2, 27 * 27, 8)


def test_grade_classifier_head_gradients_reach_both_stages():
    torch.manual_seed(0)
    conv = Generator(ConvInstanceRelu, 3)
    feature_extractor = GradeAnchorFeatureExtractor(
        conv=conv, in_channels=4, internal_channels=6, anchors_per_pos=1, num_levels=1)
    grade_head = GradeHead(in_channels=6, num_grades=4)
    head = GradeClassifierHead(feature_extractor, grade_head)

    x = torch.randn(1, 4, 2, 2, 2, requires_grad=True)
    head([x]).sum().backward()

    assert x.grad is not None and x.grad.abs().sum() > 0
    assert grade_head.classifier.weight.grad is not None
    assert grade_head.classifier.weight.grad.abs().sum() > 0


def test_grade_classifier_head_delegates_probability_conversion():
    grade_head = GradeHead(in_channels=6, loss_type="coral")
    head = GradeClassifierHead(nn.Identity(), grade_head)
    logits = torch.tensor([[1.0, 0.0, -1.0]])

    assert torch.equal(head.logits_to_probs(logits), grade_head.logits_to_probs(logits))
    assert head.loss_type == "coral"
