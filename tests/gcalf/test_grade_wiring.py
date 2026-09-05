"""Model-side grade-loss wiring (ARCHITECTURE.md Sec 8): matcher-based grade-target
gathering (BaseRetinaNet.assign_grades_to_anchors) and the per-anchor feature
extractor + head that produce grade logits laid out like box_logits
(GradeAnchorFeatureExtractor, GradeClassifierHead).

Data-pipeline threading (raw label json -> boxes_file pickle -> training batch
`target_grades`/`target_grade_supervised`) is a separate, not-yet-approved
change to the preprocessing pickle format and is out of scope here; these
tests exercise the model-side plumbing directly with hand-built targets.
"""
import torch

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


def test_no_ground_truth_boxes_yields_all_unsupervised():
    model = _make_retina_net_for_matching(anchors_per_level=[3])
    anchors = [torch.zeros(3, 4)]
    boxes = [torch.zeros(0, 4)]
    grades = [torch.zeros(0, dtype=torch.long)]
    supervised = [torch.zeros(0, dtype=torch.bool)]

    matched_grades, matched_grade_supervised = model.assign_grades_to_anchors(anchors, boxes, grades, supervised)

    assert matched_grades[0].shape == (3,)
    assert not matched_grade_supervised[0].any()


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
