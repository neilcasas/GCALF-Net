import torch

from nndet.core.retina import BaseRetinaNet
from nndet.inference.ensembler.detection import BoxEnsembler


def _postprocessor():
    model = BaseRetinaNet.__new__(BaseRetinaNet)
    model.num_foreground_classes = 1
    model.topk_candidates = 10
    model.score_thresh = 0.0
    model.remove_small_boxes = None
    model.detections_per_img = 10
    model.nms_thresh = 0.1
    return model


def test_detection_postprocessing_keeps_grade_probabilities_with_the_selected_anchor():
    model = _postprocessor()
    boxes = torch.tensor([[0., 0., 2., 2.], [4., 4., 6., 6.]])
    probs = torch.tensor([[0.2], [0.9]])
    grade_probs = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.2, 0.6]])

    out_boxes, out_probs, out_labels, out_grades = model.postprocess_detections_single_image(
        boxes, probs, (8, 8), grade_probs=grade_probs)

    assert torch.equal(out_boxes, boxes.flip(0))
    assert torch.equal(out_probs, torch.tensor([0.9, 0.2]))
    assert out_labels.tolist() == [0, 0]
    assert torch.equal(out_grades, grade_probs.flip(0))


def test_ensembler_grade_matching_uses_highest_iou_then_detection_score():
    candidate_boxes = torch.tensor([[0., 0., 2., 2.], [4., 4., 6., 6.]])
    candidate_scores = torch.tensor([0.2, 0.9])
    candidate_labels = torch.tensor([0, 0])
    candidate_grades = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.2, 0.6]])

    grades = BoxEnsembler._match_grade_probs(
        torch.tensor([[4., 4., 6., 6.]]), torch.tensor([0.9]), torch.tensor([0]),
        candidate_boxes, candidate_scores, candidate_labels, candidate_grades,
    )

    assert torch.equal(grades, candidate_grades[1:])
