import json

import numpy as np
import pytest
import SimpleITK as sitk

from gcalf_eval.grade_metrics import lesion_instances, match_grade_predictions, summarize_grade_matches


def test_lesion_instances_keeps_nndetection_native_array_axis_order(tmp_path):
    labels = np.zeros((7, 11, 13), dtype=np.uint8)  # SimpleITK array order: z, y, x.
    labels[1:4, 5:9, 2:7] = 1
    label_path = tmp_path / "case.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(labels), str(label_path))
    label_path.with_suffix("").with_suffix(".json").write_text(json.dumps({
        "instances": {"1": {}},
        "grade_supervised": {"1": True},
        "grades": {"1": 3},
    }))

    boxes, grades, supervised = lesion_instances(label_path)

    assert boxes.tolist() == [[1.0, 5.0, 4.0, 9.0, 2.0, 7.0]]
    assert grades.tolist() == [3]
    assert supervised.tolist() == [True]


def test_grade_matching_uses_detection_score_then_iou_and_reports_misses_and_false_positives():
    gt_boxes = np.array([[0, 0, 2, 2, 0, 2], [4, 4, 6, 6, 4, 6]], dtype=np.float32)
    gt_grades = np.array([2, 5])
    pred_boxes = np.array([[4, 4, 6, 6, 4, 6], [0, 0, 2, 2, 0, 2], [8, 8, 9, 9, 8, 9]], dtype=np.float32)
    pred_scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    pred_probs = np.array([[0.1, 0.1, 0.1, 0.7], [0.8, 0.1, 0.05, 0.05], [0.25] * 4], dtype=np.float32)

    truth, predicted, misses, false_positives = match_grade_predictions(
        pred_boxes, pred_scores, pred_probs, gt_boxes, gt_grades)

    assert truth == [5, 2]
    assert predicted == [5, 2]
    assert misses == 0
    assert false_positives == 1


def test_grade_summary_has_four_by_four_confusion_and_weighted_f1():
    summary = summarize_grade_matches([2, 3, 3], [2, 2, 3], misses=1, false_positives=2)

    assert summary["grade_confusion_matrix"] == [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert summary["grade_matched_detections"] == 3
    assert summary["grade_missed_supervised"] == 1
    assert summary["grade_false_positives"] == 2
    assert 0.0 <= summary["grade_weighted_f1"] <= 1.0
    assert summary["grade_score_threshold"] == 0.0
    assert "grade_false_positives_per_case" not in summary


def test_grade_summary_reports_a_per_case_rate_and_the_threshold_it_was_computed_at():
    summary = summarize_grade_matches([2, 3], [2, 3], misses=0, false_positives=4,
                                      num_cases=2, score_threshold=0.3)

    assert summary["grade_false_positives_per_case"] == pytest.approx(2.0)
    assert summary["grade_score_threshold"] == pytest.approx(0.3)


def test_grade_summary_reports_classification_and_calibration_metrics_for_matched_lesions():
    summary = summarize_grade_matches(
        [2, 2, 3, 3], [2, 2, 3, 2], misses=1, false_positives=2,
        matched_probabilities=[
            [0.9, 0.05, 0.03, 0.02], [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.7, 0.1, 0.1], [0.7, 0.2, 0.05, 0.05],
        ],
    )

    assert summary["grade_accuracy"] == pytest.approx(0.75)
    assert summary["grade_macro_f1"] == pytest.approx((0.8 + (2.0 / 3.0)) / 2.0)
    assert summary["grade_balanced_accuracy"] == pytest.approx(0.75)
    assert summary["grade_ovr_auroc"]["GGG2"] == pytest.approx(1.0)
    assert summary["grade_ovr_auroc"]["GGG3"] == pytest.approx(1.0)
    assert summary["grade_ovr_auroc"]["GGG4"] is None
    assert summary["grade_macro_ovr_auroc"] == pytest.approx(1.0)
    assert 0.0 <= summary["grade_expected_calibration_error"] <= 1.0
    assert summary["grade_mean_confidence_incorrect"] == pytest.approx(0.7)


def test_several_boxes_on_the_same_lesion_count_as_one_match_and_the_rest_as_false_positives():
    gt_boxes = np.array([[0, 0, 2, 2, 0, 2]], dtype=np.float32)
    gt_grades = np.array([3])
    pred_boxes = np.tile(gt_boxes, (3, 1))
    pred_scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    pred_probs = np.array([[0.0, 1.0, 0.0, 0.0]] * 3, dtype=np.float32)

    truth, predicted, misses, false_positives = match_grade_predictions(
        pred_boxes, pred_scores, pred_probs, gt_boxes, gt_grades)

    assert truth == [3]
    assert predicted == [3]
    assert misses == 0
    assert false_positives == 2


def test_score_threshold_drops_low_scoring_predictions_before_matching():
    gt_boxes = np.array([[0, 0, 2, 2, 0, 2]], dtype=np.float32)
    gt_grades = np.array([3])
    pred_boxes = np.tile(gt_boxes, (2, 1))
    pred_scores = np.array([0.9, 0.1], dtype=np.float32)
    pred_probs = np.array([[0.0, 1.0, 0.0, 0.0], [0.25] * 4], dtype=np.float32)

    truth, predicted, misses, false_positives = match_grade_predictions(
        pred_boxes, pred_scores, pred_probs, gt_boxes, gt_grades)
    assert false_positives == 1  # default threshold (0.0): both predictions are matched against

    truth, predicted, misses, false_positives = match_grade_predictions(
        pred_boxes, pred_scores, pred_probs, gt_boxes, gt_grades, score_threshold=0.5)
    assert truth == [3]
    assert predicted == [3]
    assert misses == 0
    assert false_positives == 0  # the 0.1-scoring duplicate is dropped before matching


def test_matching_uses_nndetection_box_order_and_does_not_penalize_ungraded_true_positives():
    gt_boxes = np.array([[1, 2, 5, 6, 3, 7], [10, 11, 12, 13, 14, 15]], dtype=np.float32)
    gt_grades = np.array([2, -1], dtype=np.int64)
    supervised = np.array([True, False])
    pred_boxes = gt_boxes.copy()
    pred_scores = np.array([0.9, 0.8], dtype=np.float32)
    pred_probs = np.array([[0.9, 0.05, 0.03, 0.02], [0.25] * 4], dtype=np.float32)

    truth, predicted, misses, false_positives, matched_ungraded = match_grade_predictions(
        pred_boxes, pred_scores, pred_probs, gt_boxes, gt_grades, supervised, return_details=True
    )

    assert truth == [2]
    assert predicted == [2]
    assert misses == 0
    assert false_positives == 0
    assert matched_ungraded == 1
