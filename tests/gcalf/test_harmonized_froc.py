import json

import numpy as np
import pytest

from gcalf_eval.grade_metrics import _greedy_match_boxes, match_grade_predictions
from gcalf_eval.harmonized_froc import (
    froc_curve,
    harmonized_froc_events,
    make_events_payload,
    pool_event_files,
    sensitivity_at_fp,
    summarize_harmonized_froc,
)


BOXES = np.asarray([
    [0, 0, 2, 2, 0, 2],
    [4, 4, 6, 6, 4, 6],
    [8, 8, 10, 10, 8, 10],
    [12, 12, 14, 14, 12, 14],
], dtype=np.float32)


def _prediction(boxes, scores, probabilities):
    return (
        np.asarray(boxes, dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
        np.asarray(probabilities, dtype=np.float32),
    )


def test_iou_threshold_is_05_while_existing_grade_matcher_keeps_01_default():
    gt = np.asarray([[0, 0, 2, 2, 0, 2]], dtype=np.float32)
    pred = np.asarray([[0, 0, 1, 1, 0, 1]], dtype=np.float32)
    scores = np.asarray([0.9], dtype=np.float32)
    probabilities = np.asarray([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    generic, grade_events = harmonized_froc_events(
        pred, scores, probabilities, gt, np.asarray([3]), np.asarray([True])
    )
    assert generic[0]["fp"] is True
    assert grade_events[3][0]["fp"] is True

    truth, predicted, misses, false_positives = match_grade_predictions(
        pred, scores, probabilities, gt, np.asarray([3])
    )
    assert truth == [3]
    assert predicted == [3]
    assert misses == 0
    assert false_positives == 0


def test_froc_curve_groups_tied_scores_and_uses_conservative_operating_points():
    events = [
        {"score": 0.9, "grade": 2, "fp": False, "patient": "p1"},
        {"score": 0.8, "grade": None, "fp": True, "patient": "p1"},
        {"score": 0.8, "grade": 3, "fp": False, "patient": "p2"},
        {"score": 0.7, "grade": 4, "fp": False, "patient": "p2"},
        {"score": 0.6, "grade": None, "fp": True, "patient": "benign"},
    ]
    fp_rates, sensitivities = froc_curve(
        events, num_patients=2, support={2: 1, 3: 1, 4: 1, 5: 0}
    )

    assert fp_rates == [0.0, 0.0, 0.5, 0.5, 1.0]
    assert sensitivities[2] == [0.0, 1.0, 1.0, 1.0, 1.0]
    assert sensitivities[3] == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert sensitivities[5] == [0.0] * 5
    assert sensitivity_at_fp([0.0, 0.5, 1.0], [0.0, 0.2, 1.0], 0.75) == pytest.approx(0.2)
    assert sensitivity_at_fp([0.0, 0.5, 0.5, 1.0], [0.0, 0.2, 0.4, 1.0], 0.75) == pytest.approx(0.4)
    assert sensitivity_at_fp([0.0, 0.5], [0.0, 0.5], 1.0) == pytest.approx(0.5)
    assert sensitivity_at_fp([0.0, 1.0], [0.0, 1.0], 0.5) == pytest.approx(0.0)


def test_generic_stratification_ignores_predicted_grade():
    grades = np.asarray([2, 3, 4, 5])
    supervised = np.ones(4, dtype=bool)
    boxes, scores, probabilities = _prediction(
        BOXES,
        [0.9, 0.8, 0.7, 0.6],
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    )
    generic, _ = harmonized_froc_events(boxes, scores, probabilities, BOXES, grades, supervised)
    permuted, _ = harmonized_froc_events(
        boxes, scores, probabilities[:, ::-1], BOXES, grades, supervised
    )
    assert [event["grade"] for event in generic] == [2, 3, 4, 5]
    assert [event["grade"] for event in permuted] == [2, 3, 4, 5]


def test_generic_matching_absorbs_unsupervised_lesions_but_grade_matching_counts_them_as_fp():
    boxes, scores, probabilities = _prediction(
        BOXES[:2], [0.9, 0.8], [[0.9, 0.1, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]]
    )
    generic, grade_events = harmonized_froc_events(
        boxes,
        scores,
        probabilities,
        BOXES[:2],
        np.asarray([-1, 2]),
        np.asarray([False, True]),
    )
    assert generic[0]["score"] == pytest.approx(0.9)
    assert generic[0]["grade"] is None
    assert generic[0]["fp"] is False
    assert generic[1]["grade"] == 2
    assert generic[1]["fp"] is False
    assert grade_events[2][0]["fp"] is True
    assert grade_events[2][1]["tp"] is True


def test_grade_specific_score_is_detection_confidence_times_grade_probability():
    boxes, scores, probabilities = _prediction(
        BOXES[:2], [0.9, 0.5], [[0.1, 0.1, 0.1, 0.7], [0.1, 0.8, 0.05, 0.05]]
    )
    _, grade_events = harmonized_froc_events(
        boxes, scores, probabilities, BOXES[:2], np.asarray([2, 3]), np.asarray([True, True])
    )
    assert [event["score"] for event in grade_events[3]] == pytest.approx([0.09, 0.4])
    assert grade_events[3][1]["tp"] is True


def test_generic_and_grade_specific_fp_semantics_differ_by_target_grade():
    boxes, scores, probabilities = _prediction(
        BOXES[:1], [0.9], [[0.0, 0.1, 0.9, 0.0]]
    )
    generic, grade_events = harmonized_froc_events(
        boxes, scores, probabilities, BOXES[:1], np.asarray([3]), np.asarray([True])
    )
    assert generic[0]["grade"] == 3
    assert generic[0]["fp"] is False
    assert grade_events[4][0]["tp"] is False
    assert grade_events[4][0]["fp"] is True
    assert grade_events[3][0]["tp"] is True


def test_full_matching_has_prefix_equivalence_and_summary_handles_zero_support():
    gt = BOXES[:3]
    grades = np.asarray([2, 3, 4])
    supervised = np.ones(3, dtype=bool)
    pred_boxes, pred_scores, probabilities = _prediction(
        [BOXES[1], BOXES[0], BOXES[1], BOXES[2]],
        [0.8, 0.9, 0.7, 0.6],
        [[0.25] * 4] * 4,
    )
    generic, grade_events = harmonized_froc_events(
        pred_boxes, pred_scores, probabilities, gt, grades, supervised
    )
    for threshold in sorted(set(pred_scores), reverse=True):
        keep = pred_scores >= threshold
        matched, _ = _greedy_match_boxes(pred_boxes[keep], pred_scores[keep], gt, 0.5)
        expected_tp = int(np.sum(matched >= 0))
        actual_tp = sum(not event["fp"] for event in generic if event["score"] >= threshold)
        assert actual_tp == expected_tp
    fp_rates, sensitivity = froc_curve(generic, 1, {2: 1, 3: 1, 4: 1, 5: 0})
    assert all(value >= 0.0 for value in fp_rates)
    assert all(value is None or 0.0 <= value <= 1.0 for values in sensitivity.values() for value in values)
    assert sum(not event["fp"] for event in generic) <= len(gt)
    assert sum(event["tp"] for event in grade_events[2]) <= 1

    summary = summarize_harmonized_froc(
        generic,
        grade_events,
        {2: 1, 3: 1, 4: 1, 5: 0},
        num_patients=1,
        num_cases=1,
    )
    assert summary["gt_lesions_per_grade"]["GGG5"] == 0
    zero_support = [record for record in summary["records"] if record["grade"] == 5]
    assert all(record["sensitivity"] is None for record in zero_support)


def test_pooling_events_matches_combined_run(tmp_path):
    first_generic = [{"score": 0.9, "grade": 2, "fp": False, "patient": "p1"}]
    second_generic = [{"score": 0.8, "grade": 3, "fp": False, "patient": "p2"}]
    empty_grade_events = {grade: [] for grade in (2, 3, 4, 5)}
    first_grade = {grade: list(events) for grade, events in empty_grade_events.items()}
    second_grade = {grade: list(events) for grade, events in empty_grade_events.items()}
    first_grade[2] = [{"score": 0.9, "tp": True, "fp": False, "patient": "p1"}]
    second_grade[3] = [{"score": 0.8, "tp": True, "fp": False, "patient": "p2"}]
    support = {2: 1, 3: 0, 4: 0, 5: 0}
    first = make_events_payload(first_generic, first_grade, support, ["p1"], 1)
    second = make_events_payload(second_generic, second_grade, {2: 0, 3: 1, 4: 0, 5: 0}, ["p2"], 1)
    first_path = tmp_path / "fold0.json"
    second_path = tmp_path / "fold1.json"
    first_path.write_text(json.dumps(first))
    second_path.write_text(json.dumps(second))

    pooled, _ = pool_event_files([first_path, second_path])
    combined = summarize_harmonized_froc(
        first_generic + second_generic,
        {grade: first_grade[grade] + second_grade[grade] for grade in (2, 3, 4, 5)},
        {2: 1, 3: 1, 4: 0, 5: 0},
        num_patients=2,
        num_cases=2,
    )
    assert pooled == combined
