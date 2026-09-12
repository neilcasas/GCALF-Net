import csv
import json
import math
import pickle

import numpy as np
import pytest

picai_eval = pytest.importorskip("picai_eval")
sitk = pytest.importorskip("SimpleITK")

from gcalf_eval.run_eval import _filter_prediction_by_score, boxes_to_detection_map, run_evaluation


def _write_prediction(directory, case_id, boxes, scores, labels, shape=(4, 4, 4), grade_probs=None):
    directory.mkdir(parents=True, exist_ok=True)
    boxes_array = np.asarray(boxes, dtype=np.float32)
    if boxes_array.size == 0:
        boxes_array = np.empty((0, len(shape) * 2), dtype=np.float32)
    prediction = {
        "original_size_of_raw_data": np.asarray(shape),
        "pred_boxes": boxes_array,
        "pred_scores": np.asarray(scores, dtype=np.float32),
        "pred_labels": np.asarray(labels, dtype=np.int64),
    }
    if grade_probs is not None:
        prediction["pred_grade_probs"] = np.asarray(grade_probs, dtype=np.float32).reshape(-1, 4)
    with (directory / f"{case_id}_boxes.pkl").open("wb") as file:
        pickle.dump(prediction, file)


def _write_label(directory, case_id, array):
    directory.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(np.asarray(array, dtype=np.uint8)), str(directory / f"{case_id}.nii.gz"))


def test_boxes_to_detection_map_uses_one_maximum_score_per_connected_candidate():
    detection_map = boxes_to_detection_map(
        {
            "original_size_of_raw_data": np.asarray([4, 4, 4]),
            "pred_boxes": np.asarray([[0, 0, 3, 3, 0, 3], [1, 1, 5, 5, 1, 5]]),
            "pred_scores": np.asarray([0.2, 0.8]),
            "pred_labels": np.asarray([0, 1]),
        }
    )
    assert detection_map.shape == (4, 4, 4)
    # The overlapping boxes form one lesion candidate, which PI-CAI requires
    # to have a single confidence throughout its connected component.
    assert detection_map[1, 1, 1] == pytest.approx(0.8)
    assert detection_map[0, 0, 0] == pytest.approx(0.8)


def test_boxes_to_detection_map_uses_26_connectivity_matching_picai_eval():
    # picai_eval.analysis_utils.parse_detection_map re-labels y_det itself with
    # 26-connectivity (`label_structure = np.ones((3, 3, 3))`) and raises if its own
    # component count is lower than the number of distinct nonzero values it finds
    # ("softmax volumes instead of detection maps"). A finer connectivity here (e.g.
    # 6-connected) can therefore split two differently-scored regions that
    # picai_eval's own pass merges back together -- which crashed `evaluate()` in
    # practice across every case of a 4-arm smoke check. Corner-only-touching boxes
    # (nearest voxels differing by 1 in every axis) must stay one component, matching
    # picai_eval, not split into two.
    detection_map = boxes_to_detection_map(
        {
            "original_size_of_raw_data": np.asarray([4, 4, 4]),
            "pred_boxes": np.asarray([[0, 0, 2, 2, 0, 2], [2, 2, 4, 4, 2, 4]]),
            "pred_scores": np.asarray([0.9, 0.1]),
            "pred_labels": np.asarray([0, 0]),
        }
    )
    assert detection_map[0, 0, 0] == pytest.approx(0.9)
    assert detection_map[3, 3, 3] == pytest.approx(0.9)


def test_filter_prediction_by_score_drops_low_confidence_candidates_together():
    prediction = {
        "original_size_of_raw_data": np.asarray([4, 4, 4]),
        "pred_boxes": np.asarray([[0, 0, 2, 2, 0, 2], [2, 2, 4, 4, 2, 4]]),
        "pred_scores": np.asarray([0.9, 0.1]),
        "pred_labels": np.asarray([0, 1]),
        "pred_grade_probs": np.asarray([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]),
    }

    filtered = _filter_prediction_by_score(prediction, score_threshold=0.5)

    assert filtered["pred_boxes"].tolist() == [[0, 0, 2, 2, 0, 2]]
    assert filtered["pred_scores"].tolist() == [0.9]
    assert filtered["pred_labels"].tolist() == [0]
    assert filtered["pred_grade_probs"].tolist() == [[0.1, 0.2, 0.3, 0.4]]

    # A threshold of 0.0 (the default) must be a no-op, not "keep nothing >= 0".
    assert _filter_prediction_by_score(prediction, score_threshold=0.0) is prediction


def test_run_evaluation_writes_finite_metrics_and_rejects_case_mismatch(tmp_path):
    predictions = tmp_path / "predictions"
    labels = tmp_path / "labelsTs"
    output = tmp_path / "results"
    positive = np.zeros((4, 4, 4), dtype=np.uint8)
    positive[1:3, 1:3, 1:3] = 1
    _write_label(labels, "positive", positive)
    _write_label(labels, "benign", np.zeros((4, 4, 4), dtype=np.uint8))
    _write_prediction(predictions, "positive", [[1, 1, 3, 3, 1, 3]], [0.9], [0])
    _write_prediction(predictions, "benign", [], [], [])

    row = run_evaluation(
        predictions,
        labels,
        ["benign", "positive"],
        output,
        "Task900_PICAI_TINY",
        "RetinaUNetV001_D3V001_3d",
        0,
        "test",
    )

    assert row["num_cases"] == 2
    assert row["num_lesions"] == 1
    assert row["picai_score"] == pytest.approx(1.0)
    assert row["detection_score_threshold"] == pytest.approx(0.0)
    with (output / "metrics.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["detection_score_threshold"] == "0.0"
    assert (output / "picai_metrics.json").is_file()

    _write_prediction(predictions, "unexpected", [], [], [])
    with pytest.raises(ValueError, match="do not match"):
        run_evaluation(
            predictions,
            labels,
            ["benign", "positive"],
            output,
            "Task900_PICAI_TINY",
            "RetinaUNetV001_D3V001_3d",
            0,
            "test",
        )


def test_run_evaluation_detection_score_threshold_drops_low_confidence_candidate(tmp_path):
    predictions = tmp_path / "predictions"
    labels = tmp_path / "labelsTs"
    output = tmp_path / "results"
    positive = np.zeros((4, 4, 4), dtype=np.uint8)
    positive[1:3, 1:3, 1:3] = 1
    _write_label(labels, "positive", positive)
    _write_label(labels, "benign", np.zeros((4, 4, 4), dtype=np.uint8))
    # A confident true-positive box plus a low-confidence spurious box that would
    # otherwise be painted into the benign case's detection map.
    _write_prediction(predictions, "positive", [[1, 1, 3, 3, 1, 3]], [0.9], [0])
    _write_prediction(predictions, "benign", [[0, 0, 1, 1, 0, 1]], [0.05], [0])

    row = run_evaluation(
        predictions,
        labels,
        ["benign", "positive"],
        output,
        "Task900_PICAI_TINY",
        "RetinaUNetV001_D3V001_3d",
        0,
        "test",
        detection_score_threshold=0.5,
    )

    assert row["detection_score_threshold"] == pytest.approx(0.5)
    # The benign case's only candidate is below threshold, so it must not count
    # as a false positive.
    assert row["picai_score"] == pytest.approx(1.0)
    with (output / "metrics.csv").open(newline="") as file:
        csv_row = next(csv.DictReader(file))
    assert csv_row["detection_score_threshold"] == "0.5"


def test_run_evaluation_with_grade_predictions_populates_grade_metrics_end_to_end(tmp_path):
    """`run_eval.py`'s grade branch (Sec 154-165) is only reached when predictions carry
    `pred_grade_probs`; the other test in this file omits it, so the branch has never run
    end-to-end in tests."""
    predictions = tmp_path / "predictions"
    labels = tmp_path / "labelsTs"
    output = tmp_path / "results"
    positive = np.zeros((4, 4, 4), dtype=np.uint8)
    positive[1:3, 1:3, 1:3] = 1
    _write_label(labels, "positive", positive)
    (labels / "positive.json").write_text(json.dumps({
        "instances": {"1": {}},
        "grade_supervised": {"1": True},
        "grades": {"1": 3},
    }))
    _write_label(labels, "benign", np.zeros((4, 4, 4), dtype=np.uint8))
    (labels / "benign.json").write_text(json.dumps({"instances": {}, "grade_supervised": {}, "grades": {}}))
    _write_prediction(
        predictions, "positive", [[1, 1, 3, 3, 1, 3]], [0.9], [0],
        grade_probs=[[0.0, 1.0, 0.0, 0.0]],
    )
    _write_prediction(predictions, "benign", [], [], [], grade_probs=[])

    row = run_evaluation(
        predictions,
        labels,
        ["benign", "positive"],
        output,
        "Task900_PICAI_TINY",
        "RetinaUNetV001_D3V001_3d",
        0,
        "test",
        grade_score_threshold=0.5,
    )

    assert row["grade_matched_detections"] == 1
    assert row["grade_missed_supervised"] == 0
    assert row["grade_false_positives"] == 0
    assert row["grade_score_threshold"] == pytest.approx(0.5)
    assert row["grade_false_positives_per_case"] == pytest.approx(0.0)
    assert (output / "grade_metrics.json").is_file()
    with (output / "metrics.csv").open(newline="") as file:
        csv_row = next(csv.DictReader(file))
    assert math.isfinite(float(csv_row["grade_false_positives_per_case"]))
    assert math.isfinite(float(csv_row["grade_score_threshold"]))
