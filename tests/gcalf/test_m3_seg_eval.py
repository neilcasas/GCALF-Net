import json
import pickle

import numpy as np
import pytest

picai_eval = pytest.importorskip("picai_eval")
sitk = pytest.importorskip("SimpleITK")

from gcalf_eval.run_eval import run_evaluation
from gcalf_eval.seg_metrics import match_and_score_segmentation, summarize_segmentation_matches


def _write_instance_label(directory, case_id, array, instances):
    """`array` carries integer instance ids (0 = background); `instances` maps
    id -> (grade_supervised, grade) exactly as `grade_metrics._load_instances` expects."""
    directory.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(np.asarray(array, dtype=np.uint8)), str(directory / f"{case_id}.nii.gz"))
    metadata = {"instances": {}, "grade_supervised": {}, "grades": {}}
    for instance_id, (supervised, grade) in instances.items():
        metadata["instances"][str(instance_id)] = {}
        metadata["grade_supervised"][str(instance_id)] = supervised
        metadata["grades"][str(instance_id)] = grade
    (directory / f"{case_id}.json").write_text(json.dumps(metadata))


def _write_prediction(directory, case_id, boxes, scores, labels, shape=(4, 4, 4)):
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
    with (directory / f"{case_id}_boxes.pkl").open("wb") as file:
        pickle.dump(prediction, file)


def _write_seg_prediction(directory, case_id, array):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{case_id}_seg.pkl").open("wb") as file:
        pickle.dump({"pred_seg": np.asarray(array, dtype=np.uint8)}, file)


def test_match_and_score_segmentation_gives_perfect_dice_for_a_perfect_prediction(tmp_path):
    label = np.zeros((4, 4, 4), dtype=np.uint8)
    label[1:3, 1:3, 1:3] = 1
    _write_instance_label(tmp_path, "case", label, {1: (True, 3)})

    predicted_seg = np.zeros((4, 4, 4), dtype=np.uint8)
    predicted_seg[1:3, 1:3, 1:3] = 1

    lesion_dice, matched, missed, false_positives = match_and_score_segmentation(
        pred_boxes=[[1, 1, 3, 3, 1, 3]], pred_scores=[0.9], predicted_seg=predicted_seg,
        label_path=tmp_path / "case.nii.gz",
    )

    assert lesion_dice == pytest.approx([1.0])
    assert matched == 1
    assert missed == 0
    assert false_positives == 0


def test_match_and_score_segmentation_gives_a_known_value_for_a_partial_prediction(tmp_path):
    label = np.zeros((4, 4, 4), dtype=np.uint8)
    label[1:3, 1:3, 1:3] = 1  # 8 voxels
    _write_instance_label(tmp_path, "case", label, {1: (True, 3)})

    # Predicts only half the lesion's voxels (z in [1:2] instead of [1:3]):
    # tp=4, fn=4, fp=0 within the lesion's own bounding box -> dice = 2*4/(2*4+0+4) = 2/3.
    predicted_seg = np.zeros((4, 4, 4), dtype=np.uint8)
    predicted_seg[1:2, 1:3, 1:3] = 1

    lesion_dice, matched, missed, false_positives = match_and_score_segmentation(
        pred_boxes=[[1, 1, 3, 3, 1, 3]], pred_scores=[0.9], predicted_seg=predicted_seg,
        label_path=tmp_path / "case.nii.gz",
    )

    assert lesion_dice == pytest.approx([2.0 / 3.0])
    assert matched == 1


def test_match_and_score_segmentation_excludes_missed_lesions_from_the_dice_list(tmp_path):
    label = np.zeros((4, 4, 4), dtype=np.uint8)
    label[1:3, 1:3, 1:3] = 1
    _write_instance_label(tmp_path, "case", label, {1: (True, 3)})
    predicted_seg = np.zeros((4, 4, 4), dtype=np.uint8)  # no foreground predicted anywhere

    lesion_dice, matched, missed, false_positives = match_and_score_segmentation(
        pred_boxes=[], pred_scores=[], predicted_seg=predicted_seg,
        label_path=tmp_path / "case.nii.gz",
    )

    assert lesion_dice == []
    assert matched == 0
    assert missed == 1
    assert false_positives == 0


def test_match_and_score_segmentation_counts_an_unmatched_detection_as_a_false_positive(tmp_path):
    label = np.zeros((4, 4, 4), dtype=np.uint8)  # benign: no lesions
    _write_instance_label(tmp_path, "case", label, {})
    predicted_seg = np.zeros((4, 4, 4), dtype=np.uint8)
    predicted_seg[0:1, 0:1, 0:1] = 1

    lesion_dice, matched, missed, false_positives = match_and_score_segmentation(
        pred_boxes=[[0, 0, 1, 1, 0, 1]], pred_scores=[0.9], predicted_seg=predicted_seg,
        label_path=tmp_path / "case.nii.gz",
    )

    assert lesion_dice == []
    assert matched == 0
    assert missed == 0
    assert false_positives == 1


def test_summarize_segmentation_matches_reports_none_when_nothing_matched():
    summary = summarize_segmentation_matches([], matched=0, missed=1, false_positives=0)
    assert summary["seg_lesion_mean_dice"] is None
    assert summary["seg_missed_lesions"] == 1


def test_run_evaluation_wires_segmentation_metrics_into_csv_and_json(tmp_path):
    predictions = tmp_path / "predictions"
    labels = tmp_path / "labelsTs"
    output = tmp_path / "results"

    positive_label = np.zeros((4, 4, 4), dtype=np.uint8)
    positive_label[1:3, 1:3, 1:3] = 1
    _write_instance_label(labels, "positive", positive_label, {1: (True, 3)})
    _write_instance_label(labels, "benign", np.zeros((4, 4, 4), dtype=np.uint8), {})

    _write_prediction(predictions, "positive", [[1, 1, 3, 3, 1, 3]], [0.9], [0])
    _write_prediction(predictions, "benign", [], [], [])

    predicted_seg_positive = np.zeros((4, 4, 4), dtype=np.uint8)
    predicted_seg_positive[1:3, 1:3, 1:3] = 1
    _write_seg_prediction(predictions, "positive", predicted_seg_positive)
    _write_seg_prediction(predictions, "benign", np.zeros((4, 4, 4), dtype=np.uint8))

    row = run_evaluation(
        predictions, labels, ["benign", "positive"], output,
        "Task900_PICAI_TINY", "RetinaUNetV001_D3V001_3d", 0, "test",
    )

    assert row["seg_lesion_mean_dice"] == pytest.approx(1.0)
    assert row["seg_matched_detections"] == 1
    assert row["seg_missed_lesions"] == 0
    assert row["seg_false_positives"] == 0
    assert "seg_status" not in row
    assert (output / "seg_metrics.json").is_file()
    with (output / "seg_metrics.json").open() as file:
        seg_metrics = json.load(file)
    assert seg_metrics["seg_lesion_mean_dice"] == pytest.approx(1.0)


def test_run_evaluation_reports_out_of_scope_status_when_segmentation_was_not_predicted(tmp_path):
    """No `{case_id}_seg.pkl` files are written here -- mirrors every prediction
    directory produced before `inference_kwargs.do_seg=true` is set."""
    predictions = tmp_path / "predictions"
    labels = tmp_path / "labelsTs"
    output = tmp_path / "results"

    positive_label = np.zeros((4, 4, 4), dtype=np.uint8)
    positive_label[1:3, 1:3, 1:3] = 1
    _write_instance_label(labels, "positive", positive_label, {1: (True, 3)})
    _write_instance_label(labels, "benign", np.zeros((4, 4, 4), dtype=np.uint8), {})
    _write_prediction(predictions, "positive", [[1, 1, 3, 3, 1, 3]], [0.9], [0])
    _write_prediction(predictions, "benign", [], [], [])

    row = run_evaluation(
        predictions, labels, ["benign", "positive"], output,
        "Task900_PICAI_TINY", "RetinaUNetV001_D3V001_3d", 0, "test",
    )

    assert row["seg_status"] == "out_of_scope_segmentation_head_not_trained"
    assert not (output / "seg_metrics.json").is_file()
