"""Grade-supervised lesion matching and GGG2-5 summary metrics."""
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


GRADE_VALUES = (2, 3, 4, 5)


def _iou(box, boxes):
    lower = np.maximum(box[None, 0::2], boxes[:, 0::2])
    upper = np.minimum(box[None, 1::2], boxes[:, 1::2])
    intersection = np.maximum(upper - lower, 0).prod(axis=1)
    box_volume = np.maximum(box[1::2] - box[0::2], 0).prod()
    boxes_volume = np.maximum(boxes[:, 1::2] - boxes[:, 0::2], 0).prod(axis=1)
    return intersection / np.maximum(box_volume + boxes_volume - intersection, 1e-8)


def supervised_instances(label_path):
    """Load grade-supervised instance boxes in the prediction's z-y-x order."""
    label_path = Path(label_path)
    with label_path.with_suffix("").with_suffix(".json").open() as file:
        metadata = json.load(file)
    labels = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
    boxes, grades = [], []
    for instance_id, supervised in metadata.get("grade_supervised", {}).items():
        if not supervised:
            continue
        voxels = np.argwhere(labels == int(instance_id))
        if voxels.size == 0:
            raise ValueError(f"{label_path}: grade-supervised instance {instance_id} is absent from its mask")
        lower, upper = voxels.min(axis=0), voxels.max(axis=0) + 1
        boxes.append(np.array([lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]], dtype=np.float32))
        grades.append(int(metadata["grades"][instance_id]))
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 6), np.asarray(grades, dtype=np.int64)


def match_grade_predictions(pred_boxes, pred_scores, pred_grade_probs, gt_boxes, gt_grades, iou_threshold=0.1):
    """Greedily match score-ranked detections to grade-supervised lesions."""
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 6)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    pred_grade_probs = np.asarray(pred_grade_probs, dtype=np.float32).reshape(-1, 4)
    if len(pred_boxes) != len(pred_scores) or len(pred_boxes) != len(pred_grade_probs):
        raise ValueError("Grade predictions must align one-to-one with predicted boxes and scores")

    matched = np.zeros(len(gt_boxes), dtype=bool)
    true_grades, predicted_grades = [], []
    false_positives = 0
    for index in np.argsort(-pred_scores):
        available = np.flatnonzero(~matched)
        if not len(available):
            false_positives += 1
            continue
        overlaps = _iou(pred_boxes[index], gt_boxes[available])
        best = int(overlaps.argmax())
        if overlaps[best] < iou_threshold:
            false_positives += 1
            continue
        target_index = available[best]
        matched[target_index] = True
        true_grades.append(int(gt_grades[target_index]))
        predicted_grades.append(int(pred_grade_probs[index].argmax()) + 2)
    return true_grades, predicted_grades, int((~matched).sum()), false_positives


def summarize_grade_matches(true_grades, predicted_grades, misses, false_positives):
    confusion = np.zeros((4, 4), dtype=np.int64)
    for truth, prediction in zip(true_grades, predicted_grades):
        confusion[truth - 2, prediction - 2] += 1
    support = confusion.sum(axis=1)
    tp = np.diag(confusion).astype(float)
    predicted = confusion.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    weighted_f1 = float(np.average(f1, weights=support)) if support.sum() else 0.0
    return {
        "grade_confusion_matrix": confusion.tolist(),
        "grade_matched_detections": int(len(true_grades)),
        "grade_missed_supervised": int(misses),
        "grade_false_positives": int(false_positives),
        "grade_weighted_f1": weighted_f1,
        "grade_per_class_sensitivity": {f"GGG{grade}": float(recall[grade - 2]) for grade in GRADE_VALUES},
    }
