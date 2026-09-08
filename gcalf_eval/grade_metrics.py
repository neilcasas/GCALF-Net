"""Grade-supervised lesion matching and GGG2-5 summary metrics."""
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


GRADE_VALUES = (2, 3, 4, 5)


def _iou(box, boxes):
    lower = np.maximum(box[None, [0, 1, 4]], boxes[:, [0, 1, 4]])
    upper = np.minimum(box[None, [2, 3, 5]], boxes[:, [2, 3, 5]])
    intersection = np.maximum(upper - lower, 0).prod(axis=1)
    box_volume = np.maximum(box[[2, 3, 5]] - box[[0, 1, 4]], 0).prod()
    boxes_volume = np.maximum(boxes[:, [2, 3, 5]] - boxes[:, [0, 1, 4]], 0).prod(axis=1)
    return intersection / np.maximum(box_volume + boxes_volume - intersection, 1e-8)


def lesion_instances(label_path):
    """Load all lesion boxes in nnDetection's ``x1,y1,x2,y2,z1,z2`` order."""
    label_path = Path(label_path)
    with label_path.with_suffix("").with_suffix(".json").open() as file:
        metadata = json.load(file)
    labels = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
    boxes, grades, supervised_flags = [], [], []
    instance_ids = metadata.get("instances", {}).keys()
    if set(instance_ids) != set(metadata.get("grade_supervised", {})):
        raise ValueError(f"{label_path}: grade supervision metadata does not cover every instance")
    for instance_id in instance_ids:
        supervised = bool(metadata["grade_supervised"][instance_id])
        voxels = np.argwhere(labels == int(instance_id))
        if voxels.size == 0:
            raise ValueError(f"{label_path}: instance {instance_id} is absent from its mask")
        lower, upper = voxels.min(axis=0), voxels.max(axis=0) + 1
        # SimpleITK arrays are z-y-x; prediction boxes are x1,y1,x2,y2,z1,z2.
        boxes.append(np.array([lower[2], lower[1], upper[2], upper[1], lower[0], upper[0]], dtype=np.float32))
        grades.append(int(metadata["grades"][instance_id]) if supervised else -1)
        supervised_flags.append(supervised)
    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 6),
        np.asarray(grades, dtype=np.int64),
        np.asarray(supervised_flags, dtype=bool),
    )


def supervised_instances(label_path):
    """Load only grade-supervised lesion boxes, retained for public API compatibility."""
    boxes, grades, supervised = lesion_instances(label_path)
    return boxes[supervised], grades[supervised]


def match_grade_predictions(pred_boxes, pred_scores, pred_grade_probs, gt_boxes, gt_grades,
                            gt_supervised=None, iou_threshold=0.1, return_details=False):
    """Match against all lesions, then score grades only for supervised matches."""
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 6)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    pred_grade_probs = np.asarray(pred_grade_probs, dtype=np.float32).reshape(-1, 4)
    if len(pred_boxes) != len(pred_scores) or len(pred_boxes) != len(pred_grade_probs):
        raise ValueError("Grade predictions must align one-to-one with predicted boxes and scores")

    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 6)
    gt_grades = np.asarray(gt_grades, dtype=np.int64).reshape(-1)
    if len(gt_boxes) != len(gt_grades):
        raise ValueError("Ground-truth boxes and grades must align one-to-one")
    if gt_supervised is None:
        gt_supervised = np.ones(len(gt_boxes), dtype=bool)
    gt_supervised = np.asarray(gt_supervised, dtype=bool).reshape(-1)
    if len(gt_supervised) != len(gt_boxes):
        raise ValueError("Ground-truth supervision flags must align one-to-one with boxes")

    matched = np.zeros(len(gt_boxes), dtype=bool)
    true_grades, predicted_grades = [], []
    false_positives = 0
    matched_ungraded = 0
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
        if gt_supervised[target_index]:
            true_grades.append(int(gt_grades[target_index]))
            predicted_grades.append(int(pred_grade_probs[index].argmax()) + 2)
        else:
            matched_ungraded += 1
    misses = int((~matched & gt_supervised).sum())
    if return_details:
        return true_grades, predicted_grades, misses, false_positives, matched_ungraded
    return true_grades, predicted_grades, misses, false_positives


def summarize_grade_matches(true_grades, predicted_grades, misses, false_positives, matched_ungraded=0):
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
        "grade_matched_ungraded_lesions": int(matched_ungraded),
        "grade_missed_supervised": int(misses),
        "grade_false_positives": int(false_positives),
        "grade_weighted_f1": weighted_f1,
        "grade_per_class_sensitivity": {f"GGG{grade}": float(recall[grade - 2]) for grade in GRADE_VALUES},
    }
