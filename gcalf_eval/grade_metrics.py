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
    """Load lesion boxes in nnDetection's native array-axis pair order.

    SimpleITK exposes volumes as ``z, y, x`` arrays.  nnDetection's box
    transforms retain that spatial-axis order and pack it as
    ``axis0_low, axis1_low, axis0_high, axis1_high, axis2_low, axis2_high``.
    Do not reverse it to physical ``x, y, z`` order: restored predictions use
    this native order too.
    """
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
        boxes.append(np.array([lower[0], lower[1], upper[0], upper[1], lower[2], upper[2]], dtype=np.float32))
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
                            gt_supervised=None, iou_threshold=0.1, score_threshold=0.0,
                            return_details=False, return_probabilities=False):
    """Match against all lesions, then score grades only for supervised matches.

    ``grade_false_positives`` counts unmatched *detections* at greedy, score-ordered,
    IoU >= ``iou_threshold`` matching -- including duplicate boxes on a lesion that a
    higher-scoring box already matched. It is not a grade-classification error count.
    There is no per-case cap on this count: nothing upstream deduplicates overlapping
    boxes into one candidate the way ``boxes_to_detection_map`` does for
    ``picai_score``/``lesion_ap``, so it can be much larger than the number of lesions.
    ``score_threshold`` (default ``0.0``, i.e. today's behaviour) drops predictions
    below it before matching, giving an explicit, recorded operating point.
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 6)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    pred_grade_probs = np.asarray(pred_grade_probs, dtype=np.float32).reshape(-1, 4)
    if len(pred_boxes) != len(pred_scores) or len(pred_boxes) != len(pred_grade_probs):
        raise ValueError("Grade predictions must align one-to-one with predicted boxes and scores")

    keep = pred_scores >= score_threshold
    pred_boxes, pred_scores, pred_grade_probs = pred_boxes[keep], pred_scores[keep], pred_grade_probs[keep]

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
    true_grades, predicted_grades, matched_probabilities = [], [], []
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
            matched_probabilities.append(pred_grade_probs[index].tolist())
        else:
            matched_ungraded += 1
    misses = int((~matched & gt_supervised).sum())
    if return_details:
        if return_probabilities:
            return (true_grades, predicted_grades, matched_probabilities, misses,
                    false_positives, matched_ungraded)
        return true_grades, predicted_grades, misses, false_positives, matched_ungraded
    return true_grades, predicted_grades, misses, false_positives


def _binary_auroc(labels, scores):
    """Return rank-based binary AUROC, or ``None`` when it is undefined."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    # Average ranks for ties, without a scikit-learn runtime dependency.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def _calibration_summary(true_grades, probabilities, bins=10):
    """Multiclass confidence calibration over matched grade-supervised lesions."""
    if not len(true_grades):
        return {"grade_multiclass_brier": None, "grade_expected_calibration_error": None,
                "grade_mean_confidence": None, "grade_mean_confidence_correct": None,
                "grade_mean_confidence_incorrect": None}
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1, 4)
    truths = np.asarray(true_grades, dtype=int) - GRADE_VALUES[0]
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == truths
    one_hot = np.eye(len(GRADE_VALUES))[truths]
    brier = float(np.mean(np.square(probabilities - one_hot).sum(axis=1)))
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        mask = (confidence >= lower) & ((confidence < upper) if upper < 1.0 else (confidence <= upper))
        if mask.any():
            ece += float(mask.mean() * abs(confidence[mask].mean() - correct[mask].mean()))
    return {
        "grade_multiclass_brier": brier,
        "grade_expected_calibration_error": float(ece),
        "grade_mean_confidence": float(confidence.mean()),
        "grade_mean_confidence_correct": float(confidence[correct].mean()) if correct.any() else None,
        "grade_mean_confidence_incorrect": float(confidence[~correct].mean()) if (~correct).any() else None,
    }


def summarize_grade_matches(true_grades, predicted_grades, misses, false_positives, matched_ungraded=0,
                            num_cases=None, score_threshold=0.0, matched_probabilities=None):
    """Summarize matched grades. ``grade_false_positives_per_case`` and
    ``grade_score_threshold`` record the operating point ``grade_false_positives`` was
    computed at; ``num_cases`` must be the case count the caller matched over (omit
    only when the per-case rate is not needed)."""
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
    present = support > 0
    accuracy = float(tp.sum() / support.sum()) if support.sum() else 0.0
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0
    balanced_accuracy = float(recall[present].mean()) if present.any() else 0.0
    summary = {
        "grade_confusion_matrix": confusion.tolist(),
        "grade_matched_detections": int(len(true_grades)),
        "grade_matched_ungraded_lesions": int(matched_ungraded),
        "grade_missed_supervised": int(misses),
        "grade_false_positives": int(false_positives),
        "grade_weighted_f1": weighted_f1,
        "grade_accuracy": accuracy,
        "grade_macro_f1": macro_f1,
        "grade_balanced_accuracy": balanced_accuracy,
        "grade_per_class_sensitivity": {f"GGG{grade}": float(recall[grade - 2]) for grade in GRADE_VALUES},
        "grade_score_threshold": float(score_threshold),
    }
    if num_cases:
        summary["grade_false_positives_per_case"] = float(false_positives) / num_cases
    if matched_probabilities is not None:
        probabilities = np.asarray(matched_probabilities, dtype=float).reshape(-1, 4)
        if len(probabilities) != len(true_grades):
            raise ValueError("Matched grade probabilities must align with matched ground-truth grades")
        ovr_auroc = {
            f"GGG{grade}": _binary_auroc(np.asarray(true_grades) == grade, probabilities[:, grade - GRADE_VALUES[0]])
            for grade in GRADE_VALUES
        }
        valid_aurocs = [value for value in ovr_auroc.values() if value is not None]
        summary["grade_ovr_auroc"] = ovr_auroc
        summary["grade_macro_ovr_auroc"] = float(np.mean(valid_aurocs)) if valid_aurocs else None
        summary.update(_calibration_summary(true_grades, probabilities))
    return summary
