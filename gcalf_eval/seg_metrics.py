"""Lesion-matched segmentation Dice (PHASE_6_evaluation.md Sec 6.3).

The only Dice elsewhere in this codebase (`nndet.evaluator.seg.SegmentationEvaluator`)
is a whole-volume, epoch-level training monitor. This module answers the distinct
question Sec 6.3 asks for: Dice against lesion delineations, for matched positive
detections, at inference time on a held-out split.
"""
import numpy as np

from gcalf_eval.grade_metrics import _greedy_match_boxes, _load_instances


def _crop_to_box(volume, box):
    """Crop `volume` to a box in nnDetection's native axis-pair order:
    (axis0_low, axis1_low, axis0_high, axis1_high, axis2_low, axis2_high) --
    see `grade_metrics.lesion_instances`. Clamped to the volume's own bounds.
    """
    lower = np.maximum(np.floor([box[0], box[1], box[4]]).astype(int), 0)
    upper = np.minimum(np.ceil([box[2], box[3], box[5]]).astype(int), volume.shape)
    return volume[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]


def _dice(prediction_mask, ground_truth_mask):
    """Binary Dice; defined as 1.0 when both masks are empty (nothing to
    segment there, nothing predicted there)."""
    intersection = float(np.logical_and(prediction_mask, ground_truth_mask).sum())
    total = float(prediction_mask.sum() + ground_truth_mask.sum())
    if total == 0.0:
        return 1.0
    return 2.0 * intersection / total


def match_and_score_segmentation(pred_boxes, pred_scores, predicted_seg, label_path,
                                  iou_threshold=0.1, score_threshold=0.0):
    """Per-lesion Dice for matched positive detections.

    Reuses the same greedy, score-ordered IoU matching as
    `grade_metrics.match_grade_predictions` (`_greedy_match_boxes`), so "matched
    positive detection" means identically the same thing for the grade and
    segmentation endpoints. Unlike the grade endpoint, every ground-truth
    lesion is in scope here, not only grade-supervised ones -- the
    segmentation head is trained foreground/background only (Sec 6.3), with no
    grade-supervision distinction.

    Each lesion's Dice is computed within its own ground-truth bounding box
    only: a false-positive segmentation elsewhere in the volume belongs to
    overall case-level segmentation quality, not to this specific lesion's
    score, so it must not corrupt it.

    Args:
        pred_boxes: predicted boxes, nnDetection's native box order [N, 6].
        pred_scores: predicted detection confidence [N].
        predicted_seg: predicted hard-label segmentation volume (as saved by
            `SegmentationEnsembler`, argmaxed foreground/background), same
            shape and space as the ground-truth label image.
        label_path: path to the ground-truth instance-labelled volume.
        iou_threshold: minimum box IoU to count as a match (matches the
            grade-matching default).
        score_threshold: drop predictions scoring below this before matching,
            an explicit, recorded operating point independent of the
            detection-map and grade-matching thresholds.

    Returns:
        lesion_dice: Dice per matched lesion (one entry per matched
            ground-truth instance; misses are excluded, never scored as 0).
        matched: number of matched ground-truth lesions scored.
        missed: ground-truth lesions with no matching detection.
        false_positives: unmatched detections (score >= score_threshold).
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 6)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    if len(pred_boxes) != len(pred_scores):
        raise ValueError("Predicted boxes and scores must align one-to-one")

    keep = pred_scores >= score_threshold
    pred_boxes, pred_scores = pred_boxes[keep], pred_scores[keep]

    instance_ids, gt_boxes, _, _, label_image = _load_instances(label_path)
    predicted_seg = np.asarray(predicted_seg)
    if predicted_seg.shape != label_image.shape:
        raise ValueError(
            f"{label_path}: predicted segmentation shape {predicted_seg.shape} "
            f"does not match ground truth shape {label_image.shape}"
        )

    matched_gt_index, matched_gt_mask = _greedy_match_boxes(pred_boxes, pred_scores, gt_boxes, iou_threshold)

    lesion_dice = []
    for target_index in np.flatnonzero(matched_gt_mask):
        box = gt_boxes[target_index]
        instance_id = instance_ids[target_index]
        ground_truth_mask = _crop_to_box(label_image, box) == instance_id
        prediction_mask = _crop_to_box(predicted_seg, box) > 0
        lesion_dice.append(_dice(prediction_mask, ground_truth_mask))

    false_positives = int((matched_gt_index < 0).sum())
    missed = int((~matched_gt_mask).sum())
    return lesion_dice, int(matched_gt_mask.sum()), missed, false_positives


def summarize_segmentation_matches(lesion_dice, matched, missed, false_positives,
                                    num_cases=None, score_threshold=0.0):
    """Summarize matched-lesion Dice. Mirrors `grade_metrics.summarize_grade_matches`:
    misses and false positives are reported beside the Dice mean, never folded
    into it, and the operating point (`score_threshold`) is always recorded
    alongside the count it was computed at."""
    summary = {
        "seg_lesion_mean_dice": float(np.mean(lesion_dice)) if lesion_dice else None,
        "seg_matched_detections": int(matched),
        "seg_missed_lesions": int(missed),
        "seg_false_positives": int(false_positives),
        "seg_score_threshold": float(score_threshold),
    }
    if num_cases:
        summary["seg_false_positives_per_case"] = float(false_positives) / num_cases
    return summary
