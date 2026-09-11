"""Evaluate nnDetection box predictions with PI-CAI detection metrics."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from picai_eval import evaluate
from scipy import ndimage

from gcalf_eval.grade_metrics import lesion_instances, match_grade_predictions, summarize_grade_matches
from nndet.io.load import load_pickle
from nndet.io.paths import get_task, get_training_dir


METRIC_FIELDS = (
    "task",
    "model",
    "fold",
    "split",
    "num_cases",
    "num_lesions",
    "picai_score",
    "auroc",
    "lesion_ap",
    "grade_matched_detections",
    "grade_matched_ungraded_lesions",
    "grade_missed_supervised",
    "grade_false_positives",
    "grade_false_positives_per_case",
    "grade_score_threshold",
    "grade_weighted_f1",
    "grade_accuracy",
    "grade_macro_f1",
    "grade_balanced_accuracy",
    "grade_macro_ovr_auroc",
    "grade_multiclass_brier",
    "grade_expected_calibration_error",
    "grade_mean_confidence",
)


def _box_coordinate_pairs(dimensions: int) -> Sequence[Tuple[int, int]]:
    if dimensions == 2:
        return ((0, 2), (1, 3))
    if dimensions == 3:
        return ((0, 2), (1, 3), (4, 5))
    raise ValueError(f"Only 2D and 3D boxes are supported, found {dimensions} dimensions")


def boxes_to_detection_map(prediction: Dict) -> np.ndarray:
    """Rasterize restored nnDetection boxes into a PI-CAI detection map.

    PI-CAI represents each lesion candidate as one connected component with a
    single confidence. Overlapping nnDetection boxes can otherwise form a
    component containing several scores, which PI-CAI correctly rejects as a
    softmax volume. Consolidating each connected candidate to its maximum
    score preserves the post-NMS confidence while producing the required
    detection-map representation.
    """
    shape = tuple(int(value) for value in np.asarray(prediction["original_size_of_raw_data"]).tolist())
    if not shape or any(value <= 0 for value in shape):
        raise ValueError(f"Invalid original_size_of_raw_data: {shape}")

    boxes = np.asarray(prediction["pred_boxes"], dtype=np.float64)
    scores = np.asarray(prediction["pred_scores"], dtype=np.float64).reshape(-1)
    labels = np.asarray(prediction["pred_labels"]).reshape(-1)
    coordinate_pairs = _box_coordinate_pairs(len(shape))
    if boxes.ndim != 2 or boxes.shape[1] != len(shape) * 2:
        raise ValueError(f"Expected boxes shaped (N, {len(shape) * 2}), found {boxes.shape}")
    if boxes.shape[0] != len(scores) or boxes.shape[0] != len(labels):
        raise ValueError("Prediction boxes, scores, and labels must have the same length")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("Prediction boxes and scores must be finite")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("Prediction scores must be probabilities in [0, 1]")

    detection_map = np.zeros(shape, dtype=np.float32)
    for box, score in zip(boxes, scores):
        slices = []
        for axis, (start_index, end_index) in enumerate(coordinate_pairs):
            lower, upper = box[start_index], box[end_index]
            if lower >= upper:
                raise ValueError(f"Invalid box bounds {box.tolist()}")
            start = max(0, int(math.floor(lower)))
            end = min(shape[axis], int(math.ceil(upper)))
            if end <= start:
                break
            slices.append(slice(start, end))
        else:
            region = tuple(slices)
            detection_map[region] = np.maximum(detection_map[region], float(score))

    # Convert overlapping/touching box regions into valid single-confidence
    # lesion candidates. ``label`` sees only positive voxels, so the zero
    # background remains untouched.
    components, num_components = ndimage.label(
        detection_map > 0, structure=np.ones((3,) * detection_map.ndim, dtype=np.uint8)
    )
    for component_id in range(1, num_components + 1):
        component = components == component_id
        detection_map[component] = detection_map[component].max()
    return detection_map


def _prediction_case_ids(prediction_dir: Path) -> List[str]:
    return sorted(path.stem.rsplit("_boxes", 1)[0] for path in prediction_dir.glob("*_boxes.pkl"))


def _load_ground_truth(label_path: Path) -> np.ndarray:
    if not label_path.is_file():
        raise ValueError(f"Ground-truth label is missing: {label_path}")
    return (sitk.GetArrayFromImage(sitk.ReadImage(str(label_path))) > 0).astype(np.int32)


def run_evaluation(
    prediction_dir: Path,
    ground_truth_dir: Path,
    case_ids: Iterable[str],
    output_dir: Path,
    task: str,
    model: str,
    fold: int,
    split: str,
    grade_score_threshold: float = 0.0,
) -> Dict[str, object]:
    """Evaluate exactly the requested cases and persist the M3 metric artifacts."""
    prediction_dir = Path(prediction_dir)
    ground_truth_dir = Path(ground_truth_dir)
    output_dir = Path(output_dir)
    expected_case_ids = sorted(case_ids)
    if not expected_case_ids:
        raise ValueError("Evaluation requires at least one case")
    if len(expected_case_ids) != len(set(expected_case_ids)):
        raise ValueError("Evaluation case identifiers must be unique")

    found_case_ids = _prediction_case_ids(prediction_dir)
    if found_case_ids != expected_case_ids:
        raise ValueError(
            f"Prediction cases do not match expected cases: expected {expected_case_ids}, found {found_case_ids}"
        )

    detection_maps = []
    ground_truth_masks = []
    grade_true, grade_predicted, grade_probabilities = [], [], []
    grade_misses = grade_false_positives = grade_matched_ungraded = 0
    has_grade_predictions = True
    for case_id in expected_case_ids:
        prediction = load_pickle(prediction_dir / f"{case_id}_boxes.pkl")
        detection_map = boxes_to_detection_map(prediction)
        ground_truth = _load_ground_truth(ground_truth_dir / f"{case_id}.nii.gz")
        if detection_map.shape != ground_truth.shape:
            raise ValueError(
                f"Prediction and ground truth shapes differ for {case_id}: "
                f"{detection_map.shape} != {ground_truth.shape}"
            )
        detection_maps.append(detection_map)
        ground_truth_masks.append(ground_truth)
        if "pred_grade_probs" not in prediction:
            has_grade_predictions = False
        else:
            gt_boxes, gt_grades, gt_supervised = lesion_instances(ground_truth_dir / f"{case_id}.nii.gz")
            truth, predicted, probabilities, misses, false_positives, matched_ungraded = match_grade_predictions(
                prediction["pred_boxes"], prediction["pred_scores"], prediction["pred_grade_probs"],
                gt_boxes, gt_grades, gt_supervised, score_threshold=grade_score_threshold,
                return_details=True, return_probabilities=True)
            grade_true.extend(truth)
            grade_predicted.extend(predicted)
            grade_probabilities.extend(probabilities)
            grade_misses += misses
            grade_false_positives += false_positives
            grade_matched_ungraded += matched_ungraded

    has_positive = any(mask.any() for mask in ground_truth_masks)
    has_negative = any(not mask.any() for mask in ground_truth_masks)
    if not has_positive or not has_negative:
        raise ValueError("M3 evaluation requires both positive and benign cases")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(
        y_det=detection_maps,
        y_true=ground_truth_masks,
        subject_list=expected_case_ids,
        num_parallel_calls=1,
    )
    row = {
        "task": task,
        "model": model,
        "fold": fold,
        "split": split,
        "num_cases": metrics.num_cases,
        "num_lesions": metrics.num_lesions,
        "picai_score": metrics.score,
        "auroc": metrics.auroc,
        "lesion_ap": metrics.AP,
    }
    if has_grade_predictions:
        row.update(summarize_grade_matches(
            grade_true, grade_predicted, grade_misses, grade_false_positives, grade_matched_ungraded,
            num_cases=len(expected_case_ids), score_threshold=grade_score_threshold,
            matched_probabilities=grade_probabilities,
        ))
    if not all(math.isfinite(float(row[field])) for field in ("picai_score", "auroc", "lesion_ap")):
        raise ValueError(f"PI-CAI metrics are not finite: {row}")

    metrics.save(output_dir / "picai_metrics.json")
    if has_grade_predictions:
        with (output_dir / "grade_metrics.json").open("w") as file:
            json.dump({key: value for key, value in row.items() if key.startswith("grade_")}, file, indent=2)
    with (output_dir / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        # Detailed grade diagnostics are intentionally persisted in
        # ``grade_metrics.json``. Keep this thesis table scalar-only so CSV
        # consumers do not receive nested confusion-matrix structures.
        writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})
    return row


def _resolve_paths(task: str, model: str, fold: int, split: str) -> Tuple[Path, Path, List[str], Path, str]:
    task_dir = get_task(task)
    task_name = task_dir.name
    model_root = Path(os.environ["det_models"]) / task_name / model
    training_dir = get_training_dir(model_root, fold)
    if split == "test":
        prediction_dir = training_dir / "test_predictions"
        ground_truth_dir = task_dir / "raw_splitted" / "labelsTs"
        case_ids = sorted(path.name[:-7] for path in ground_truth_dir.glob("*.nii.gz"))
    else:
        prediction_dir = training_dir / "val_predictions"
        ground_truth_dir = task_dir / "raw_splitted" / "labelsTr"
        splits = load_pickle(training_dir / "splits.pkl")
        try:
            case_ids = sorted(splits[fold]["val"])
        except (IndexError, KeyError) as error:
            raise ValueError(f"Validation split is missing for fold {fold}") from error
    return prediction_dir, ground_truth_dir, case_ids, training_dir / f"{split}_results" / "picai", task_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="nnDetection task identifier")
    parser.add_argument("--model", required=True, help="full nnDetection model identifier")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prediction-dir", type=Path,
                        help="Evaluate this restored prediction directory instead of the model's default split path.")
    parser.add_argument("--ground-truth-dir", type=Path,
                        help="Ground-truth label directory required with --prediction-dir.")
    parser.add_argument("--case-ids-file", type=Path,
                        help="One case identifier per line; required with --prediction-dir.")
    parser.add_argument(
        "--grade-score-threshold", type=float, default=0.0,
        help="Drop grade-matching predictions scoring below this before matching (default: 0.0, i.e. no threshold).",
    )
    args = parser.parse_args()

    direct_paths = (args.prediction_dir, args.ground_truth_dir, args.case_ids_file)
    if any(path is not None for path in direct_paths):
        if not all(path is not None for path in direct_paths):
            parser.error("--prediction-dir, --ground-truth-dir, and --case-ids-file must be supplied together")
        prediction_dir = args.prediction_dir
        ground_truth_dir = args.ground_truth_dir
        case_ids = [line.strip() for line in args.case_ids_file.read_text().splitlines() if line.strip()]
        default_output_dir = prediction_dir.parent / "picai_results"
        task_name = args.task
    else:
        prediction_dir, ground_truth_dir, case_ids, default_output_dir, task_name = _resolve_paths(
            args.task,
            args.model,
            args.fold,
            args.split,
        )
    row = run_evaluation(
        prediction_dir=prediction_dir,
        ground_truth_dir=ground_truth_dir,
        case_ids=case_ids,
        output_dir=args.output_dir or default_output_dir,
        task=task_name,
        model=args.model,
        fold=args.fold,
        split=args.split,
        grade_score_threshold=args.grade_score_threshold,
    )
    print(row)


if __name__ == "__main__":
    main()
