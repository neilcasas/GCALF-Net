"""Evaluate nnDetection box predictions with PI-CAI detection metrics."""

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from picai_eval import evaluate

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
)


def _box_coordinate_pairs(dimensions: int) -> Sequence[Tuple[int, int]]:
    if dimensions == 2:
        return ((0, 2), (1, 3))
    if dimensions == 3:
        return ((0, 2), (1, 3), (4, 5))
    raise ValueError(f"Only 2D and 3D boxes are supported, found {dimensions} dimensions")


def boxes_to_detection_map(prediction: Dict) -> np.ndarray:
    """Rasterize restored nnDetection boxes into a PI-CAI confidence map."""
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
    if not all(math.isfinite(float(row[field])) for field in ("picai_score", "auroc", "lesion_ap")):
        raise ValueError(f"PI-CAI metrics are not finite: {row}")

    metrics.save(output_dir / "picai_metrics.json")
    with (output_dir / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerow(row)
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
    args = parser.parse_args()

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
    )
    print(row)


if __name__ == "__main__":
    main()
