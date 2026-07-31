"""Validate the raw PI-CAI GGG2--5 nnDetection task before planning."""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from gcalf_data.build_labels import GGG_LABELS, marksheet_summary
from gcalf_data.prepare_picai import load_splits


EXPECTED_MODALITIES = {"0": "T2W", "1": "ADC", "2": "HBV"}


def case_ids(images_dir: Path) -> List[str]:
    return sorted(path.name[:-12] for path in images_dir.glob("*_0000.nii.gz"))


def validate_splits(splits: List[Dict[str, List[str]]], expected_cases: Iterable[str]) -> None:
    expected_cases = set(expected_cases)
    validation_cases = set()
    validation_patients = set()
    for fold, split in enumerate(splits):
        train = set(split["train"])
        validation = set(split["val"])
        assert train.isdisjoint(validation), f"Fold {fold} has overlapping train and validation cases"
        assert train | validation == expected_cases, f"Fold {fold} does not cover the raw task cases"
        train_patients = {case_id.split("_", 1)[0] for case_id in train}
        validation_patients_fold = {case_id.split("_", 1)[0] for case_id in validation}
        assert train_patients.isdisjoint(validation_patients_fold), f"Fold {fold} leaks a patient across splits"
        assert validation_cases.isdisjoint(validation), f"Case occurs in validation more than once: fold {fold}"
        assert validation_patients.isdisjoint(validation_patients_fold), (
            f"Patient occurs in validation more than once: fold {fold}"
        )
        validation_cases.update(validation)
        validation_patients.update(validation_patients_fold)
    assert validation_cases == expected_cases, "Validation folds do not cover every task case exactly once"


def validate_task(task_dir: Path) -> Tuple[Counter, List[str]]:
    try:
        import numpy as np
        import SimpleITK as sitk
    except ImportError as error:
        raise RuntimeError("sanity_checks requires numpy and SimpleITK from the M0 environment") from error

    with (task_dir / "dataset.json").open() as file:
        dataset = json.load(file)
    assert dataset["labels"] == {str(key - 1): value for key, value in GGG_LABELS.items()}
    assert dataset["modalities"] == EXPECTED_MODALITIES

    images_dir = task_dir / "raw_splitted" / "imagesTr"
    labels_dir = task_dir / "raw_splitted" / "labelsTr"
    cases = case_ids(images_dir)
    assert cases, f"No T2W images found in {images_dir}"
    counts = Counter()
    for case_id in cases:
        images = [sitk.ReadImage(str(images_dir / f"{case_id}_{modality:04d}.nii.gz")) for modality in range(3)]
        geometry = (images[0].GetSize(), images[0].GetSpacing(), images[0].GetOrigin(), images[0].GetDirection())
        for image in images[1:]:
            assert (image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection()) == geometry

        label_path = labels_dir / f"{case_id}.nii.gz"
        label = sitk.ReadImage(str(label_path))
        assert (label.GetSize(), label.GetSpacing(), label.GetOrigin(), label.GetDirection()) == geometry
        ids = set(np.unique(sitk.GetArrayFromImage(label)).tolist()) - {0}
        with (labels_dir / f"{case_id}.json").open() as file:
            instances = json.load(file)["instances"]
        assert ids == {int(instance_id) for instance_id in instances}, (
            f"Instance IDs disagree for {case_id}"
        )
        assert all(int(class_id) in range(4) for class_id in instances.values()), (
            f"Invalid GGG class for {case_id}"
        )
        counts.update(int(class_id) for class_id in instances.values())
    return counts, cases


def validate_plan(plan_path: Path) -> None:
    with plan_path.open("rb") as file:
        plan = pickle.load(file)
    architecture = plan["architecture"]
    assert architecture["in_channels"] == 3
    assert architecture["classifier_classes"] == 4


def write_report(
    report_path: Path,
    instance_counts: Counter,
    marksheet_path: Path,
    splits: List[Dict[str, List[str]]],
) -> None:
    marksheet = marksheet_summary(marksheet_path)
    lines = [
        "# PI-CAI M1 data report",
        "",
        "- Cohort: 1,295 cases with original granular human-expert csPCa masks.",
        "- Target: per-lesion GGG2--5; PI-CAI does not provide spatial GGG1 masks.",
        "- ISUP 0 and 1 cases remain zero-instance negatives; no benign foreground class is created.",
        "- Modalities: T2W, ADC, HBV/high-b DWI (`_0000`, `_0001`, `_0002`).",
        "",
        "## Instance counts",
        "",
    ]
    lines.extend(
        f"- {GGG_LABELS[class_id + 1]}: {instance_counts[class_id]}" for class_id in range(4)
    )
    lines.extend(["", "## Marksheet case ISUP counts", ""])
    lines.extend(
        f"- ISUP {grade}: {count}" for grade, count in sorted(marksheet["case_isup"].items())
    )
    lines.extend(["", "## Official fold sizes", ""])
    lines.extend(
        f"- Fold {index}: train={len(split['train'])}, val={len(split['val'])}"
        for index, split in enumerate(splits)
    )
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--marksheet", type=Path, required=True)
    parser.add_argument("--plan-path", type=Path)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()

    instance_counts, cases = validate_task(args.task_dir)
    splits = load_splits(args.task_dir / "splits.json")
    validate_splits(splits, cases)
    if args.plan_path:
        validate_plan(args.plan_path)
    if args.report_path:
        write_report(args.report_path, instance_counts, args.marksheet, splits)
    print("All PI-CAI M1 sanity checks passed.")


if __name__ == "__main__":
    main()
