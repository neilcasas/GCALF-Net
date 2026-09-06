"""Validate the raw PI-CAI csPCa nnDetection task before planning."""

import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from gcalf_data.build_labels import GRADE_NAMES, VALID_GRADES, marksheet_summary
from gcalf_data.audit_crop_retention import load_exclusions, summarize, validate_audit
from gcalf_data.prepare_picai import load_splits
from gcalf_data.preprocessing import TARGET_FOV_MM, fov_mm_to_inplane_size

EXPECTED_MODALITIES = {"0": "T2W", "1": "ADC", "2": "HBV"}
EXPECTED_NUM_SLICES = 32


def case_ids(images_dir: Path) -> List[str]:
    return sorted(path.name[:-12] for path in images_dir.glob("*_0000.nii.gz"))


def _assert_matching_geometry(reference, image, case_id: str, label: str) -> None:
    """Compare two sitk.Image geometries the same way nndet's own
    `_check_itk_params` does: exact size, approximate spacing/origin/direction.
    NIfTI stores the affine as float32, so real (gantry-tilted, non-axis-aligned)
    direction matrices pick up ~1e-6-scale round-trip noise on every extra
    read/write (e.g. nnunet2nndet's own re-encode) -- numpy's very tight default
    allclose tolerance (atol=1e-8) flags that noise as a mismatch, so use a
    tolerance sized for float32 precision instead."""
    import numpy as np

    assert reference.GetSize() == image.GetSize(), f"{case_id}: {label} size differs"
    assert np.allclose(reference.GetSpacing(), image.GetSpacing(), atol=1e-4), f"{case_id}: {label} spacing differs"
    assert np.allclose(reference.GetOrigin(), image.GetOrigin(), atol=1e-4), f"{case_id}: {label} origin differs"
    assert np.allclose(reference.GetDirection(), image.GetDirection(), atol=1e-4), f"{case_id}: {label} direction differs"


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


def validate_task(task_dir: Path) -> Tuple[Counter, int, int, List[str]]:
    """Return grade counts, ungraded-positive count, positive-case count, and IDs."""
    try:
        import numpy as np
        import SimpleITK as sitk
    except ImportError as error:
        raise RuntimeError("sanity_checks requires numpy and SimpleITK from the M0 environment") from error

    with (task_dir / "dataset.json").open() as file:
        dataset = json.load(file)
    assert dataset["labels"] == {"0": "csPCa"}, f"Expected a single csPCa foreground class, found {dataset['labels']}"
    assert dataset["modalities"] == EXPECTED_MODALITIES

    images_dir = task_dir / "raw_splitted" / "imagesTr"
    labels_dir = task_dir / "raw_splitted" / "labelsTr"
    cases = case_ids(images_dir)
    assert cases, f"No T2W images found in {images_dir}"

    grade_counts: Counter = Counter()
    ungraded_positive_count = 0
    positive_case_count = 0
    for case_id in cases:
        images = [sitk.ReadImage(str(images_dir / f"{case_id}_{modality:04d}.nii.gz")) for modality in range(3)]
        for image in images[1:]:
            _assert_matching_geometry(images[0], image, case_id, "modality")
        spacing = images[0].GetSpacing()  # (x, y, z)
        expected_size_y, expected_size_x = fov_mm_to_inplane_size((spacing[1], spacing[0]), TARGET_FOV_MM)
        size = images[0].GetSize()  # (x, y, z)
        assert size[0] == expected_size_x and size[1] == expected_size_y, (
            f"{case_id}: expected {expected_size_x}x{expected_size_y} in-plane "
            f"({TARGET_FOV_MM:.0f}mm FOV at spacing {spacing[0]:.3f}x{spacing[1]:.3f}mm), found {size[0]}x{size[1]}"
        )
        assert size[2] == EXPECTED_NUM_SLICES, f"{case_id}: expected {EXPECTED_NUM_SLICES} slices, found {size[2]}"

        label_path = labels_dir / f"{case_id}.nii.gz"
        label = sitk.ReadImage(str(label_path))
        _assert_matching_geometry(images[0], label, case_id, "label")

        with (labels_dir / f"{case_id}.json").open() as file:
            metadata = json.load(file)
        instances = metadata["instances"]
        grades = metadata.get("grades", {})
        grade_sources = metadata.get("grade_sources", {})
        grade_supervised = metadata.get("grade_supervised", {})

        ids_in_mask = set(np.unique(sitk.GetArrayFromImage(label)).tolist()) - {0}
        assert ids_in_mask == {int(instance_id) for instance_id in instances}, f"Instance IDs disagree for {case_id}"
        assert all(class_id == 0 for class_id in instances.values()), f"Non-zero detection class for {case_id}"
        assert set(grade_supervised) == set(instances), f"grade_supervised keys disagree with instances for {case_id}"
        if instances:
            positive_case_count += 1

        for instance_id, supervised in grade_supervised.items():
            if supervised:
                assert instance_id in grades and grades[instance_id] in VALID_GRADES, (
                    f"{case_id} instance {instance_id}: grade_supervised but missing/invalid grade"
                )
                assert instance_id in grade_sources, f"{case_id} instance {instance_id}: missing grade_source"
                grade_counts[grades[instance_id]] += 1
            else:
                assert instance_id not in grades, (
                    f"{case_id} instance {instance_id}: grade present despite grade_supervised=false"
                )
                ungraded_positive_count += 1
    return grade_counts, ungraded_positive_count, positive_case_count, cases


def marksheet_positive_case_ids(marksheet_path: Path) -> set:
    """Return cases whose marksheet pathology establishes a csPCa lesion."""
    positive = set()
    with marksheet_path.open(newline="") as file:
        for row in csv.DictReader(file):
            if int(row["case_ISUP"]) >= 2:
                positive.add(f"{row['patient_id']}_{row['study_id']}")
    return positive


def validate_marksheet_positive_cases(
    task_dir: Path,
    marksheet_path: Path,
    exclusions: set,
    crop_records: List[Dict[str, object]] = None,
) -> None:
    """Prevent a cropped-away positive from silently becoming a benign case."""
    labels_dir = task_dir / "raw_splitted" / "labelsTr"
    known_positive = marksheet_positive_case_ids(marksheet_path)
    if crop_records is not None:
        audited_positive = {
            str(record["case_id"])
            for record in crop_records
            if int(record["source_voxels"]) > 0
        }
        if audited_positive != known_positive:
            raise AssertionError(
                "Crop-retention source-positive cases disagree with marksheet-positive cases: "
                f"audit_only={sorted(audited_positive - known_positive)} "
                f"marksheet_only={sorted(known_positive - audited_positive)}"
            )
    raw_case_ids = case_ids(task_dir / "raw_splitted" / "imagesTr")
    missing = known_positive - set(raw_case_ids) - exclusions
    if missing:
        raise AssertionError(f"Marksheet-positive cases missing without an exclusion: {sorted(missing)}")
    for case_id in sorted(known_positive & set(raw_case_ids)):
        with (labels_dir / f"{case_id}.json").open() as file:
            instances = json.load(file)["instances"]
        if not instances:
            raise AssertionError(f"{case_id}: marksheet-positive case has no retained detection instance")


def validate_held_out_grades(task_dir: Path, splits: List[Dict[str, List[str]]]) -> None:
    """Every retained validation fold needs each GGG2-5 supervision class."""
    labels_dir = task_dir / "raw_splitted" / "labelsTr"
    required = set(VALID_GRADES)
    for fold, split in enumerate(splits):
        held_out_grades = set()
        for case_id in split["val"]:
            with (labels_dir / f"{case_id}.json").open() as file:
                metadata = json.load(file)
            held_out_grades.update(int(grade) for grade in metadata.get("grades", {}).values())
        missing = required - held_out_grades
        assert not missing, f"Fold {fold} has no grade-supervised instances for {sorted(missing)}"


def validate_plan(plan_path: Path) -> dict:
    with plan_path.open("rb") as file:
        plan = pickle.load(file)
    architecture = plan["architecture"]
    assert architecture["in_channels"] == 3
    assert architecture["classifier_classes"] == 1
    assert len(architecture["conv_kernels"]) == 5, "GCALF-Net requires a five-level planned encoder"
    assert len(plan["patch_size"]) == 3
    assert len(plan["target_spacing"]) == 3
    assert set(plan["normalization_schemes"].values()) == {"nonCT"}
    assert len(plan["use_mask_for_norm"]) == 3
    return plan


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value):
    return value.tolist() if hasattr(value, "tolist") else value


def write_manifest(task_dir: Path, plan_path: Path, cases: List[str]) -> Path:
    """Capture the resolved preprocessing and five-level plan beside the task."""
    plan = validate_plan(plan_path)
    architecture = plan["architecture"]
    manifest = {
        "task": task_dir.name,
        "retained_case_count": len(cases),
        "case_ids_sha256": hashlib.sha256("\n".join(cases).encode()).hexdigest(),
        "dataset_json_sha256": _sha256(task_dir / "dataset.json"),
        "splits_sha256": _sha256(task_dir / "splits.json"),
        "crop_retention_sha256": _sha256(task_dir / "crop_retention.csv"),
        "excluded_cases_sha256": _sha256(task_dir / "excluded_cases.json"),
        "target_spacing": _json_value(plan["target_spacing"]),
        "patch_size": _json_value(plan["patch_size"]),
        "normalization_schemes": _json_value(plan["normalization_schemes"]),
        "use_mask_for_norm": _json_value(plan["use_mask_for_norm"]),
        "architecture": {
            key: _json_value(architecture[key])
            for key in ("in_channels", "classifier_classes", "conv_kernels", "strides")
        },
    }
    output_path = task_dir / "dataset_manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output_path


def write_report(
    report_path: Path,
    grade_counts: Counter,
    ungraded_positive_count: int,
    positive_case_count: int,
    marksheet_path: Path,
    splits: List[Dict[str, List[str]]],
    crop_records: List[Dict[str, object]],
    exclusions: set,
) -> None:
    marksheet = marksheet_summary(marksheet_path)
    retention = summarize(crop_records)
    source_positive_count = sum(int(record["source_voxels"]) > 0 for record in crop_records)
    lines = [
        "# PI-CAI M1 data report",
        "",
        f"- Cohort: {positive_case_count} retained positive cases across "
        f"{sum(len(split['val']) for split in splits)} retained cases train csPCa detection (single foreground class).",
        f"- Source-positive cases audited: {source_positive_count}; excluded after target-independent crop: "
        f"{len(exclusions)} ({', '.join(sorted(exclusions)) or 'none'}).",
        "- Grade supervision: only grade-resolved lesions (human_expert masks directly, plus "
        "audit-recovered Pooch25 unifocal cases) train the separate GGG2-5 grade head; this is "
        f"{sum(grade_counts.values())} of {sum(grade_counts.values()) + ungraded_positive_count} "
        "positive lesions -- never the full detection-training cohort.",
        "- ISUP 0 and 1 cases remain zero-instance negatives; no benign or GGG1 foreground class is created.",
        "- Modalities: T2W, ADC, HBV/high-b DWI (`_0000`, `_0001`, `_0002`).",
        "",
        "## Grade-supervised instance counts",
        "",
    ]
    lines.extend(f"- {GRADE_NAMES[grade]}: {grade_counts[grade]}" for grade in sorted(VALID_GRADES))
    lines.append(f"- Ungraded positive instances (Pooch25, not audit-recoverable): {ungraded_positive_count}")
    lines.extend(["", "## Marksheet case ISUP counts", ""])
    lines.extend(
        f"- ISUP {grade}: {count}" for grade, count in sorted(marksheet["case_isup"].items())
    )
    lines.extend(["", "## Crop-retention audit", ""])
    lines.extend(f"- {status}: {retention[status]}" for status in sorted(retention))
    exceptions = [
        record
        for record in crop_records
        if int(record["source_voxels"]) > 0 and record["status"] != "fully_retained"
    ]
    lines.extend(["", "| Case | Status | Source voxels | Final voxels | Source components | Final components |", "|---|---|---:|---:|---:|---:|"])
    lines.extend(
        "| {case_id} | {status} | {source_voxels} | {final_voxels} | {source_components} | {final_components} |".format(
            **record
        )
        for record in exceptions
    )
    lines.extend(["", "## Official fold sizes", ""])
    lines.extend(
        f"- Fold {index}: train={len(split['train'])}, val={len(split['val'])}"
        for index, split in enumerate(splits)
    )
    lines.extend([
        "",
        "## Cohort limitations",
        "",
        "- Grade supervision never covers the full retained detection-training cohort. State this explicitly "
        "wherever weighted F1 or the confusion matrix is reported.",
        "- GGG4 and GGG5 are small even before folding; report per-grade counts and bootstrap CIs "
        "everywhere (Phase 6).",
        "- Multi-component or multi-marksheet-lesion Pooch25 cases that the audit could not resolve "
        "remain detection-positive but grade-unsupervised. This boundary is deliberate, not a gap to "
        "close under schedule pressure.",
    ])
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--marksheet", type=Path, required=True)
    parser.add_argument("--plan-path", type=Path)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()

    grade_counts, ungraded_positive_count, positive_case_count, cases = validate_task(args.task_dir)
    splits = load_splits(args.task_dir / "splits.json")
    validate_splits(splits, cases)
    raw_case_ids = set(cases)
    split_case_ids = {case_id for split in splits for key in ("train", "val") for case_id in split[key]}
    crop_records = validate_audit(args.task_dir, raw_case_ids, split_case_ids)
    exclusions = load_exclusions(args.task_dir)
    validate_marksheet_positive_cases(args.task_dir, args.marksheet, exclusions, crop_records)
    validate_held_out_grades(args.task_dir, splits)
    if args.plan_path:
        write_manifest(args.task_dir, args.plan_path, cases)
    if args.report_path:
        write_report(
            args.report_path,
            grade_counts,
            ungraded_positive_count,
            positive_case_count,
            args.marksheet,
            splits,
            crop_records,
            exclusions,
        )
    print("All PI-CAI M1 sanity checks passed.")


if __name__ == "__main__":
    main()
