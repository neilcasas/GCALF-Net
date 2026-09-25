#!/usr/bin/env python3
"""Compare raw and nnDetection-processed oracle-ROI grade features on PI-CAI folds."""

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gcalf_data.prepare_picai import (
    _case_image_paths,
    _index_patient_dirs,
    _whole_gland_mask_path,
    load_splits,
    patient_id as task_patient_id,
)
from gcalf_data.preprocessing import resample_fov_to_reference, resample_to_reference
from gcalf_eval.grade_metrics import GRADE_VALUES, summarize_grade_matches
from gcalf_eval.grade_pilot import paired_bootstrap_auc_delta, patient_id


FEATURE_NAMES = (
    "adc_p10",
    "adc_median",
    "hbv_p90_to_gland_median",
    "t2w_lesion_to_gland_median_ratio",
    "volume_mm3",
)
UNIT_RANGES = {
    "micro_mm2_per_s": (100.0, 10000.0),
    "mm2_per_s": (1e-5, 1e-2),
}


def _case_ids(images_dir):
    return sorted(path.name[:-12] for path in images_dir.glob("*_0000.nii.gz"))


def _load_pickle(path):
    with Path(path).open("rb") as file:
        return pickle.load(file)


def _adc_distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("ADC field of view contains no nonzero voxels")
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "p1": float(np.quantile(values, 0.01)),
        "median": float(np.median(values)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def _unit_class(median):
    matches = [name for name, (lower, upper) in UNIT_RANGES.items() if lower <= median <= upper]
    return matches[0] if len(matches) == 1 else "unrecognized"


def _source_fov_masks(images_by_patient, case_id, reference):
    """Recover each source scan's geometric coverage on the raw task grid.

    ``raw_splitted`` has already resampled ADC/HBV onto T2W and zero-filled
    uncovered voxels, so its voxel values cannot distinguish real zeros from
    out-of-FOV fill. Reprojecting an all-ones mask from each original MHA keeps
    those cases separate.
    """
    import SimpleITK as sitk

    patient_dir = images_by_patient.get(task_patient_id(case_id))
    if patient_dir is None:
        raise ValueError(f"No source image directory found for patient {task_patient_id(case_id)}")
    source_paths = _case_image_paths(patient_dir, case_id)
    return [
        sitk.GetArrayFromImage(resample_fov_to_reference(sitk.ReadImage(str(source_paths[key])), reference)) > 0
        for key in ("t2w", "adc", "hbv")
    ]


def audit_adc_units(task_dir, images_dirs):
    """Write per-case raw ADC quantiles and stop when recognized scales are mixed."""
    import SimpleITK as sitk

    images_dir = Path(task_dir) / "raw_splitted" / "imagesTr"
    images_by_patient = _index_patient_dirs(images_dirs)
    rows = []
    for case_id in _case_ids(images_dir):
        adc_path = images_dir / f"{case_id}_0001.nii.gz"
        reference = sitk.ReadImage(str(images_dir / f"{case_id}_0000.nii.gz"))
        adc = sitk.GetArrayFromImage(sitk.ReadImage(str(adc_path))).astype(np.float64)
        adc_fov = _source_fov_masks(images_by_patient, case_id, reference)[1]
        values = adc[adc_fov]
        distribution = _adc_distribution(values)
        unit_class = _unit_class(distribution["median"])
        rows.append({"case_id": case_id, **distribution, "unit_class": unit_class,
                     "looks_like_10^-6_mm2_per_s": unit_class == "micro_mm2_per_s"})
    classes = {row["unit_class"] for row in rows}
    if len(classes) > 1 and classes != {"unrecognized"}:
        mixed = sorted(row["case_id"] for row in rows if row["unit_class"] != "micro_mm2_per_s")
        return rows, mixed
    return rows, []


def _safe_ratio(numerator, denominator):
    if not np.isfinite(denominator) or abs(float(denominator)) < 1e-8:
        return float("nan")
    return float(numerator) / float(denominator)


def _feature_vector(data, lesion, gland, fovs, voxel_volume):
    lesion = np.asarray(lesion, dtype=bool)
    gland = np.asarray(gland, dtype=bool)
    adc_values = np.asarray(data[1])[lesion & fovs[1]]
    hbv_values = np.asarray(data[2])[lesion & fovs[2]]
    hbv_gland = np.asarray(data[2])[gland & fovs[2]]
    t2w_values = np.asarray(data[0])[lesion & fovs[0]]
    t2w_gland = np.asarray(data[0])[gland & fovs[0]]
    if len(hbv_values) and len(hbv_gland):
        hbv_ratio = _safe_ratio(np.quantile(hbv_values, 0.90), np.median(hbv_gland))
    else:
        hbv_ratio = float("nan")
    if len(t2w_values) and len(t2w_gland):
        t2w_ratio = _safe_ratio(np.median(t2w_values), np.median(t2w_gland))
    else:
        t2w_ratio = float("nan")
    return np.asarray([
        float(np.quantile(adc_values, 0.10)) if len(adc_values) else np.nan,
        float(np.median(adc_values)) if len(adc_values) else np.nan,
        hbv_ratio,
        t2w_ratio,
        float(lesion.sum()) * float(voxel_volume),
    ], dtype=np.float64)


def _mask_in_processed_frame(mask, properties, shape, plan):
    from scripts.backfill_anatomy_metadata import _mask_in_preprocessed_frame

    return _mask_in_preprocessed_frame(
        np.asarray(mask, dtype=np.uint8), properties, shape,
        transpose_forward=plan["transpose_forward"],
        target_spacing=plan.get("target_spacing_transposed"),
        resample_anisotropy_threshold=float(plan.get("resample_anisotropy_threshold", 3.0)),
        context="grade-signal audit mask",
    ) > 0


def _load_processed_case(images_dir, case_id):
    from scripts.backfill_anatomy_metadata import _load_preprocessed_case

    return _load_preprocessed_case(images_dir, case_id)


def collect_feature_rows(task_dir, labels_root, plan_path, displaced_report, images_dirs):
    import SimpleITK as sitk

    task_dir = Path(task_dir)
    labels_root = Path(labels_root)
    raw_images = task_dir / "raw_splitted" / "imagesTr"
    labels_dir = task_dir / "raw_splitted" / "labelsTr"
    processed_images = task_dir / "preprocessed" / "D3V001_3d" / "imagesTr"
    images_by_patient = _index_patient_dirs(images_dirs)
    plan = _load_pickle(plan_path)
    with Path(displaced_report).open() as file:
        report = json.load(file)
    displaced_cases = set(report.get("gate_excluded_cases", {}))

    rows = []
    for case_id in _case_ids(raw_images):
        image_paths = [raw_images / f"{case_id}_{index:04d}.nii.gz" for index in range(3)]
        raw_images_array = [sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
                            for path in image_paths]
        raw_label = sitk.GetArrayFromImage(sitk.ReadImage(str(labels_dir / f"{case_id}.nii.gz")))
        reference = sitk.ReadImage(str(image_paths[0]))
        gland_image = sitk.ReadImage(str(_whole_gland_mask_path(labels_root, case_id)))
        raw_gland = sitk.GetArrayFromImage(resample_to_reference(gland_image, reference, is_label=True)) > 0
        raw_fovs = _source_fov_masks(images_by_patient, case_id, reference)
        raw_spacing = reference.GetSpacing()
        raw_voxel_volume = float(np.prod(raw_spacing))

        processed_data, processed_seg = _load_processed_case(processed_images, case_id)
        properties = _load_pickle(processed_images / f"{case_id}.pkl")
        processed_shape = tuple(int(value) for value in processed_seg.shape)
        processed_gland = _mask_in_processed_frame(raw_gland, properties, processed_shape, plan)
        processed_fovs = [
            _mask_in_processed_frame(raw_fovs[index], properties, processed_shape, plan)
            for index in range(3)
        ]
        if processed_data.shape[0] != 3:
            raise ValueError(f"{case_id}: expected T2W/ADC/HBV in processed data, got {processed_data.shape}")
        spacing = np.asarray(properties["spacing_after_resampling"], dtype=np.float64)
        processed_voxel_volume = float(np.prod(spacing))
        with (labels_dir / f"{case_id}.json").open() as file:
            metadata = json.load(file)
        grades = metadata.get("grades", {})
        sources = metadata.get("grade_sources", {})
        supervised = metadata.get("grade_supervised", {})

        for raw_instance_id, is_supervised in supervised.items():
            if not is_supervised:
                continue
            instance_id = int(raw_instance_id)
            grade = int(grades.get(str(raw_instance_id), grades.get(instance_id)))
            source = sources.get(str(raw_instance_id), sources.get(instance_id))
            if grade not in GRADE_VALUES or source not in {"human_expert_mask", "audit_unifocal"}:
                raise ValueError(f"{case_id}:{instance_id}: invalid supervised grade/source {grade}/{source}")
            raw_lesion = raw_label == instance_id
            processed_lesion = processed_seg == instance_id
            if not raw_lesion.any() or not processed_lesion.any():
                raise ValueError(f"{case_id}:{instance_id}: lesion is missing in raw or processed labels")
            fov_fraction = float(raw_fovs[1][raw_lesion].mean())
            containment = float(processed_gland[processed_lesion].mean())
            aligned = fov_fraction >= 1.0 and case_id not in displaced_cases
            raw_features = _feature_vector(raw_images_array, raw_lesion, raw_gland, raw_fovs, raw_voxel_volume)
            processed_features = _feature_vector(
                processed_data, processed_lesion, processed_gland, processed_fovs, processed_voxel_volume)
            row = {
                "case_id": case_id,
                "patient_id": patient_id(case_id),
                "instance_id": instance_id,
                "grade": grade,
                "grade_source": source,
                "dwi_fov_fraction": fov_fraction,
                "lesion_gland_containment": containment,
                "displaced_case": case_id in displaced_cases,
                "aligned": aligned,
            }
            row.update({f"raw_{name}": float(value) for name, value in zip(FEATURE_NAMES, raw_features)})
            row.update({f"processed_{name}": float(value)
                        for name, value in zip(FEATURE_NAMES, processed_features)})
            rows.append(row)
    if not rows:
        raise ValueError("No grade-supervised lesions found for the feature audit")
    return rows


def _metric_summary(grades, probabilities):
    if not len(grades):
        return {"n_lesions": 0, "n_patients": 0, "macro_ovr_auroc": None,
                "per_grade_auroc": {f"GGG{grade}": None for grade in GRADE_VALUES},
                "balanced_accuracy": None}
    predicted = probabilities.argmax(axis=1) + GRADE_VALUES[0]
    metrics = summarize_grade_matches(
        grades.astype(int).tolist(), predicted.astype(int).tolist(), 0, 0,
        matched_probabilities=probabilities.tolist(),
    )
    return {
        "n_lesions": int(len(grades)),
        "n_patients": None,
        "macro_ovr_auroc": metrics["grade_macro_ovr_auroc"],
        "per_grade_auroc": metrics["grade_ovr_auroc"],
        "balanced_accuracy": metrics["grade_balanced_accuracy"],
    }


def _fit_oof(features, grades, case_ids, splits):
    probabilities = np.full((len(grades), len(GRADE_VALUES)), np.nan, dtype=np.float64)
    case_to_fold = {}
    for fold, split in enumerate(splits):
        for case_id in split["val"]:
            if case_id in case_to_fold:
                raise ValueError(f"Case occurs in validation more than once: {case_id}")
            case_to_fold[case_id] = fold
    missing = sorted(set(case_ids) - set(case_to_fold))
    if missing:
        raise ValueError(f"Task cases are absent from the predefined validation folds: {missing[:10]}")

    for fold, split in enumerate(splits):
        train_cases, validation_cases = set(split["train"]), set(split["val"])
        train_mask = np.isin(case_ids, list(train_cases))
        validation_mask = np.isin(case_ids, list(validation_cases))
        train_patients = {patient_id(case) for case in case_ids[train_mask]}
        validation_patients = {patient_id(case) for case in case_ids[validation_mask]}
        if train_patients & validation_patients:
            raise ValueError(f"Fold {fold} leaks a patient between training and validation")
        if set(grades[train_mask].tolist()) != set(GRADE_VALUES):
            raise ValueError(f"Fold {fold} training data does not contain all GGG2-5 labels")
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(solver="lbfgs", max_iter=2000, random_state=2026),
        )
        model.fit(features[train_mask], grades[train_mask])
        fold_probabilities = model.predict_proba(features[validation_mask])
        classes = model[-1].classes_.astype(int).tolist()
        if classes != list(GRADE_VALUES):
            raise ValueError(f"Fold {fold} model classes differ from GGG2-5: {classes}")
        probabilities[validation_mask] = fold_probabilities
    if not np.isfinite(probabilities).all():
        raise ValueError("The predefined folds did not produce finite out-of-fold probabilities for every lesion")
    return probabilities


def evaluate_subgroups(rows, splits, bootstrap_samples=2000):
    grades = np.asarray([row["grade"] for row in rows], dtype=np.int64)
    case_ids = np.asarray([row["case_id"] for row in rows], dtype=str)
    patients = np.asarray([row["patient_id"] for row in rows], dtype=str)
    raw = np.asarray([[row[f"raw_{name}"] for name in FEATURE_NAMES] for row in rows], dtype=np.float64)
    processed = np.asarray([[row[f"processed_{name}"] for name in FEATURE_NAMES]
                            for row in rows], dtype=np.float64)
    raw_probabilities = _fit_oof(raw, grades, case_ids, splits)
    processed_probabilities = _fit_oof(processed, grades, case_ids, splits)
    groups = {
        "all": np.ones(len(rows), dtype=bool),
        "aligned": np.asarray([row["aligned"] for row in rows], dtype=bool),
        "human_expert_mask": np.asarray([row["grade_source"] == "human_expert_mask" for row in rows]),
        "audit_unifocal": np.asarray([row["grade_source"] == "audit_unifocal" for row in rows]),
    }
    output = {}
    for name, mask in groups.items():
        if not mask.any():
            output[name] = {"status": "empty subgroup"}
            continue
        selected_grades = grades[mask]
        raw_selected = raw_probabilities[mask]
        processed_selected = processed_probabilities[mask]
        raw_metrics = _metric_summary(selected_grades, raw_selected)
        processed_metrics = _metric_summary(selected_grades, processed_selected)
        raw_metrics["n_patients"] = len(set(patients[mask].tolist()))
        processed_metrics["n_patients"] = raw_metrics["n_patients"]
        try:
            delta = paired_bootstrap_auc_delta(
                selected_grades, raw_selected, processed_selected, patients[mask],
                samples=bootstrap_samples, seed=2026)
        except ValueError as error:
            delta = {"status": str(error)}
        output[name] = {
            "n_lesions": int(mask.sum()),
            "n_patients": len(set(patients[mask].tolist())),
            "raw": raw_metrics,
            "processed": processed_metrics,
            "paired_delta_raw_minus_processed": delta,
        }
    return output


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, action="append", required=True,
                        help="original picai_public_images_fold directory; repeat once for each fold")
    parser.add_argument("--splits", type=Path, required=True,
                        help="picai_baseline/src/picai_baseline/splits/picai/splits.json")
    parser.add_argument("--displaced-report", type=Path, required=True,
                        help="ADR 0005 anatomy-containment JSON with gate_excluded_cases")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite existing output directory: {args.output_dir}")
    for path in (args.task_dir, args.labels_root, args.splits, args.displaced_report, *args.images_dir):
        if not path.exists():
            parser.error(f"input path does not exist: {path}")

    args.output_dir.mkdir(parents=True)
    unit_rows, mixed_cases = audit_adc_units(args.task_dir, args.images_dir)
    _write_csv(args.output_dir / "raw_adc_units_by_case.csv", unit_rows,
               ["case_id", "n", "min", "p1", "median", "p99", "max", "unit_class",
                "looks_like_10^-6_mm2_per_s"])
    unit_classes = sorted({row["unit_class"] for row in unit_rows})
    unit_report = {"case_count": len(unit_rows), "unit_classes": unit_classes,
                   "flagged_cases": [row["case_id"] for row in unit_rows if not row["looks_like_10^-6_mm2_per_s"]],
                   "mixed_scale_cases": mixed_cases}
    (args.output_dir / "raw_adc_units.json").write_text(json.dumps(unit_report, indent=2) + "\n")
    if mixed_cases:
        raise RuntimeError(
            "Raw ADC scale is mixed across cases; see raw_adc_units_by_case.csv and raw_adc_units.json"
        )

    plan_path = args.plan or (args.task_dir / "preprocessed" / "D3V001_3d.pkl")
    if not plan_path.is_file():
        parser.error(f"preprocessing plan does not exist: {plan_path}")
    rows = collect_feature_rows(
        args.task_dir, args.labels_root, plan_path, args.displaced_report, args.images_dir)
    splits = load_splits(args.splits)
    results = evaluate_subgroups(rows, splits, bootstrap_samples=args.bootstrap_samples)
    metadata = {
        "schema_version": 1,
        "protocol": "oracle-ROI linear first-order feature probe; five predefined patient-disjoint folds",
        "interpretation": "A null result is not a ceiling on bpMRI grade information.",
        "features": list(FEATURE_NAMES),
        "bootstrap": {"method": "paired patient-cluster bootstrap", "samples": args.bootstrap_samples,
                      "seed": 2026},
        "raw_adc_units": unit_report,
        "lesion_count": len(rows),
        "subgroups": results,
    }
    _write_csv(args.output_dir / "grade_signal_features.csv", rows, list(rows[0]))
    (args.output_dir / "grade_signal_results.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"lesion_count": len(rows), "subgroups": results}, indent=2))


if __name__ == "__main__":
    main()
