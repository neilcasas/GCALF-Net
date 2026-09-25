#!/usr/bin/env python3
"""Fit and evaluate the preregistered patient-disjoint ADR 0005 grade probe."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gcalf_eval.grade_metrics import GRADE_VALUES, summarize_grade_matches
from gcalf_eval.grade_pilot import (
    bootstrap_metrics,
    make_split_manifest,
    paired_bootstrap_auc_delta,
    patient_id,
)


SEED = 2026
BOOTSTRAP_SAMPLES = 2000


def validate_export(features, grades, supervised_mask, case_ids, instance_ids=None, anchor_ious=None):
    features = np.asarray(features)
    grades = np.asarray(grades)
    supervised_mask = np.asarray(supervised_mask, dtype=bool)
    case_ids = np.asarray(case_ids)
    rows = len(features)
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError(f"Expected feature matrix [rows, width], got {features.shape}")
    if any(array.ndim != 1 or len(array) != rows
           for array in (grades, supervised_mask, case_ids)):
        raise ValueError("Exported features, labels, supervision masks, and case IDs are misaligned")
    if instance_ids is not None and (np.asarray(instance_ids).ndim != 1 or len(instance_ids) != rows):
        raise ValueError("Exported instance IDs must align with feature rows")
    if anchor_ious is not None and (np.asarray(anchor_ious).ndim != 1 or len(anchor_ious) != rows):
        raise ValueError("Exported anchor IoUs must align with feature rows")
    if not np.isfinite(features).all():
        raise ValueError("Feature matrix contains NaN or infinite values")
    if not supervised_mask.any():
        raise ValueError("Feature export contains no supervised rows")
    invalid = sorted(set(grades[supervised_mask].astype(int).tolist()) - set(GRADE_VALUES))
    if invalid:
        raise ValueError(f"Supervised labels are outside GGG2-5: {invalid}")
    if np.any(case_ids[supervised_mask] == ""):
        raise ValueError("A supervised feature row has an empty case ID")
    if instance_ids is not None:
        instance_ids = np.asarray(instance_ids, dtype=np.int64)
        if np.any(instance_ids[supervised_mask] <= 0):
            raise ValueError("A supervised feature row has a non-positive instance ID")
    if anchor_ious is not None:
        anchor_ious = np.asarray(anchor_ious, dtype=np.float64)
        if not np.isfinite(anchor_ious).all() or np.any((anchor_ious < 0) | (anchor_ious > 1)):
            raise ValueError("Anchor IoUs must be finite values in [0, 1]")
    result = (features, grades.astype(np.int64, copy=False), supervised_mask, case_ids.astype(str, copy=False),
              instance_ids, anchor_ious)
    return result if instance_ids is not None or anchor_ious is not None else result[:4]


def aggregate_feature_readout(features, grades, supervised_mask, case_ids, instance_ids, anchor_ious,
                              readout="anchor"):
    """Reduce matched-positive anchor rows to one row per (case, instance)."""
    if readout not in {"anchor", "pooled"}:
        raise ValueError(f"Unknown feature readout: {readout}")
    if instance_ids is None or anchor_ious is None:
        raise ValueError("Lesion readouts require exported instance IDs and anchor IoUs")
    groups = {}
    for index, key in enumerate(zip(case_ids.astype(str), instance_ids.astype(int))):
        groups.setdefault(key, []).append(index)
    if not groups:
        raise ValueError("Feature export contains no lesion groups")

    grouped_features, grouped_grades, grouped_supervised, grouped_cases, grouped_instances = [], [], [], [], []
    for (case_id, instance_id), indices in groups.items():
        group_grades = np.unique(grades[indices])
        group_supervision = np.unique(supervised_mask[indices])
        if len(group_grades) != 1 or len(group_supervision) != 1:
            raise ValueError(f"Conflicting grade metadata for {case_id}:{instance_id}")
        if readout == "anchor":
            selected = indices[int(np.argmax(anchor_ious[indices]))]
            feature = features[selected]
        else:
            feature = features[indices].mean(axis=0)
        grouped_features.append(feature)
        grouped_grades.append(group_grades[0])
        grouped_supervised.append(group_supervision[0])
        grouped_cases.append(case_id)
        grouped_instances.append(instance_id)
    return (
        np.asarray(grouped_features, dtype=features.dtype),
        np.asarray(grouped_grades, dtype=np.int64),
        np.asarray(grouped_supervised, dtype=bool),
        np.asarray(grouped_cases, dtype=str),
        np.asarray(grouped_instances, dtype=np.int64),
    )


def assign_patient_roles(manifest, case_ids):
    patient_roles = {}
    for role, patients_in_role in manifest["patients"].items():
        for patient in patients_in_role:
            if patient in patient_roles:
                raise RuntimeError(f"Patient appears in multiple split roles: {patient}")
            patient_roles[patient] = role
    row_patients = np.asarray([patient_id(case_id) for case_id in case_ids], dtype=str)
    row_roles = np.asarray([patient_roles.get(patient) for patient in row_patients], dtype=object)
    if any(role is None for role in row_roles):
        raise RuntimeError("A supervised feature row is absent from the seed-2026 patient split")
    return patient_roles, row_patients, row_roles


def _macro_ovr_auc(grades, probabilities):
    predictions = np.asarray(probabilities).argmax(axis=1) + GRADE_VALUES[0]
    summary = summarize_grade_matches(
        np.asarray(grades).astype(int).tolist(),
        predictions.astype(int).tolist(),
        misses=0,
        false_positives=0,
        matched_probabilities=np.asarray(probabilities, dtype=float).tolist(),
    )
    return summary["grade_macro_ovr_auroc"]


def _fit_and_bootstrap(train_features, train_grades, eval_features, eval_grades, eval_patients,
                       seed=SEED, samples=BOOTSTRAP_SAMPLES):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed),
    )
    model.fit(train_features, train_grades)
    class_ids = model[-1].classes_.astype(int).tolist()
    if class_ids != list(GRADE_VALUES):
        raise ValueError(f"Probe fit classes do not cover GGG2-5: {class_ids}")
    probabilities = model.predict_proba(eval_features)
    point_estimate = _macro_ovr_auc(eval_grades, probabilities)
    if point_estimate is None:
        raise ValueError("Held-out data does not define macro OvR AUROC")

    rows = bootstrap_metrics(
        eval_grades,
        probabilities.argmax(axis=1) + GRADE_VALUES[0],
        probabilities,
        eval_patients,
        samples=samples,
        seed=seed,
    )
    bootstrap_values = [row["grade_macro_ovr_auroc"] for row in rows
                        if row["grade_macro_ovr_auroc"] is not None]
    if not bootstrap_values:
        raise ValueError("No patient bootstrap resample defined macro OvR AUROC")
    interval = np.percentile(np.asarray(bootstrap_values), [2.5, 97.5])
    return model, probabilities, float(point_estimate), [float(interval[0]), float(interval[1])], len(bootstrap_values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--readout", choices=("anchor", "pooled"), default="anchor")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite existing output directory: {args.output_dir}")
    if not args.feature_file.is_file() or not args.export_manifest.is_file():
        parser.error("--feature-file and --export-manifest must name existing files")

    with np.load(args.feature_file, allow_pickle=False) as archive:
        if "instance_ids" not in archive or "anchor_ious" not in archive:
            parser.error("Feature archive must include instance_ids and anchor_ious; re-run export_grade_features.py")
        exported = validate_export(
            archive["features"], archive["grades"], archive["supervised_mask"], archive["case_ids"],
            archive["instance_ids"], archive["anchor_ious"])
    anchor = aggregate_feature_readout(*exported, readout="anchor")
    pooled = aggregate_feature_readout(*exported, readout="pooled")
    for index in (1, 2, 3, 4):
        if not np.array_equal(anchor[index], pooled[index]):
            raise RuntimeError("Anchor and pooled readouts do not contain the same lesion rows")
    readouts = {"anchor": anchor[0], "pooled": pooled[0]}
    grades, supervised, case_ids = anchor[1:4]
    rows = np.flatnonzero(supervised)
    case_grades = {
        case_id: grades[(case_ids == case_id) & supervised].astype(int).tolist()
        for case_id in sorted(set(case_ids[supervised].tolist()))
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    split_path = args.output_dir / "split_seed2026.json"
    manifest = make_split_manifest(case_grades, split_path, seed=SEED)
    split_digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    _, row_patients, row_roles = assign_patient_roles(manifest, case_ids[rows])
    fit_mask = row_roles == "fit"
    evaluation_mask = np.isin(row_roles, ("selection", "calibration"))
    if np.any(fit_mask & evaluation_mask) or not fit_mask.any() or not evaluation_mask.any():
        raise RuntimeError("Seed-2026 probe split has an empty role or overlapping rows")
    fit_patients = set(row_patients[fit_mask].tolist())
    evaluation_patients = set(row_patients[evaluation_mask].tolist())
    overlap = fit_patients & evaluation_patients
    if overlap:
        raise RuntimeError(f"Patient leakage across fit/evaluation split: {len(overlap)} patient(s)")

    fit_grades = grades[rows][fit_mask]
    eval_grades = grades[rows][evaluation_mask]
    eval_patients = row_patients[evaluation_mask]
    if set(fit_grades.tolist()) != set(GRADE_VALUES):
        raise ValueError("The fit partition must contain supervised examples for GGG2-5")
    if set(eval_grades.tolist()) != set(GRADE_VALUES):
        raise ValueError("The held-out partitions must contain supervised examples for GGG2-5")

    readout_results = {}
    readout_probabilities = {}
    for readout_name, readout_features in readouts.items():
        fit_features = readout_features[rows][fit_mask]
        eval_features = readout_features[rows][evaluation_mask]
        _, probabilities, point_estimate, ci, valid_resamples = _fit_and_bootstrap(
            fit_features, fit_grades, eval_features, eval_grades, eval_patients,
            seed=SEED, samples=BOOTSTRAP_SAMPLES)
        readout_probabilities[readout_name] = probabilities
        readout_results[readout_name] = {
            "point_estimate": float(point_estimate),
            "patient_bootstrap_95_ci": ci,
            "bootstrap_resamples": BOOTSTRAP_SAMPLES,
            "bootstrap_valid_resamples": valid_resamples,
            "bootstrap_undefined_resamples": BOOTSTRAP_SAMPLES - valid_resamples,
            "bootstrap_seed": SEED,
        }
    paired_delta = paired_bootstrap_auc_delta(
        eval_grades,
        readout_probabilities["anchor"],
        readout_probabilities["pooled"],
        eval_patients,
        samples=BOOTSTRAP_SAMPLES,
        seed=SEED,
    )
    selected_result = readout_results[args.readout]
    selected_lower_bound = selected_result["patient_bootstrap_95_ci"][0]
    record = {
        "schema_version": 3,
        "protocol": "ADR 0005 patient-disjoint lesion-level multinomial logistic regression",
        "readout": args.readout,
        "readouts": readout_results,
        "paired_delta_anchor_minus_pooled": paired_delta,
        "checkpoint_identity": json.loads(args.export_manifest.read_text()),
        "split": {
            "seed": SEED,
            "fit_fraction": 0.70,
            "evaluation_roles": ["selection", "calibration"],
            "manifest": str(split_path),
            "manifest_sha256": split_digest,
            "fit_patients": len(fit_patients),
            "evaluation_patients": len(evaluation_patients),
            "fit_supervised_rows": int(len(fit_grades)),
            "evaluation_supervised_rows": int(len(eval_grades)),
            "lesion_count": int(len(anchor[0])),
            "patient_overlap": 0,
        },
        "metric": "macro OvR AUROC",
        "point_estimate": selected_result["point_estimate"],
        "patient_bootstrap_95_ci": selected_result["patient_bootstrap_95_ci"],
        "bootstrap_resamples": BOOTSTRAP_SAMPLES,
        "bootstrap_valid_resamples": selected_result["bootstrap_valid_resamples"],
        "bootstrap_undefined_resamples": selected_result["bootstrap_undefined_resamples"],
        "bootstrap_seed": SEED,
        "interpretation": "supports proceeding" if selected_lower_bound > 0.5 else "null expectation",
        "locked_build_unchanged": True,
    }
    (args.output_dir / "probe_result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in (
        "readouts", "paired_delta_anchor_minus_pooled", "interpretation")}, indent=2))


if __name__ == "__main__":
    main()
