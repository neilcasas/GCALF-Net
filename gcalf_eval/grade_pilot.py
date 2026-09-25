"""Artifacts and statistical utilities for the grade-overfitting recovery pilot."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from gcalf_eval.grade_metrics import GRADE_VALUES, summarize_grade_matches


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patient_id(case_id):
    """PI-CAI case IDs encode patient and study as ``patient_study``."""
    return str(case_id).split("_", 1)[0]


def make_split_manifest(case_grades, output_path, seed=2026):
    """Deterministically stratify patient groups into 70/15/15 pilot roles.

    ``case_grades`` maps case ID to all supervised GGG2-5 labels in that
    case.  Each case belongs to one patient group; conflicting patient IDs
    are rejected rather than silently split across roles.
    """
    rng = np.random.RandomState(seed)
    groups = {}
    for case, grades in case_grades.items():
        groups.setdefault(patient_id(case), []).extend(int(grade) for grade in grades)
    for grade in GRADE_VALUES:
        if not any(grade in values for values in groups.values()):
            raise ValueError("Fold-0 training data has no GGG%d lesion" % grade)
    # Greedy rare-grade-first allocation minimizes global deviation from the
    # requested patient and per-grade proportions without splitting groups.
    assignments = {"fit": [], "selection": [], "calibration": []}
    totals = {name: Counter() for name in assignments}
    targets = {"fit": .70, "selection": .15, "calibration": .15}
    total_grade_support = Counter(grade for values in groups.values() for grade in values)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda key: (
        min(Counter(groups[key]).values()) if groups[key] else float("inf"), -len(groups[key])
    ))
    for group in shuffled:
        def allocation_error(selected):
            error = 0.0
            for name in assignments:
                size = len(assignments[name]) + (name == selected)
                error += 0.1 * (size / len(groups) - targets[name]) ** 2
                for grade in GRADE_VALUES:
                    count = totals[name][grade] + (groups[group].count(grade) if name == selected else 0)
                    error += (count / total_grade_support[grade] - targets[name]) ** 2
            return error
        name = min((allocation_error(name), name) for name in assignments)[1]
        assignments[name].append(group)
        totals[name].update(groups[group])
    missing = {name: [grade for grade in GRADE_VALUES if not totals[name][grade]] for name in assignments}
    if any(missing.values()):
        raise ValueError("Could not represent every grade in each partition: %s" % missing)
    manifest = {"seed": seed, "fractions": targets, "patients": assignments,
                "grade_support": {name: {str(g): totals[name][g] for g in GRADE_VALUES} for name in assignments}}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def temperature_scale(probabilities, temperature, epsilon=1e-7):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    probabilities = np.clip(np.asarray(probabilities, dtype=float), epsilon, 1.0)
    logits = np.log(probabilities) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(probabilities, grades):
    probabilities, grades = np.asarray(probabilities), np.asarray(grades, dtype=int) - 2
    if len(probabilities) != len(grades) or not len(grades):
        raise ValueError("Calibration requires aligned, nonempty matched lesions")
    candidates = np.exp(np.linspace(np.log(.05), np.log(20), 801))
    nll = [float(-np.log(temperature_scale(probabilities, t)[np.arange(len(grades)), grades]).mean()) for t in candidates]
    return float(candidates[int(np.argmin(nll))])


def bootstrap_metrics(grades, predictions, probabilities, patients, samples=2000, seed=2026):
    grades, predictions, probabilities, patients = map(np.asarray, (grades, predictions, probabilities, patients))
    unique = np.unique(patients)
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(samples):
        selected = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([np.flatnonzero(patients == patient) for patient in selected])
        rows.append(summarize_grade_matches(grades[index].tolist(), predictions[index].tolist(), 0, 0,
                                           matched_probabilities=probabilities[index].tolist()))
    return rows


def paired_bootstrap_auc_delta(grades, probabilities_a, probabilities_b, patients,
                               samples=2000, seed=2026):
    """Patient-cluster bootstrap for paired macro OvR AUROC differences (A - B)."""
    grades = np.asarray(grades, dtype=np.int64)
    probabilities_a = np.asarray(probabilities_a, dtype=np.float64)
    probabilities_b = np.asarray(probabilities_b, dtype=np.float64)
    patients = np.asarray(patients, dtype=str)
    if (probabilities_a.shape != probabilities_b.shape or probabilities_a.shape != (len(grades), 4)
            or patients.shape != grades.shape or not len(grades)):
        raise ValueError("Paired grades, four-class probabilities, and patients must align")
    if not np.isfinite(probabilities_a).all() or not np.isfinite(probabilities_b).all():
        raise ValueError("Paired probabilities must be finite")

    def macro_auc(selected_grades, selected_probabilities):
        predicted = selected_probabilities.argmax(axis=1) + GRADE_VALUES[0]
        summary = summarize_grade_matches(
            selected_grades.tolist(), predicted.tolist(), 0, 0,
            matched_probabilities=selected_probabilities.tolist(),
        )
        return summary["grade_macro_ovr_auroc"]

    point_a = macro_auc(grades, probabilities_a)
    point_b = macro_auc(grades, probabilities_b)
    if point_a is None or point_b is None:
        raise ValueError("Paired cohort must define macro OvR AUROC in both conditions")
    point_delta = float(point_a - point_b)

    unique = np.unique(patients)
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(samples):
        selected = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([np.flatnonzero(patients == patient) for patient in selected])
        auc_a = macro_auc(grades[index], probabilities_a[index])
        auc_b = macro_auc(grades[index], probabilities_b[index])
        if auc_a is not None and auc_b is not None:
            deltas.append(auc_a - auc_b)
    if not deltas:
        raise ValueError("No paired patient-bootstrap resample defined macro OvR AUROC")
    interval = np.percentile(np.asarray(deltas), [2.5, 97.5])
    return {
        "point_estimate": point_delta,
        "patient_bootstrap_95_ci": [float(interval[0]), float(interval[1])],
        "bootstrap_resamples": int(samples),
        "bootstrap_valid_resamples": len(deltas),
        "bootstrap_undefined_resamples": int(samples - len(deltas)),
        "bootstrap_seed": int(seed),
    }
