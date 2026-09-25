"""Harmonized IoU-matched FROC-style evaluation for GGG2--5 lesions."""

import argparse
import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from gcalf_eval.grade_metrics import GRADE_VALUES, _greedy_match_boxes, lesion_instances
from gcalf_eval.grade_pilot import patient_id


IOU_THRESHOLD = 0.5
TARGET_FP_PER_PATIENT = (0.5, 1.0)
OPERATING_POINT_RULE = (
    "last curve point with FP-rate <= target, following "
    "picai_eval Metrics.lesion_TPR_at_FPR; no interpolation"
)
DESCRIPTION = (
    "Harmonized FROC-style comparison using score-ordered IoU matching. "
    "This is not an exact PDHD-Net reproduction and is not classification recall."
)


def _grade_label(grade: int) -> str:
    return f"GGG{int(grade)}"


def _support_by_grade(support: Mapping) -> Dict[int, int]:
    """Normalize integer, numeric-string, or ``GGGk`` support keys."""
    if not isinstance(support, Mapping):
        raise TypeError("Grade support must be a mapping from grades to counts")
    normalized = {}
    for grade in GRADE_VALUES:
        value = None
        for key in (grade, str(grade), _grade_label(grade)):
            if key in support:
                value = support[key]
                break
        if value is None:
            value = 0
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(f"Support for {_grade_label(grade)} must be a non-negative integer")
        normalized[grade] = int(value)
    return normalized


def _explicit_support_grades(support: Mapping) -> List[int]:
    if not isinstance(support, Mapping):
        raise TypeError("Grade support must be a mapping from grades to counts")
    grades = []
    for key in support:
        for grade in GRADE_VALUES:
            if key in (grade, str(grade), _grade_label(grade)) and grade not in grades:
                grades.append(grade)
    return grades


def _prediction_arrays(pred_boxes, pred_scores, pred_grade_probs, gt_boxes, gt_grades, gt_supervised):
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 6)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    pred_grade_probs = np.asarray(pred_grade_probs, dtype=np.float32).reshape(-1, len(GRADE_VALUES))
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 6)
    gt_grades = np.asarray(gt_grades, dtype=np.int64).reshape(-1)
    gt_supervised = np.asarray(gt_supervised, dtype=bool).reshape(-1)

    if len(pred_boxes) != len(pred_scores) or len(pred_boxes) != len(pred_grade_probs):
        raise ValueError("Prediction boxes, scores, and grade probabilities must align")
    if len(gt_boxes) != len(gt_grades) or len(gt_boxes) != len(gt_supervised):
        raise ValueError("Ground-truth boxes, grades, and supervision flags must align")
    if not np.isfinite(pred_boxes).all() or not np.isfinite(pred_scores).all():
        raise ValueError("Prediction boxes and scores must be finite")
    if not np.isfinite(pred_grade_probs).all():
        raise ValueError("Grade probabilities must be finite")
    if np.any(pred_scores < 0.0) or np.any(pred_scores > 1.0):
        raise ValueError("Detection confidences must be in [0, 1]")
    if np.any(pred_grade_probs < 0.0) or np.any(pred_grade_probs > 1.0):
        raise ValueError("Grade probabilities must be in [0, 1]")
    return pred_boxes, pred_scores, pred_grade_probs, gt_boxes, gt_grades, gt_supervised


def harmonized_froc_events(
    pred_boxes,
    pred_scores,
    pred_grade_probs,
    gt_boxes,
    gt_grades,
    gt_supervised,
    iou_threshold: float = IOU_THRESHOLD,
) -> Tuple[List[dict], Dict[int, List[dict]]]:
    """Return generic and grade-specific per-candidate FROC events for one case.

    Generic matching uses every ground-truth lesion. A matched unsupervised lesion
    is absorbed rather than counted as either a TP or an FP. Grade-specific matching
    uses only supervised lesions of the requested grade and scores each candidate by
    ``pred_scores * P(GGG=k)``.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be in (0, 1]")
    (
        pred_boxes,
        pred_scores,
        pred_grade_probs,
        gt_boxes,
        gt_grades,
        gt_supervised,
    ) = _prediction_arrays(
        pred_boxes, pred_scores, pred_grade_probs, gt_boxes, gt_grades, gt_supervised
    )

    generic_matches, _ = _greedy_match_boxes(pred_boxes, pred_scores, gt_boxes, iou_threshold)
    generic_events = []
    for index, target_index in enumerate(generic_matches):
        grade = None
        if target_index >= 0 and gt_supervised[target_index]:
            grade = int(gt_grades[target_index])
        generic_events.append({
            "score": float(pred_scores[index]),
            "grade": grade,
            "fp": bool(target_index < 0),
        })

    grade_events = {}
    for grade in GRADE_VALUES:
        grade_mask = gt_supervised & (gt_grades == grade)
        class_scores = pred_scores * pred_grade_probs[:, grade - GRADE_VALUES[0]]
        grade_matches, _ = _greedy_match_boxes(
            pred_boxes, class_scores, gt_boxes[grade_mask], iou_threshold
        )
        grade_events[grade] = [
            {
                "score": float(class_scores[index]),
                "tp": bool(target_index >= 0),
                "fp": bool(target_index < 0),
            }
            for index, target_index in enumerate(grade_matches)
        ]
    return generic_events, grade_events


def froc_curve(events: Sequence[dict], num_patients: int, support: Mapping):
    """Sweep tied scores together and return FP/patient and per-grade sensitivity.

    Generic events identify a TP with their ``grade`` field. Grade-specific events
    identify a TP with ``tp`` and are passed with a one-grade support mapping.
    """
    if num_patients <= 0:
        raise ValueError("FROC requires at least one patient")
    explicit_grades = _explicit_support_grades(support)
    support = _support_by_grade(support)
    events = list(events)
    if any("tp" in event for event in events):
        if any("tp" not in event for event in events):
            raise ValueError("FROC events must use one event format consistently")
        active_grades = explicit_grades if len(explicit_grades) == 1 else list(support)
        if len(active_grades) != 1 and any(event.get("grade") is None for event in events):
            raise ValueError("Grade-specific events require support for exactly one grade")
    else:
        active_grades = explicit_grades if explicit_grades else list(support)

    events_by_score = {}
    for event in events:
        score = float(event["score"])
        if not np.isfinite(score):
            raise ValueError("FROC event scores must be finite")
        if "fp" not in event:
            raise ValueError("FROC events must contain an fp field")
        events_by_score.setdefault(score, []).append(event)

    fp = 0
    tp_by_grade = {grade: 0 for grade in active_grades}
    fp_rates = [0.0]
    sensitivities = {grade: [0.0] for grade in active_grades}
    for score in sorted(events_by_score, reverse=True):
        for event in events_by_score[score]:
            is_fp = bool(event["fp"])
            if is_fp:
                fp += 1
            if "tp" in event:
                if bool(event["tp"]) and is_fp:
                    raise ValueError("An FROC event cannot be both TP and FP")
                if bool(event["tp"]):
                    grade = event.get("grade", active_grades[0])
                    if grade not in tp_by_grade:
                        raise ValueError(f"Event grade {grade} is not in the supplied support")
                    tp_by_grade[grade] += 1
            elif not is_fp and event.get("grade") in tp_by_grade:
                tp_by_grade[event["grade"]] += 1
        fp_rates.append(fp / float(num_patients))
        for grade in active_grades:
            sensitivities[grade].append(
                float(tp_by_grade[grade]) / support[grade] if support[grade] else 0.0
            )
    return fp_rates, sensitivities


def sensitivity_at_fp(fp_rates: Sequence[float], sensitivities: Sequence, target: float):
    """Read sensitivity conservatively at an FP/patient target.

    This follows ``picai_eval.metrics.Metrics.lesion_TPR_at_FPR``: use the
    sensitivity at the last curve point whose FP rate is at most the target,
    rather than interpolating between points or exceeding the FP budget.
    """
    if target < 0.0 or not np.isfinite(target):
        raise ValueError("Target FP rate must be finite and non-negative")
    fp_rates = list(fp_rates)
    sensitivities = list(sensitivities)
    if len(fp_rates) != len(sensitivities) or not fp_rates:
        raise ValueError("FP rates and sensitivities must be aligned and non-empty")
    eligible = [index for index, fp_rate in enumerate(fp_rates) if fp_rate <= target]
    if not eligible:
        return 0.0
    value = sensitivities[eligible[-1]]
    return None if value is None else float(value)


def _curve_json(fp_rates, sensitivities):
    return {
        "fp_per_patient": [float(value) for value in fp_rates],
        "sensitivity_by_grade": {
            _grade_label(grade): [None if value is None else float(value) for value in values]
            for grade, values in sensitivities.items()
        },
    }


def _events_for_grade(grade_events: Mapping, grade: int) -> Sequence[dict]:
    for key in (grade, str(grade), _grade_label(grade)):
        if key in grade_events:
            return grade_events[key]
    return []


def summarize_harmonized_froc(
    generic_events: Sequence[dict],
    grade_events: Mapping,
    support: Mapping,
    num_patients: int,
    num_cases: int,
    target_fp_rates: Sequence[float] = TARGET_FP_PER_PATIENT,
    iou_threshold: float = IOU_THRESHOLD,
):
    """Build the JSON-ready harmonized FROC summary and complete curves."""
    support = _support_by_grade(support)
    if num_patients <= 0 or num_cases <= 0:
        raise ValueError("Harmonized FROC requires positive patient and case counts")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be in (0, 1]")
    if any(target < 0.0 or not np.isfinite(target) for target in target_fp_rates):
        raise ValueError("Target FP rates must be finite and non-negative")

    generic_fp, generic_sensitivity = froc_curve(generic_events, num_patients, support)
    grade_curves = {}
    records = []
    for grade in GRADE_VALUES:
        events = _events_for_grade(grade_events, grade)
        class_fp, class_sensitivity = froc_curve(events, num_patients, {grade: support[grade]})
        grade_values = class_sensitivity[grade]
        grade_curves[_grade_label(grade)] = {
            "fp_per_patient": [float(value) for value in class_fp],
            "sensitivity": [None if value is None else float(value) for value in grade_values],
        }
        for target in target_fp_rates:
            records.append({
                "analysis": "generic_detection_stratified_by_gt_grade",
                "grade": grade,
                "iou_threshold": float(iou_threshold),
                "target_fp_per_patient": float(target),
                "sensitivity": (
                    None if support[grade] == 0 else sensitivity_at_fp(
                        generic_fp, generic_sensitivity[grade], target
                    )
                ),
                "grade_lesion_support": support[grade],
                "patient_count": int(num_patients),
            })
            records.append({
                "analysis": "grade_specific_candidate_froc",
                "grade": grade,
                "iou_threshold": float(iou_threshold),
                "target_fp_per_patient": float(target),
                "sensitivity": (
                    None if support[grade] == 0 else sensitivity_at_fp(class_fp, grade_values, target)
                ),
                "grade_lesion_support": support[grade],
                "patient_count": int(num_patients),
            })

    return {
        "schema_version": 1,
        "description": DESCRIPTION,
        "pdhd_net_note": (
            "The comparison is harmonized rather than an exact PDHD-Net reproduction; "
            "per-grade values are detection sensitivities, not classification recall."
        ),
        "matcher": "score-ordered IoU matching in nnDetection native array-axis order",
        "iou_threshold": float(iou_threshold),
        "operating_point_rule": OPERATING_POINT_RULE,
        "target_fp_per_patient": [float(value) for value in target_fp_rates],
        "num_patients": int(num_patients),
        "num_cases": int(num_cases),
        "gt_lesions_per_grade": {
            _grade_label(grade): support[grade] for grade in GRADE_VALUES
        },
        "num_candidates": int(len(generic_events)),
        "num_grade_candidates": int(sum(len(_events_for_grade(grade_events, grade)) for grade in GRADE_VALUES)),
        "curves": {
            "generic_detection_stratified_by_gt_grade": _curve_json(
                generic_fp, generic_sensitivity
            ),
            "grade_specific_candidate_froc": grade_curves,
        },
        "records": records,
    }


def make_events_payload(
    generic_events: Sequence[dict],
    grade_events: Mapping,
    support: Mapping,
    patients: Iterable[str],
    num_cases: int,
):
    """Create the fold event artifact consumed by the pooling CLI."""
    patients = sorted(set(str(value) for value in patients))
    support = _support_by_grade(support)
    if not patients:
        raise ValueError("FROC event artifacts require at least one patient")
    return {
        "schema_version": 1,
        "num_cases": int(num_cases),
        "num_patients": len(patients),
        "patients": patients,
        "support": {_grade_label(grade): support[grade] for grade in GRADE_VALUES},
        "generic_events": list(generic_events),
        "grade_events": {
            _grade_label(grade): list(_events_for_grade(grade_events, grade))
            for grade in GRADE_VALUES
        },
    }


def evaluate_prediction_dir(
    prediction_dir: Path,
    labels_dir: Path,
    case_ids: Iterable[str],
    iou_threshold: float = IOU_THRESHOLD,
    target_fp_rates: Sequence[float] = TARGET_FP_PER_PATIENT,
):
    """Evaluate saved box predictions and return ``(summary, events_payload)``."""
    prediction_dir = Path(prediction_dir)
    labels_dir = Path(labels_dir)
    case_ids = sorted(case_ids)
    if not case_ids:
        raise ValueError("FROC requires at least one case")
    found = sorted(path.name[:-10] for path in prediction_dir.glob("*_boxes.pkl"))
    if found != case_ids:
        raise ValueError(f"Prediction cases differ from requested fold: expected {case_ids}, found {found}")

    generic_events = []
    grade_events = {grade: [] for grade in GRADE_VALUES}
    support = {grade: 0 for grade in GRADE_VALUES}
    patients = {patient_id(case_id) for case_id in case_ids}
    for case_id in case_ids:
        label_path = labels_dir / f"{case_id}.nii.gz"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing ground-truth label: {label_path}")
        gt_boxes, gt_grades, gt_supervised = lesion_instances(label_path)
        for grade in GRADE_VALUES:
            support[grade] += int(np.sum(gt_supervised & (gt_grades == grade)))
        with (prediction_dir / f"{case_id}_boxes.pkl").open("rb") as file:
            prediction = pickle.load(file)
        if "pred_grade_probs" not in prediction:
            raise ValueError(f"{case_id}: prediction has no pred_grade_probs")
        case_generic, case_grade = harmonized_froc_events(
            prediction["pred_boxes"],
            prediction["pred_scores"],
            prediction["pred_grade_probs"],
            gt_boxes,
            gt_grades,
            gt_supervised,
            iou_threshold=iou_threshold,
        )
        generic_events.extend({**event, "patient": patient_id(case_id)} for event in case_generic)
        for grade in GRADE_VALUES:
            grade_events[grade].extend(
                {**event, "patient": patient_id(case_id)} for event in case_grade[grade]
            )

    summary = summarize_harmonized_froc(
        generic_events,
        grade_events,
        support,
        num_patients=len(patients),
        num_cases=len(case_ids),
        target_fp_rates=target_fp_rates,
        iou_threshold=iou_threshold,
    )
    return summary, make_events_payload(generic_events, grade_events, support, patients, len(case_ids))


def pool_event_files(event_paths: Sequence[Path]):
    """Pool fold event artifacts without reopening predictions or labels."""
    if not event_paths:
        raise ValueError("At least one event artifact is required")
    generic_events = []
    grade_events = {grade: [] for grade in GRADE_VALUES}
    support = {grade: 0 for grade in GRADE_VALUES}
    patients = set()
    num_cases = 0
    for event_path in event_paths:
        with Path(event_path).open() as file:
            payload = json.load(file)
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported FROC event schema in {event_path}")
        patients.update(str(value) for value in payload.get("patients", []))
        num_cases += int(payload["num_cases"])
        fold_support = _support_by_grade(payload["support"])
        for grade in GRADE_VALUES:
            support[grade] += fold_support[grade]
        generic_events.extend(payload.get("generic_events", []))
        for grade in GRADE_VALUES:
            grade_events[grade].extend(
                payload.get("grade_events", {}).get(_grade_label(grade), [])
            )
    summary = summarize_harmonized_froc(
        generic_events,
        grade_events,
        support,
        num_patients=len(patients),
        num_cases=num_cases,
    )
    return summary, make_events_payload(generic_events, grade_events, support, patients, num_cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite existing output directory: {args.output_dir}")
    summary, payload = pool_event_files(args.events)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "harmonized_froc.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "harmonized_froc_events.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary["records"], indent=2))


if __name__ == "__main__":
    main()
