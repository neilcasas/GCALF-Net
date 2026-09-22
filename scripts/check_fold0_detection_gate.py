#!/usr/bin/env python3
"""Enforce ADR 0005's fold-0 detection safety gate.

The gate compares only the PI-CAI detection endpoint of the final intervention
and the same-wave baseline.  Grade metrics are intentionally not read here.
The 0.02 lesion-AP and 0.03 PI-CAI-score drop thresholds are protocol-locked;
they are written into and validated against the authorization record.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping


MAX_LESION_AP_DROP = 0.02
MAX_PICAI_SCORE_DROP = 0.03
DETECTION_METRICS = ("lesion_ap", "picai_score")


def _train_dir(models_root: Path, task: str, arm: str, fold: int) -> Path:
    return models_root / task / f"RetinaUNetV001_D3V001_3d_{arm}" / f"fold{fold}"


def _read_metrics(path: Path) -> Dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Metrics file has no rows: {path}")
    row = rows[-1]
    values = {}
    for key in ("lesion_ap", "picai_score"):
        try:
            values[key] = float(row[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Metrics file is missing a numeric {key}: {path}") from error
    return values


def _ensure_metrics(models_root: Path, task: str, arm: str, fold: int) -> Dict[str, float]:
    train_dir = _train_dir(models_root, task, arm, fold)
    metrics_path = train_dir / "val_results" / "picai" / "metrics.csv"
    if not metrics_path.is_file():
        model = f"RetinaUNetV001_D3V001_3d_{arm}"
        environment = os.environ.copy()
        environment["det_models"] = str(models_root)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "gcalf_eval.run_eval",
                "--task",
                task,
                "--model",
                model,
                "--fold",
                str(fold),
                "--split",
                "val",
            ],
            check=True,
            env=environment,
        )
    return _read_metrics(metrics_path)


def _record_metrics(record: Mapping[str, Any], name: str) -> Dict[str, float]:
    values = record.get(name)
    if not isinstance(values, dict):
        raise ValueError(f"gate record field {name!r} must be an object")

    result = {}
    for metric in DETECTION_METRICS:
        value = values.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"gate record field {name}.{metric} must be a finite number")
        result[metric] = float(value)
    return result


def validate_gate_record(record_path: Path, task: str) -> None:
    """Validate the immutable fold-0 record used to authorize later folds."""
    try:
        with record_path.open() as file:
            record = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read gate record {record_path}: {error}") from error

    if not isinstance(record, dict):
        raise ValueError("gate record must be a JSON object")
    if record.get("task") != task:
        raise ValueError(f"gate record task is not {task!r}")
    if record.get("fold") != 0:
        raise ValueError("gate record must be for fold 0")
    if record.get("thresholds") != {
        "lesion_ap": MAX_LESION_AP_DROP,
        "picai_score": MAX_PICAI_SCORE_DROP,
    }:
        raise ValueError("gate record thresholds do not match the locked 0.02/0.03 thresholds")

    baseline = _record_metrics(record, "baseline")
    full = _record_metrics(record, "full")
    drops = _record_metrics(record, "drops")
    expected_drops = {
        metric: baseline[metric] - full[metric] for metric in DETECTION_METRICS
    }
    for metric in DETECTION_METRICS:
        if not math.isclose(drops[metric], expected_drops[metric], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"gate record {metric} drop does not match baseline minus full")

    expected_passed = (
        drops["lesion_ap"] <= MAX_LESION_AP_DROP
        and drops["picai_score"] <= MAX_PICAI_SCORE_DROP
    )
    if record.get("passed") is not expected_passed:
        raise ValueError("gate record passed value does not match the locked thresholds")
    if not expected_passed:
        raise ValueError("gate record is not a passing fold-0 gate")


def run_gate(
    task: str,
    models_root: Path,
    fold: int = 0,
) -> bool:
    if fold != 0:
        raise ValueError("The ADR 0005 detection safety gate is defined for fold 0 only")
    baseline = _ensure_metrics(models_root, task, "baseline", fold)
    full = _ensure_metrics(models_root, task, "full", fold)
    drops = {
        "lesion_ap": baseline["lesion_ap"] - full["lesion_ap"],
        "picai_score": baseline["picai_score"] - full["picai_score"],
    }
    passed = (
        drops["lesion_ap"] <= MAX_LESION_AP_DROP
        and drops["picai_score"] <= MAX_PICAI_SCORE_DROP
    )
    record = {
        "task": task,
        "fold": fold,
        "baseline": baseline,
        "full": full,
        "drops": drops,
        "thresholds": {
            "lesion_ap": MAX_LESION_AP_DROP,
            "picai_score": MAX_PICAI_SCORE_DROP,
        },
        "passed": passed,
    }
    record_path = _train_dir(models_root, task, "full", fold) / "fold0_detection_gate.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("w") as file:
        json.dump(record, file, indent=2)
        file.write("\n")

    print(
        f"baseline: lesion_ap={baseline['lesion_ap']:.6f} "
        f"picai_score={baseline['picai_score']:.6f}"
    )
    print(
        f"full:     lesion_ap={full['lesion_ap']:.6f} "
        f"picai_score={full['picai_score']:.6f}"
    )
    print(
        f"drops:    lesion_ap={drops['lesion_ap']:.6f} "
        f"picai_score={drops['picai_score']:.6f}"
    )
    if not passed:
        print(
            "Fold-0 detection safety gate failed. ADR 0005 fallback: stop, relaunch the "
            "locked baseline (ce, transfer off), and report grade as the preregistered null.",
            file=sys.stderr,
        )
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--models-root", type=Path, default=None)
    parser.add_argument(
        "--validate-record",
        type=Path,
        help="validate an existing fold-0 record for authorization of later folds",
    )
    args = parser.parse_args()
    if args.validate_record is not None:
        try:
            validate_gate_record(args.validate_record, args.task)
        except ValueError as error:
            parser.error(str(error))
        print(f"validated passing fold-0 detection gate: {args.validate_record}")
        return

    models_root = args.models_root or Path(os.environ["det_models"])
    if not run_gate(
        args.task,
        models_root,
        fold=args.fold,
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
