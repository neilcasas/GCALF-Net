#!/usr/bin/env python3
"""Poll the protected Task2202 matrix and evaluate completed folds automatically.

The watcher reads only model/checkpoint/prediction artifacts below ``--models-root``.
All evaluator and pooling outputs, logs, and the watcher lock live below
``--evidence-root``. It is intentionally Task2202-specific so a path typo cannot
silently move this workflow onto the protected Task2201 study.
"""

import argparse
import fcntl
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TASK = "Task2202_PICAI_csPCa"
MODEL_PREFIX = "RetinaUNetV001_D3V001_3d_"
ARMS = ("baseline", "lff", "caf", "full")
FOLDS = tuple(range(5))


def _train_dir(models_root: Path, task: str, arm: str, fold: int) -> Path:
    return models_root / task / f"{MODEL_PREFIX}{arm}" / f"fold{fold}"


def _case_ids(train_dir: Path, fold: int) -> Tuple[List[str], List[str]]:
    splits_path = train_dir / "splits.pkl"
    prediction_dir = train_dir / "val_predictions"
    if not splits_path.is_file():
        return [], []
    try:
        with splits_path.open("rb") as file:
            splits = pickle.load(file)
        expected = sorted(str(case_id) for case_id in splits[fold]["val"])
    except (IndexError, KeyError, TypeError, OSError, pickle.UnpicklingError):
        return [], []
    found = sorted(path.name[:-10] for path in prediction_dir.glob("*_boxes.pkl"))
    return expected, found


def fold_arm_status(models_root: Path, task: str, arm: str, fold: int) -> Tuple[bool, str]:
    """Return whether one arm has the immutable completion artifacts needed by run_eval."""
    train_dir = _train_dir(models_root, task, arm, fold)
    if not train_dir.is_dir():
        return False, "training directory is absent"
    marker = train_dir / "val_results" / "results_boxes.json"
    if not marker.is_file():
        return False, "val_results/results_boxes.json is absent"
    expected, found = _case_ids(train_dir, fold)
    if not expected:
        return False, "validation split is absent or empty"
    if expected != found:
        return False, f"val_predictions incomplete ({len(found)}/{len(expected)} cases)"
    return True, "ready"


def _evaluation_dir(evidence_root: Path, arm: str, fold: int) -> Path:
    return evidence_root / arm / f"fold{fold}"


def _evaluation_complete(evidence_root: Path, arm: str, fold: int) -> bool:
    output_dir = _evaluation_dir(evidence_root, arm, fold)
    return (output_dir / "harmonized_froc.json").is_file() and (
        output_dir / "harmonized_froc_events.json"
    ).is_file()


def _run_command(command: Sequence[str], repo_root: Path, environment: Dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        timestamp = datetime.now(timezone.utc).isoformat()
        log.write(f"\n[{timestamp}] {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(command, cwd=repo_root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"[{datetime.now(timezone.utc).isoformat()}] exit_code={result.returncode}\n")
    return result.returncode


def evaluate_arm(
    repo_root: Path,
    models_root: Path,
    evidence_root: Path,
    task: str,
    arm: str,
    fold: int,
) -> bool:
    """Run one CPU-only evaluation, writing only under the evidence root."""
    output_dir = _evaluation_dir(evidence_root, arm, fold)
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["det_models"] = str(models_root)
    command = [
        "nice",
        "-n",
        "19",
        sys.executable,
        "-m",
        "gcalf_eval.run_eval",
        "--task",
        task,
        "--model",
        f"{MODEL_PREFIX}{arm}",
        "--fold",
        str(fold),
        "--split",
        "val",
        "--output-dir",
        str(output_dir),
    ]
    log_path = output_dir / "watcher_run.log"
    print(f"evaluating {arm} fold{fold} -> {output_dir}", flush=True)
    exit_code = _run_command(command, repo_root, environment, log_path)
    if exit_code != 0:
        print(f"evaluation failed for {arm} fold{fold}; see {log_path}", file=sys.stderr, flush=True)
        return False
    if not _evaluation_complete(evidence_root, arm, fold):
        print(
            f"evaluation exited successfully but did not write both FROC artifacts for {arm} fold{fold}",
            file=sys.stderr,
            flush=True,
        )
        return False
    return True


def pool_arm(
    repo_root: Path,
    evidence_root: Path,
    arm: str,
    folds: Iterable[int],
) -> bool:
    """Pool one arm's completed fold events without touching model artifacts."""
    output_dir = evidence_root / "pooled" / arm
    if output_dir.exists():
        return (
            (output_dir / "harmonized_froc.json").is_file()
            and (output_dir / "harmonized_froc_events.json").is_file()
        )
    event_paths = [
        _evaluation_dir(evidence_root, arm, fold) / "harmonized_froc_events.json"
        for fold in folds
    ]
    if not all(path.is_file() for path in event_paths):
        return False
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    command = ["nice", "-n", "19", sys.executable, "-m", "gcalf_eval.harmonized_froc", "--events"]
    command.extend(str(path) for path in event_paths)
    command.extend(("--output-dir", str(output_dir)))
    print(f"pooling {arm} -> {output_dir}", flush=True)
    log_path = evidence_root / "pooled" / f"{arm}.log"
    return _run_command(command, repo_root, environment, log_path) == 0 and (
        output_dir / "harmonized_froc.json"
    ).is_file()


def _all_fold_arms_ready(models_root: Path, task: str, arms: Sequence[str], fold: int) -> Tuple[bool, str]:
    statuses = [fold_arm_status(models_root, task, arm, fold) for arm in arms]
    if all(ready for ready, _ in statuses):
        return True, "ready"
    reason = "; ".join(f"{arm}: {status}" for arm, (ready, status) in zip(arms, statuses) if not ready)
    return False, reason


def poll_once(
    repo_root: Path,
    models_root: Path,
    evidence_root: Path,
    task: str,
    arms: Sequence[str],
    folds: Sequence[int],
    pool_after_all_folds: bool,
) -> str:
    """Process every newly complete fold and return a stable waiting/status message."""
    waiting = []
    for fold in folds:
        ready, reason = _all_fold_arms_ready(models_root, task, arms, fold)
        if not ready:
            waiting.append(f"fold{fold}: {reason}")
            continue
        for arm in arms:
            if not _evaluation_complete(evidence_root, arm, fold):
                evaluate_arm(repo_root, models_root, evidence_root, task, arm, fold)

    if pool_after_all_folds:
        evaluations_complete = all(
            _evaluation_complete(evidence_root, arm, fold)
            for arm in arms
            for fold in folds
        )
        if evaluations_complete:
            for arm in arms:
                pool_arm(repo_root, evidence_root, arm, folds)

    if waiting:
        return "waiting: " + " | ".join(waiting)
    return "all configured folds have complete matrix artifacts"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True, help="isolated checkout containing this code")
    parser.add_argument("--models-root", type=Path, required=True, help="read-only det_models root")
    parser.add_argument("--evidence-root", type=Path, required=True, help="write-only evaluation evidence root")
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(FOLDS))
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--once", action="store_true", help="poll once and exit")
    parser.add_argument("--no-pool", action="store_true", help="do not pool after all configured folds finish")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.task != TASK:
        _parser().error(f"this watcher is locked to {TASK}")
    if args.poll_seconds <= 0:
        _parser().error("--poll-seconds must be positive")
    if any(fold not in FOLDS for fold in args.folds) or len(set(args.folds)) != len(args.folds):
        _parser().error("--folds must be unique values from 0 through 4")
    repo_root = args.repo_root.resolve()
    models_root = args.models_root.resolve()
    evidence_root = args.evidence_root.resolve()
    if not repo_root.is_dir() or not models_root.is_dir():
        _parser().error("--repo-root and --models-root must be existing directories")
    try:
        evidence_root.relative_to(models_root)
    except ValueError:
        pass
    else:
        _parser().error("--evidence-root must not be inside the protected --models-root")

    evidence_root.mkdir(parents=True, exist_ok=True)
    lock_path = evidence_root / ".harmonized_froc_watcher.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            _parser().error(f"another watcher already owns {lock_path}")
        last_status = None
        while True:
            status = poll_once(
                repo_root,
                models_root,
                evidence_root,
                args.task,
                args.arms,
                args.folds,
                not args.no_pool,
            )
            if status != last_status:
                print(status, flush=True)
                last_status = status
            if args.once:
                return
            time.sleep(min(args.poll_seconds, 60.0))


if __name__ == "__main__":
    main()
