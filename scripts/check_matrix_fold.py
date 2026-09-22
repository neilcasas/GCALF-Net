"""Infrastructure/correctness gate for one fold of the 20-run study matrix.

This checks that each arm's run *executed correctly* -- it never inspects a
grade metric or any outcome that could motivate a protocol change. Per ADR
0003's Statistical Consequences, endpoints and configuration must not change
once fold results are known; using this gate to decide whether an arm's
*science* looks good would be exactly that. It exists to catch the class of
bug found in the original Fix B diagnostic (a config silently not doing what
its name implied), before it silently consumes a fold's worth of GPU-hours.

Checks, per arm, all pass/fail only on infrastructure grounds:
  1. The run directory and both required checkpoints exist.
  2. ``train.log`` contains no real exception (the batchgenerators
     worker-teardown ``RuntimeError`` at process exit is a known cosmetic
     side effect of the spawn fix and is excluded).
  3. ``config_resolved.yaml`` resolves to the locked protocol values -- the
     exact class of drift that made the original Fix B diagnostic void.
  4. The detection monitor metric is finite and clears a permissive floor
     (catches a collapsed/diverged model; it is not a quality bar).
  5. Wall-clock time is within a generous multiple of the M5-measured
     per-arm throughput (catches an accidental debug-mode/single-thread
     regression, not minor variance).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import yaml

LOCKED_TRAINER_CFG = {
    "grade_class_weight_source": "anchor",
    "grade_freeze_patience": None,
    "grade_checkpoint": "post_swa",
    "max_num_epochs": 50,
    "swa_epochs": 10,
}
LOCKED_MODEL_CFG = {
    "head_grade_kwargs.grade_loss_type": "coral",
}
LOCKED_DATALOADER_KWARGS = {
    "grade_balanced_sampling": False,
    "lesion_transfer_cfg.enabled": True,
    "lesion_transfer_cfg.shuffle_labels": False,
    "lesion_transfer_cfg.min_gland_frac": 0.95,
    "lesion_transfer_cfg.min_zone_frac": 0.50,
}
LOCKED_DATALOADER = "DataLoader{}DLesionTransfer"
SHUFFLED_DATALOADER_KWARGS = {
    **LOCKED_DATALOADER_KWARGS,
    "lesion_transfer_cfg.shuffle_labels": True,
}

# The three non-final detection arms remain the already-locked detection
# controls. The full arm is the single declared CORAL + transfer intervention.
CONTROL_TRAINER_CFG = {
    **LOCKED_TRAINER_CFG,
    "grade_class_weight_source": "lesion",
}
CONTROL_MODEL_CFG = {"head_grade_kwargs.grade_loss_type": "ce"}
CONTROL_DATALOADER_KWARGS = {"grade_balanced_sampling": False}
CONTROL_DATALOADER = "DataLoader{}DOffset"

# Known-cosmetic: batchgenerators worker-teardown race at process exit,
# occurring only after all real work (checkpoints, metrics) is saved.
# See evidence/stage1-mp16-spawn-20260915/ and SITREP-20260916.md.
BENIGN_LOG_PATTERNS = (
    "One or more background workers are no longer alive",
)

MAP_FLOOR = 0.05
WALL_CLOCK_SLACK_FACTOR = 2.0


def _fail(problems, message):
    problems.append(message)


def check_checkpoints(train_dir, problems):
    for name in ("model_best.ckpt", "model_last.ckpt", "model_best_grade.ckpt"):
        if not (train_dir / name).is_file():
            _fail(problems, f"missing checkpoint: {name}")


def check_log_errors(train_dir, problems):
    log_path = train_dir / "train.log"
    if not log_path.is_file():
        _fail(problems, "missing train.log")
        return
    text = log_path.read_text(errors="replace")
    for line in text.splitlines():
        lowered = line.lower()
        if ("traceback" in lowered or "error" in lowered) and not any(
            benign in line for benign in BENIGN_LOG_PATTERNS
        ):
            _fail(problems, f"unexpected error in train.log: {line.strip()[:200]}")


def _nested_get(mapping, dotted_key, default=None):
    value = mapping
    for key in dotted_key.split("."):
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


def check_resolved_config(train_dir, problems, arm=None):
    config_path = train_dir / "config_resolved.yaml"
    if not config_path.is_file():
        _fail(problems, "missing config_resolved.yaml")
        return
    resolved = yaml.safe_load(config_path.read_text())
    trainer_cfg = resolved.get("trainer_cfg", {})
    is_locked_transfer = arm in (None, "full", "shuffled")
    locked_trainer = LOCKED_TRAINER_CFG if is_locked_transfer else CONTROL_TRAINER_CFG
    locked_model = LOCKED_MODEL_CFG if is_locked_transfer else CONTROL_MODEL_CFG
    if arm == "shuffled":
        locked_dataloader = SHUFFLED_DATALOADER_KWARGS
    else:
        locked_dataloader = LOCKED_DATALOADER_KWARGS if is_locked_transfer else CONTROL_DATALOADER_KWARGS
    expected_dataloader = LOCKED_DATALOADER if is_locked_transfer else CONTROL_DATALOADER
    for key, expected in locked_trainer.items():
        actual = trainer_cfg.get(key)
        if actual != expected:
            _fail(problems, f"trainer_cfg.{key} = {actual!r}, expected {expected!r}")
    for key, expected in locked_model.items():
        actual = _nested_get(resolved.get("model_cfg", {}), key)
        if actual != expected:
            _fail(problems, f"model_cfg.{key} = {actual!r}, expected {expected!r}")
    dataloader_kwargs = resolved.get("augment_cfg", {}).get("dataloader_kwargs", {}) or {}
    for key, expected in locked_dataloader.items():
        actual = _nested_get(dataloader_kwargs, key, expected if expected is False else None)
        if actual != expected:
            _fail(problems, f"augment_cfg.dataloader_kwargs.{key} = {actual!r}, expected {expected!r}")
    actual_dataloader = resolved.get("augment_cfg", {}).get("dataloader")
    if actual_dataloader != expected_dataloader:
        _fail(problems, f"augment_cfg.dataloader = {actual_dataloader!r}, expected {expected_dataloader!r}")
    if locked_trainer["grade_class_weight_source"] == "anchor":
        counts = trainer_cfg.get("grade_anchor_class_counts")
        if not isinstance(counts, list) or len(counts) != 4 or any(float(value) <= 0 for value in counts):
            _fail(problems, "trainer_cfg.grade_anchor_class_counts must be four positive measured counts")


def check_detection_metric(train_dir, problems):
    # scripts/train.py's `_evaluate` writes `save_metric_output(scores, curves, save_dir,
    # "results_boxes")` with `save_dir = training_dir / "val_results"` -- the "boxes"
    # subdirectory it also passes to `evaluate_box_dir` holds only plots (FROC.png etc.),
    # not this JSON.
    scores_path = train_dir / "val_results" / "results_boxes.json"
    if not scores_path.is_file():
        _fail(problems, "missing val_results/results_boxes.json")
        return
    scores = json.loads(scores_path.read_text())
    map_keys = [key for key in scores if key.startswith("mAP_IoU_")]
    if not map_keys:
        _fail(problems, "no mAP_IoU_* key in results_boxes.json")
        return
    value = float(scores[map_keys[0]])
    if not math.isfinite(value):
        _fail(problems, f"{map_keys[0]} is not finite: {value}")
    elif value < MAP_FLOOR:
        _fail(problems, f"{map_keys[0]} = {value:.4f} is below the {MAP_FLOOR} sanity floor")


def check_wall_clock(train_dir, arm, m5_record, problems):
    if m5_record is None:
        return
    log_path = train_dir / "train.log"
    if not log_path.is_file():
        return
    stat = log_path.stat()
    per_step = m5_record["measurement"]["seconds_per_step_by_arm"].get(arm)
    steps_per_run = m5_record["measurement"]["steps_per_run"]
    if per_step is None:
        return
    expected_seconds = per_step * steps_per_run
    ctime, mtime = stat.st_ctime, stat.st_mtime
    elapsed = mtime - ctime
    if elapsed <= 0:
        return
    if elapsed > expected_seconds * WALL_CLOCK_SLACK_FACTOR:
        _fail(
            problems,
            f"wall clock {elapsed / 3600:.2f}h exceeds {WALL_CLOCK_SLACK_FACTOR}x the M5 "
            f"projection ({expected_seconds / 3600:.2f}h) -- possible debug-mode regression",
        )


def check_arm(models_root, task, arm, fold, m5_record):
    train_dir = models_root / task / f"RetinaUNetV001_D3V001_3d_{arm}" / f"fold{fold}"
    problems = []
    if not train_dir.is_dir():
        return [f"{arm}: run directory does not exist: {train_dir}"]
    check_checkpoints(train_dir, problems)
    check_log_errors(train_dir, problems)
    check_resolved_config(train_dir, problems, arm=arm)
    check_detection_metric(train_dir, problems)
    check_wall_clock(train_dir, arm, m5_record, problems)
    return [f"{arm}: {problem}" for problem in problems]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--arms", nargs="+", default=["baseline", "lff", "caf", "full"])
    parser.add_argument("--models-root", type=Path, default=None,
                        help="Defaults to $det_models")
    parser.add_argument("--m5-record", type=Path, default=None,
                        help="Path to evidence/m5-budget-rung.json; skips the wall-clock "
                             "check if omitted")
    args = parser.parse_args()

    import os
    models_root = args.models_root or Path(os.environ["det_models"])
    m5_record = json.loads(args.m5_record.read_text()) if args.m5_record else None

    all_problems = []
    for arm in args.arms:
        all_problems.extend(check_arm(models_root, args.task, arm, args.fold, m5_record))

    if all_problems:
        print(f"FOLD {args.fold} FAILED the infrastructure sanity gate:", file=sys.stderr)
        for problem in all_problems:
            print(f"  - {problem}", file=sys.stderr)
        sys.exit(1)

    print(f"FOLD {args.fold}: all {len(args.arms)} arms passed the infrastructure sanity gate.")


if __name__ == "__main__":
    main()
