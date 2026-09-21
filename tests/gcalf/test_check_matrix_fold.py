import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_matrix_fold.py"
spec = importlib.util.spec_from_file_location("check_matrix_fold", SCRIPT_PATH)
check_matrix_fold = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = check_matrix_fold
spec.loader.exec_module(check_matrix_fold)


def test_check_detection_metric_reads_val_results_not_val_results_boxes(tmp_path):
    """`scripts/train.py`'s `_evaluate` writes `results_boxes.json` directly under
    `val_results/` (`save_metric_output(scores, curves, save_dir, "results_boxes")` with
    `save_dir = training_dir / "val_results"`); `val_results/boxes/` holds only plots.
    A Fix B' run surfaced this: the gate reported a missing file against a real run that
    had completed successfully, because it looked one directory too deep."""
    train_dir = tmp_path
    (train_dir / "val_results").mkdir()
    (train_dir / "val_results" / "boxes").mkdir()
    (train_dir / "val_results" / "boxes" / "FROC.png").write_bytes(b"not json")
    (train_dir / "val_results" / "results_boxes.json").write_text(
        json.dumps({"mAP_IoU_0.10_0.50_0.05_MaxDet_100": "0.2208818067139275"})
    )

    problems = []
    check_matrix_fold.check_detection_metric(train_dir, problems)

    assert problems == []


def test_check_detection_metric_fails_on_missing_file(tmp_path):
    problems = []
    check_matrix_fold.check_detection_metric(tmp_path, problems)
    assert problems == ["missing val_results/results_boxes.json"]


def test_check_detection_metric_fails_below_map_floor(tmp_path):
    (tmp_path / "val_results").mkdir()
    (tmp_path / "val_results" / "results_boxes.json").write_text(
        json.dumps({"mAP_IoU_0.10_0.50_0.05_MaxDet_100": "0.01"})
    )
    problems = []
    check_matrix_fold.check_detection_metric(tmp_path, problems)
    assert len(problems) == 1 and "sanity floor" in problems[0]


def test_check_resolved_config_locks_the_grade_loss_type(tmp_path):
    resolved = {
        "trainer_cfg": {
            "grade_class_weight_source": "lesion",
            "grade_freeze_patience": None,
            "grade_checkpoint": "post_swa",
            "max_num_epochs": 50,
            "swa_epochs": 10,
        },
        "model_cfg": {"head_grade_kwargs": {"grade_loss_type": "coral"}},
        "augment_cfg": {"dataloader_kwargs": {"grade_balanced_sampling": False}},
    }
    (tmp_path / "config_resolved.yaml").write_text(json.dumps(resolved))

    problems = []
    check_matrix_fold.check_resolved_config(tmp_path, problems)

    assert problems == ["model_cfg.head_grade_kwargs.grade_loss_type = 'coral', expected 'ce'"]
