import pickle

from scripts.watch_harmonized_froc import fold_arm_status, poll_once


TASK = "Task2202_PICAI_csPCa"


def _training_dir(root, arm="baseline", fold=0):
    return root / TASK / f"RetinaUNetV001_D3V001_3d_{arm}" / f"fold{fold}"


def _write_split(train_dir, cases):
    train_dir.mkdir(parents=True)
    with (train_dir / "splits.pkl").open("wb") as file:
        pickle.dump([{"val": cases}], file)


def test_fold_status_requires_marker_and_exact_validation_prediction_set(tmp_path):
    models = tmp_path / "det_models"
    train_dir = _training_dir(models)
    _write_split(train_dir, ["patient_a", "patient_b"])
    (train_dir / "val_results").mkdir()
    (train_dir / "val_results" / "results_boxes.json").write_text("{}")
    (train_dir / "val_predictions").mkdir()
    (train_dir / "val_predictions" / "patient_a_boxes.pkl").write_bytes(b"x")

    ready, reason = fold_arm_status(models, TASK, "baseline", 0)
    assert ready is False
    assert "incomplete" in reason

    (train_dir / "val_predictions" / "patient_b_boxes.pkl").write_bytes(b"x")
    ready, reason = fold_arm_status(models, TASK, "baseline", 0)
    assert ready is True
    assert reason == "ready"

    (train_dir / "val_results" / "results_boxes.json").unlink()
    ready, reason = fold_arm_status(models, TASK, "baseline", 0)
    assert ready is False
    assert "absent" in reason


def test_poll_once_waits_for_all_four_arms_before_launching_evaluation(tmp_path):
    models = tmp_path / "det_models"
    evidence = tmp_path / "evidence"
    train_dir = _training_dir(models, "baseline")
    _write_split(train_dir, ["patient_a"])
    (train_dir / "val_results").mkdir()
    (train_dir / "val_results" / "results_boxes.json").write_text("{}")
    (train_dir / "val_predictions").mkdir()
    (train_dir / "val_predictions" / "patient_a_boxes.pkl").write_bytes(b"x")

    status = poll_once(
        tmp_path,
        models,
        evidence,
        TASK,
        ("baseline", "lff", "caf", "full"),
        (0,),
        pool_after_all_folds=False,
    )

    assert "fold0" in status
    assert not evidence.exists() or not list(evidence.rglob("watcher_run.log"))
