import json
import pickle
from pathlib import Path

import pytest

from gcalf_data.prepare_picai import build_tiny_task, dataset_json, load_splits


def test_dataset_metadata_declares_single_cspca_foreground_class():
    metadata = dataset_json("Task2201_PICAI_csPCa")
    assert metadata["modality"] == {"0": "T2W", "1": "ADC", "2": "HBV"}
    assert metadata["labels"] == {"0": "background", "1": "csPCa"}


def test_load_splits_requires_five_train_validation_folds(tmp_path):
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps([{"train": [], "val": []}]))
    with pytest.raises(ValueError, match="five folds"):
        load_splits(split_path)


def _write_source_case(task_dir: Path, case_id: str, grades):
    """Write a synthetic case with one instance per entry in `grades`
    (an int 2-5 for a grade-supervised instance, or None for an ungraded
    positive instance). An empty list writes a benign (zero-instance) case.
    """
    images_dir = task_dir / "raw_splitted" / "imagesTr"
    labels_dir = task_dir / "raw_splitted" / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for modality in range(3):
        (images_dir / f"{case_id}_{modality:04d}.nii.gz").write_bytes(b"image")
    (labels_dir / f"{case_id}.nii.gz").write_bytes(b"label")

    instance_ids = [str(index) for index in range(1, len(grades) + 1)]
    instances = {instance_id: 0 for instance_id in instance_ids}
    grade_by_id = {
        instance_id: grade for instance_id, grade in zip(instance_ids, grades) if grade is not None
    }
    grade_supervised = {instance_id: instance_id in grade_by_id for instance_id in instance_ids}
    grade_sources = {instance_id: "human_expert_mask" for instance_id in grade_by_id}
    (labels_dir / f"{case_id}.json").write_text(
        json.dumps(
            {
                "instances": instances,
                "grades": grade_by_id,
                "grade_sources": grade_sources,
                "grade_supervised": grade_supervised,
            }
        )
    )


def _write_source_task(task_dir: Path):
    task_dir.mkdir()
    (task_dir / "dataset.json").write_text(
        json.dumps(
            {
                "task": "Task2201_PICAI_csPCa",
                "name": "source",
                "dim": 3,
                "modalities": {"0": "T2W", "1": "ADC", "2": "HBV"},
                "labels": {"0": "csPCa"},
            }
        )
    )
    for grade in (2, 3, 4, 5):
        _write_source_case(task_dir, f"1000{grade}_100000{grade}", [grade])
    _write_source_case(task_dir, "10006_1000006", [None])  # ungraded positive (unresolved Pooch25)
    _write_source_case(task_dir, "10007_1000007", [])  # benign


def test_build_tiny_task_creates_fixed_holdout_alias_and_split(tmp_path):
    source = tmp_path / "Task2201_PICAI_csPCa"
    target = tmp_path / "Task900_PICAI_TINY"
    _write_source_task(source)

    manifest = build_tiny_task(source, target)

    assert manifest["train_cases"] == [f"1000{grade}_100000{grade}" for grade in (2, 3, 4, 5)]
    assert manifest["validation_cases"] == ["10006_1000006", "10007_1000007"]
    assert manifest["test_cases"] == manifest["validation_cases"]
    metadata = json.loads((target / "dataset.json").read_text())
    assert metadata["task"] == "Task900_PICAI_TINY"
    assert metadata["test_labels"] is True
    for case_id in manifest["validation_cases"]:
        assert (target / "raw_splitted" / "imagesTs" / f"{case_id}_0000.nii.gz").is_file()
        assert (target / "raw_splitted" / "labelsTs" / f"{case_id}.json").is_file()
    with (target / "preprocessed" / "splits_final.pkl").open("rb") as file:
        assert pickle.load(file) == [{"train": manifest["train_cases"], "val": manifest["validation_cases"]}]


def test_build_tiny_task_rejects_missing_grade_and_existing_target(tmp_path):
    source = tmp_path / "Task2201_PICAI_csPCa"
    _write_source_task(source)
    missing_grade = source / "raw_splitted" / "labelsTr" / "10005_1000005.json"
    missing_grade.write_text(
        json.dumps({"instances": {}, "grades": {}, "grade_sources": {}, "grade_supervised": {}})
    )
    with pytest.raises(ValueError, match="GGG5"):
        build_tiny_task(source, tmp_path / "Task900_PICAI_TINY")

    target = tmp_path / "Task901_PICAI_TINY"
    target.mkdir()
    (target / "stale").write_text("stale")
    with pytest.raises(FileExistsError, match="existing task"):
        build_tiny_task(source, target)


def test_build_tiny_task_requires_a_grade_unsupervised_positive(tmp_path):
    source = tmp_path / "Task2201_PICAI_csPCa"
    _write_source_task(source)
    ungraded = source / "raw_splitted" / "labelsTr" / "10006_1000006.json"
    ungraded.write_text(json.dumps({
        "instances": {"1": 0},
        "grades": {"1": 2},
        "grade_sources": {"1": "human_expert_mask"},
        "grade_supervised": {"1": True},
    }))
    with pytest.raises(ValueError, match="grade-unsupervised positive"):
        build_tiny_task(source, tmp_path / "Task902_PICAI_TINY")
