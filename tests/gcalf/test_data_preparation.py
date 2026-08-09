import json
import pickle
from pathlib import Path

import pytest

from gcalf_data.build_labels import GGG_LABELS, parse_lesion_isup, remap_source_label
from gcalf_data.prepare_picai import build_tiny_task, dataset_json, load_splits


def test_source_labels_are_remapped_to_contiguous_foreground_classes():
    assert [remap_source_label(label) for label in (0, 2, 3, 4, 5)] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="Unsupported"):
        remap_source_label(1)


def test_dataset_metadata_describes_four_ggg_foreground_classes():
    metadata = dataset_json("Task2201_PICAI_GGG")
    assert metadata["modality"] == {"0": "T2W", "1": "ADC", "2": "HBV"}
    assert metadata["labels"] == {"0": "background", "1": "GGG2", "2": "GGG3", "3": "GGG4", "4": "GGG5"}
    assert GGG_LABELS == {1: "GGG2", 2: "GGG3", 3: "GGG4", 4: "GGG5"}


def test_marksheet_parser_ignores_ungraded_lesions():
    assert list(parse_lesion_isup("2,N/A,4")) == [2, 4]


def test_load_splits_requires_five_train_validation_folds(tmp_path):
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps([{"train": [], "val": []}]))
    with pytest.raises(ValueError, match="five folds"):
        load_splits(split_path)


def _write_source_case(task_dir: Path, case_id: str, classes):
    images_dir = task_dir / "raw_splitted" / "imagesTr"
    labels_dir = task_dir / "raw_splitted" / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for modality in range(3):
        (images_dir / f"{case_id}_{modality:04d}.nii.gz").write_bytes(b"image")
    (labels_dir / f"{case_id}.nii.gz").write_bytes(b"label")
    instances = {str(index): class_id for index, class_id in enumerate(classes, start=1)}
    (labels_dir / f"{case_id}.json").write_text(json.dumps({"instances": instances}))


def _write_source_task(task_dir: Path):
    task_dir.mkdir()
    (task_dir / "dataset.json").write_text(
        json.dumps(
            {
                "task": "Task2201_PICAI_GGG",
                "name": "source",
                "dim": 3,
                "modalities": {"0": "T2W", "1": "ADC", "2": "HBV"},
                "labels": {"0": "GGG2", "1": "GGG3", "2": "GGG4", "3": "GGG5"},
            }
        )
    )
    for class_id in range(4):
        _write_source_case(task_dir, f"1000{class_id}_100000{class_id}", [class_id])
    _write_source_case(task_dir, "10004_1000004", [0])
    _write_source_case(task_dir, "10005_1000005", [])


def test_build_tiny_task_creates_fixed_holdout_alias_and_split(tmp_path):
    source = tmp_path / "Task2201_PICAI_GGG"
    target = tmp_path / "Task900_PICAI_TINY"
    _write_source_task(source)

    manifest = build_tiny_task(source, target)

    assert manifest["train_cases"] == [f"1000{class_id}_100000{class_id}" for class_id in range(4)]
    assert manifest["validation_cases"] == ["10004_1000004", "10005_1000005"]
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
    source = tmp_path / "Task2201_PICAI_GGG"
    _write_source_task(source)
    missing_grade = source / "raw_splitted" / "labelsTr" / "10003_1000003.json"
    missing_grade.write_text(json.dumps({"instances": {}}))
    with pytest.raises(ValueError, match="GGG5"):
        build_tiny_task(source, tmp_path / "Task900_PICAI_TINY")

    target = tmp_path / "Task901_PICAI_TINY"
    target.mkdir()
    (target / "stale").write_text("stale")
    with pytest.raises(FileExistsError, match="existing task"):
        build_tiny_task(source, target)
