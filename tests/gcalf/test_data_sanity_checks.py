import json

import numpy as np
import pytest
import SimpleITK as sitk

from gcalf_data.sanity_checks import validate_splits, validate_task


def test_validate_splits_rejects_patient_leakage():
    splits = [
        {"train": ["10000_1000000"], "val": ["10000_1000001"]},
        {"train": ["10000_1000001"], "val": ["10000_1000000"]},
    ]
    with pytest.raises(AssertionError, match="leaks a patient"):
        validate_splits(splits, {"10000_1000000", "10000_1000001"})


def _write_case(images_dir, labels_dir, case_id, instances, grades, grade_sources, grade_supervised, size=(128, 128, 32)):
    image_array = np.zeros((size[2], size[1], size[0]), dtype=np.float32)
    for modality in range(3):
        image = sitk.GetImageFromArray(image_array)
        image.SetSpacing((1.0, 1.0, 3.0))
        sitk.WriteImage(image, str(images_dir / f"{case_id}_{modality:04d}.nii.gz"))

    label_array = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)
    for index, instance_id in enumerate(instances, start=1):
        label_array[0, 0, index - 1] = int(instance_id)
    label_image = sitk.GetImageFromArray(label_array)
    label_image.SetSpacing((1.0, 1.0, 3.0))
    sitk.WriteImage(label_image, str(labels_dir / f"{case_id}.nii.gz"))

    (labels_dir / f"{case_id}.json").write_text(
        json.dumps(
            {
                "instances": instances,
                "grades": grades,
                "grade_sources": grade_sources,
                "grade_supervised": grade_supervised,
            }
        )
    )


def _make_task(tmp_path):
    task_dir = tmp_path / "Task2201_PICAI_csPCa"
    images_dir = task_dir / "raw_splitted" / "imagesTr"
    labels_dir = task_dir / "raw_splitted" / "labelsTr"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (task_dir / "dataset.json").write_text(
        json.dumps({"labels": {"0": "csPCa"}, "modalities": {"0": "T2W", "1": "ADC", "2": "HBV"}})
    )
    return task_dir, images_dir, labels_dir


def test_validate_task_accepts_well_formed_single_class_task(tmp_path):
    task_dir, images_dir, labels_dir = _make_task(tmp_path)
    _write_case(
        images_dir, labels_dir, "10000_1000000",
        instances={"1": 0}, grades={"1": 3}, grade_sources={"1": "human_expert_mask"},
        grade_supervised={"1": True},
    )
    _write_case(
        images_dir, labels_dir, "10001_1000001",
        instances={"1": 0}, grades={}, grade_sources={}, grade_supervised={"1": False},
    )

    grade_counts, ungraded, cases = validate_task(task_dir)

    assert grade_counts == {3: 1}
    assert ungraded == 1
    assert cases == ["10000_1000000", "10001_1000001"]


def test_validate_task_rejects_nonzero_detection_class(tmp_path):
    task_dir, images_dir, labels_dir = _make_task(tmp_path)
    _write_case(
        images_dir, labels_dir, "10000_1000000",
        instances={"1": 1}, grades={"1": 3}, grade_sources={"1": "human_expert_mask"},
        grade_supervised={"1": True},
    )
    with pytest.raises(AssertionError, match="Non-zero detection class"):
        validate_task(task_dir)


def test_validate_task_rejects_grade_present_despite_unsupervised(tmp_path):
    task_dir, images_dir, labels_dir = _make_task(tmp_path)
    _write_case(
        images_dir, labels_dir, "10000_1000000",
        instances={"1": 0}, grades={"1": 3}, grade_sources={"1": "human_expert_mask"},
        grade_supervised={"1": False},
    )
    with pytest.raises(AssertionError, match="grade present despite"):
        validate_task(task_dir)
