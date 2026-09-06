import json
import pickle

import numpy as np
import pytest
import SimpleITK as sitk

from gcalf_data.sanity_checks import (
    validate_marksheet_positive_cases,
    validate_plan,
    validate_splits,
    validate_task,
)


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

    grade_counts, ungraded, positive_cases, cases = validate_task(task_dir)

    assert grade_counts == {3: 1}
    assert ungraded == 1
    assert positive_cases == 2
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


def test_validate_marksheet_positive_cases_rejects_an_empty_retained_label(tmp_path):
    task_dir, images_dir, labels_dir = _make_task(tmp_path)
    _write_case(
        images_dir, labels_dir, "10000_1000000",
        instances={}, grades={}, grade_sources={}, grade_supervised={},
    )
    marksheet = tmp_path / "marksheet.csv"
    marksheet.write_text("patient_id,study_id,case_ISUP\n10000,1000000,2\n")

    with pytest.raises(AssertionError, match="marksheet-positive"):
        validate_marksheet_positive_cases(task_dir, marksheet, exclusions=set())


def test_validate_plan_requires_three_modalities_and_five_encoder_levels(tmp_path):
    plan_path = tmp_path / "D3V001_3d.pkl"
    plan = {
        "patch_size": [32, 128, 128],
        "target_spacing": [3.0, 0.5, 0.5],
        "normalization_schemes": {0: "nonCT", 1: "nonCT", 2: "nonCT"},
        "use_mask_for_norm": {0: False, 1: False, 2: False},
        "architecture": {
            "in_channels": 3,
            "classifier_classes": 1,
            "conv_kernels": [(3, 3, 3)] * 5,
            "strides": [(2, 2, 2)] * 4,
        },
    }
    with plan_path.open("wb") as file:
        pickle.dump(plan, file)

    assert validate_plan(plan_path)["architecture"]["classifier_classes"] == 1

    plan["architecture"]["conv_kernels"] = [(3, 3, 3)] * 6
    with plan_path.open("wb") as file:
        pickle.dump(plan, file)
    with pytest.raises(AssertionError, match="five-level"):
        validate_plan(plan_path)
