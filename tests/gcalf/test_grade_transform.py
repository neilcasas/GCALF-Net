"""Instances2Grades (nndet/io/transforms/instances.py): extracts per-instance
grade/grade_supervised from the batch's `properties` dicts, aligned with
Instances2Boxes's target_classes ordering (ARCHITECTURE.md Sec 8).

This is the transform that finally threads target_grades/target_grade_supervised
into training_step's targets dict -- the piece flagged earlier as missing.
It must also be a no-op (all unsupervised, never a crash) for any task in this
repository whose properties dicts carry no grade metadata at all, since
RetinaUNetModule is shared across every projects/Task0XX_* detection task,
not just GCALF-Net.
"""
import torch

from nndet.io.transforms.instances import Instances2Grades, get_instance_grade_from_properties


def test_get_instance_grade_from_properties_looks_up_grade_and_supervision():
    instance_idx = torch.tensor([1, 2])
    grades = {"1": 3}  # instance 2 is a genuine positive but ungraded (Pooch25/Bosma22a)
    grade_supervised = {"1": True, "2": False}

    grade_values, supervised_values = get_instance_grade_from_properties(instance_idx, grades, grade_supervised)

    assert grade_values.dtype == torch.long
    assert supervised_values.dtype == torch.bool
    assert grade_values.tolist() == [3, 0]  # instance 2's grade is an unused placeholder
    assert supervised_values.tolist() == [True, False]


def test_get_instance_grade_from_properties_defaults_to_unsupervised_when_absent():
    """A task with no grade metadata at all (any non-GCALF projects/Task0XX_*
    task sharing RetinaUNetModule) must not crash -- every instance is simply
    treated as unsupervised, contributing zero grade loss."""
    instance_idx = torch.tensor([1, 2, 3])

    grade_values, supervised_values = get_instance_grade_from_properties(instance_idx, {}, {})

    assert grade_values.tolist() == [0, 0, 0]
    assert supervised_values.tolist() == [False, False, False]


def test_instances2grades_aligns_with_present_instances_per_case():
    transform = Instances2Grades(
        properties_key="properties",
        present_instances="present_instances",
        grade_key="grades",
        grade_supervised_key="grade_supervised",
    )
    data = {
        "properties": [
            {"grades": {"1": 4}, "grade_supervised": {"1": True}},
            {},  # a case with no grade metadata (background patch or non-GCALF task)
        ],
        "present_instances": [
            torch.tensor([1]),
            torch.tensor([], dtype=torch.long),
        ],
    }

    out = transform.forward(**data)

    assert out["grades"][0].tolist() == [4]
    assert out["grade_supervised"][0].tolist() == [True]
    assert out["grades"][1].tolist() == []
    assert out["grade_supervised"][1].tolist() == []
