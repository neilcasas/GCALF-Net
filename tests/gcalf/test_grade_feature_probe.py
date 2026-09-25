import numpy as np
import pytest

from gcalf_eval.grade_pilot import make_split_manifest, patient_id
from scripts.grade_feature_probe import (
    _fit_and_bootstrap,
    aggregate_feature_readout,
    assign_patient_roles,
    validate_export,
)


def test_validate_export_keeps_feature_labels_masks_and_case_ids_aligned():
    features = np.ones((4, 3), dtype=np.float32)
    grades = np.array([2, 3, 4, 5])
    supervised = np.array([True, True, True, True])
    case_ids = np.array(["10000_1000000", "10001_1000001", "10002_1000002", "10003_1000003"])

    validated = validate_export(features, grades, supervised, case_ids)

    assert [len(value) for value in validated] == [4, 4, 4, 4]
    with pytest.raises(ValueError, match="misaligned"):
        validate_export(features, grades[:-1], supervised, case_ids)


def test_anchor_and_pooled_readouts_group_by_case_and_instance():
    features = np.asarray([[1.0, 0.0], [3.0, 0.0], [5.0, 2.0], [7.0, 2.0]])
    grades = np.asarray([4, 4, 5, 5])
    supervised = np.ones(4, dtype=bool)
    case_ids = np.asarray(["10000_1"] * 4)
    instance_ids = np.asarray([8, 8, 9, 9])
    anchor_ious = np.asarray([0.2, 0.9, 0.8, 0.4])

    anchor = aggregate_feature_readout(
        features, grades, supervised, case_ids, instance_ids, anchor_ious, readout="anchor")
    pooled = aggregate_feature_readout(
        features, grades, supervised, case_ids, instance_ids, anchor_ious, readout="pooled")

    np.testing.assert_array_equal(anchor[0], [[3.0, 0.0], [5.0, 2.0]])
    np.testing.assert_array_equal(pooled[0], [[2.0, 0.0], [6.0, 2.0]])
    np.testing.assert_array_equal(anchor[4], [8, 9])
    np.testing.assert_array_equal(anchor[1], [4, 5])


def test_seed_2026_manifest_keeps_patient_groups_disjoint(tmp_path):
    case_grades = {}
    for patient_number in range(30):
        for grade in (2, 3, 4, 5):
            case_id = f"{patient_number:05d}_{1000000 + patient_number * 10 + grade}"
            case_grades[case_id] = [grade]

    manifest = make_split_manifest(case_grades, tmp_path / "split.json", seed=2026)
    roles = manifest["patients"]
    assigned = [patient for patients in roles.values() for patient in patients]

    assert set(roles) == {"fit", "selection", "calibration"}
    assert len(assigned) == len(set(assigned)) == 30
    assert not (set(roles["fit"]) & (set(roles["selection"]) | set(roles["calibration"])))
    _, row_patients, row_roles = assign_patient_roles(manifest, list(case_grades) * 2)
    assert all(patient_id(case) == row_patients[index]
               for index, case in enumerate(list(case_grades) * 2))
    for patient in set(row_patients.tolist()):
        assert len(set(row_roles[row_patients == patient].tolist())) == 1


def test_patient_bootstrap_is_reproducible():
    rng = np.random.RandomState(17)
    grades = np.tile(np.array([2, 3, 4, 5]), 30)
    features = rng.normal(size=(len(grades), 8))
    features[np.arange(len(grades)), grades - 2] += 2.0
    patients = np.repeat(np.arange(30).astype(str), 4)
    train_mask = np.isin(patients, np.arange(21).astype(str))
    eval_mask = ~train_mask

    first = _fit_and_bootstrap(
        features[train_mask], grades[train_mask], features[eval_mask], grades[eval_mask], patients[eval_mask],
        samples=100)
    second = _fit_and_bootstrap(
        features[train_mask], grades[train_mask], features[eval_mask], grades[eval_mask], patients[eval_mask],
        samples=100)

    assert first[2:5] == second[2:5]
    assert first[4] == 100
